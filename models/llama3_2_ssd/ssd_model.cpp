// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "ssd_model.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <variant>

#include "llm/llm_utils.h"
#include "logging.h"
#include "utils.h"

namespace geniex {

SSDModel::SSDModel(LLMSpec spec, SSDConfig ssd_cfg, ParsedGenieConfig gc)
    : LLMModel(std::move(spec), std::move(gc)), ssd_cfg_(std::move(ssd_cfg)) {
    draft_levels_  = ssd_cfg_.branches.size();
    attention_map_ = genAttentionMap();
    // One decode pass submits the whole tree: draft nodes plus their forecast
    // chains. Tell the loader so it selects an AR variant wide enough (30 -> ar32)
    // instead of the narrowest the bundle happens to ship (ar1).
    spec_.min_decode_seq_len = attention_map_.size();
}

bool SSDModel::onInitialized() {
    if (!LLMModel::onInitialized()) return false;

    // head_dim is only known after the base class inferred the spec from graphs.
    if (spec_.head_dim == 0) {
        GENIEX_LOG_ERROR("SSDModel: head_dim was not inferred from the graphs; RoPE would be empty");
        return false;
    }
    rope_.emplace();
    rope_->build(spec_.head_dim, gc_, ssd_cfg_.rope_theta);
    // Pre-cache KV tensor pointers and layout to avoid repeated graph lookups during generation.
    {
        const auto& kv_block = spec_.state_blocks[kv_state_block_idx_];
        kv_tensor_cache_.clear();

        for (size_t s = 0; s < spec_.shards.size(); ++s) {
            const auto& pairs = kv_block.shard_pairs[s];
            if (pairs.empty()) continue;

            const size_t gi = graphIndex(1, s, active_cl_idx_);
            Graph&       g  = graph(gi);

            for (const auto& pair : pairs) {
                KVTensorInfo info;
                info.shard = s;

                const auto& ki_spec  = g.inputSpec(pair.key_in);
                const auto& ko_spec  = g.outputSpec(pair.key_out);
                info.key_in_ptr      = g.inputPtr(pair.key_in);
                info.key_out_ptr     = g.outputPtr(pair.key_out);
                info.key_in_kv_len   = ki_spec.shape[3];
                info.key_out_seq_len = ko_spec.shape[3];
                info.key_elem_size   = ki_spec.elementSize();
                info.key_n_rows      = spec_.num_kv_heads * spec_.head_dim;

                const auto& vi_spec  = g.inputSpec(pair.value_in);
                const auto& vo_spec  = g.outputSpec(pair.value_out);
                info.val_in_ptr      = g.inputPtr(pair.value_in);
                info.val_out_ptr     = g.outputPtr(pair.value_out);
                info.val_in_kv_len   = vi_spec.shape[2];
                info.val_out_seq_len = vo_spec.shape[2];
                info.val_token_size  = spec_.head_dim * vi_spec.elementSize();
                info.val_n_heads     = spec_.num_kv_heads;

                kv_tensor_cache_.push_back(info);
            }
        }
    }

    // Load into the prefill graph (phase=0), not decode. The decode graph uses a
    // larger kv_len stride that gets corrupted when reshapeKV is called during the
    // prefill→decode transition. All phase/CL variants share physical buffers via
    // QNN's shared-buffer mechanism, so loading once into phase=0 is sufficient.
    if (!ssd_cfg_.forecast_prefix_path.empty()) {
        if (!loadForecastPrefix()) {
            fprintf(stderr, "Warning: Failed to load forecast prefix KV cache.\n");
        } else {
            n_past_ = ssd_cfg_.forecast_prefix;
        }
    }

    // Prefill must start writing KV after the pre-loaded forecast prefix entries.
    n_past_ = ssd_cfg_.forecast_prefix;

    return true;
}

std::vector<int32_t> SSDModel::genAttentionMap() {
    std::vector<int32_t> tree = {-1};

    samples_per_draft_level_.clear();
    nodes_per_draft_level_.clear();

    size_t start_idx = 0;
    for (size_t d = 0; d < draft_levels_; ++d) {
        const size_t end_idx      = tree.size();
        const size_t branch_count = ssd_cfg_.branches[d];

        samples_per_draft_level_.push_back(branch_count + 1);

        for (size_t node_idx = start_idx; node_idx < end_idx; ++node_idx) {
            for (size_t c = 0; c < branch_count; ++c) {
                tree.push_back(static_cast<int32_t>(node_idx));
            }
        }

        nodes_per_draft_level_.push_back(tree.size() - end_idx);
        start_idx = end_idx;
    }

    num_draft_nodes_ = tree.size();

    const size_t end_idx = tree.size();
    for (size_t node_idx = 0; node_idx < end_idx; ++node_idx) {
        tree.push_back(static_cast<int32_t>(node_idx));
        for (size_t d = 1; d < draft_levels_; ++d) {
            tree.push_back(static_cast<int32_t>(tree.size() - 1));
        }
    }

    return tree;
}

std::vector<int32_t> SSDModel::genForecastTokens(size_t repeat) const {
    std::vector<int32_t> forecast(draft_levels_);
    std::iota(forecast.begin(), forecast.end(), static_cast<int32_t>(spec_.vocab_size));

    std::vector<int32_t> result;
    result.reserve(repeat * draft_levels_);
    for (size_t i = 0; i < repeat; ++i) result.insert(result.end(), forecast.begin(), forecast.end());
    return result;
}

std::vector<int32_t> SSDModel::topKLogits(const float* logits_row, size_t k) const {
    std::vector<int32_t> indices(spec_.vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    if (k > spec_.vocab_size) k = spec_.vocab_size;
    std::partial_sort(indices.begin(),
        indices.begin() + static_cast<ptrdiff_t>(k),
        indices.end(),
        [logits_row](int32_t a, int32_t b) { return logits_row[a] > logits_row[b]; });
    indices.resize(k);
    return indices;
}

int32_t SSDModel::argmaxLogits(const float* logits_row) const {
    return static_cast<int32_t>(std::max_element(logits_row, logits_row + spec_.vocab_size) - logits_row);
}

std::vector<int32_t> SSDModel::buildSampleTree(int32_t last_token, size_t phase, size_t start_offset) const {
    const size_t         V    = spec_.vocab_size;
    std::vector<int32_t> tree = {last_token};

    size_t               draft_level    = 0;
    size_t               draft_node_idx = 0;
    std::vector<int32_t> samples;
    size_t               sample_idx = 0;

    std::vector<float> row_buf(V);

    for (size_t cur_idx = 1; draft_level < draft_levels_; ++cur_idx) {
        const int32_t parent_idx = attention_map_[cur_idx];

        if (parent_idx != attention_map_[cur_idx - 1]) {
            sample_idx = 0;
        }

        if (draft_node_idx == 0) {
            readLogitsAt(phase, start_offset + draft_level, row_buf.data());
            samples = topKLogits(row_buf.data(), samples_per_draft_level_[draft_level]);
        }

        if (samples[sample_idx] == tree[static_cast<size_t>(parent_idx)]) {
            sample_idx++;
        }
        tree.push_back(samples[sample_idx++]);

        if (++draft_node_idx >= nodes_per_draft_level_[draft_level]) {
            draft_level++;
            draft_node_idx = 0;
        }
    }

    return tree;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> SSDModel::verifyDraftTree(
    const std::vector<int32_t>& draft_tree, size_t phase) const {
    const size_t V = spec_.vocab_size;

    std::vector<float> row_buf(V);

    readLogitsAt(phase, 0, row_buf.data());
    std::vector<int32_t> accepted_ids    = {0};
    std::vector<int32_t> accepted_tokens = {argmaxLogits(row_buf.data())};

    for (int32_t eos_id : spec_.eos_token_ids) {
        if (accepted_tokens.back() == eos_id) return {accepted_tokens, accepted_ids};
    }

    for (size_t cur_idx = 1; cur_idx < num_draft_nodes_; ++cur_idx) {
        const int32_t parent_idx = attention_map_[cur_idx];
        if (parent_idx == accepted_ids.back() && draft_tree[cur_idx] == accepted_tokens.back()) {
            readLogitsAt(phase, cur_idx, row_buf.data());
            int32_t verified = argmaxLogits(row_buf.data());
            accepted_tokens.push_back(verified);
            accepted_ids.push_back(static_cast<int32_t>(cur_idx));

            for (int32_t eos_id : spec_.eos_token_ids) {
                if (verified == eos_id) return {accepted_tokens, accepted_ids};
            }
        }
    }

    return {accepted_tokens, accepted_ids};
}

std::vector<float> SSDModel::buildTreeAttentionMask(
    size_t n_past, size_t num_tokens, size_t seq_len, size_t row_len, size_t new_base, size_t kv_prefix_offset) const {
    const size_t       fp = ssd_cfg_.forecast_prefix;
    std::vector<float> mask(seq_len * row_len, -1e9f);

    // kv-prefix-skip / kv-prefix-offset semantics:
    //   All positions default to skipping the prefix [0, fp).
    //   Positions >= kv_prefix_offset have the skip UNDONE → see [0, n_past).
    //   Positions <  kv_prefix_offset keep the skip       → see [fp, n_past) only.

    for (size_t i = 0; i < num_tokens; ++i) {
        float* row = mask.data() + i * row_len;

        if (i < kv_prefix_offset) {
            for (size_t j = fp; j < n_past; ++j) {
                row[j] = 0.0f;
            }
        } else {
            for (size_t j = 0; j < n_past; ++j) {
                row[j] = 0.0f;
            }
        }

        row[new_base + i] = 0.0f;

        int32_t ancestor = attention_map_[i];
        while (ancestor >= 0) {
            row[new_base + static_cast<size_t>(ancestor)] = 0.0f;
            ancestor                                      = attention_map_[static_cast<size_t>(ancestor)];
        }
    }

    return mask;
}

void SSDModel::readLogitsAt(size_t phase, size_t position, float* dst) const {
    const size_t V          = spec_.vocab_size;
    const size_t last_shard = spec_.shards.size() - 1;
    const size_t g_idx      = graphIndex(phase, last_shard, active_cl_idx_);
    const Graph& g          = graph(g_idx);

    g.read(spec_.shards.back().out_state_name, dst, V, position * V);
}

// Batches consecutive accepted KV positions into single memcpy calls using
// pre-cached tensor pointers, avoiding per-element copies.
void SSDModel::selectiveKVUpdate(const std::vector<bool>& selected, size_t n_accepted) {
    struct CopyRun {
        size_t src_start;
        size_t count;
    };
    std::vector<CopyRun> runs;
    runs.reserve(n_accepted);

    for (size_t pos = 0; pos < selected.size(); ++pos) {
        if (!selected[pos]) continue;
        if (!runs.empty() && runs.back().src_start + runs.back().count == pos) {
            runs.back().count++;
        } else {
            runs.push_back({pos, 1});
        }
    }

    for (const auto& info : kv_tensor_cache_) {
        // Key layout: [H, 1, hd, kv_len] input ← [H, 1, hd, seq_len] output
        {
            auto*        dst        = static_cast<uint8_t*>(info.key_in_ptr);
            const auto*  src        = static_cast<const uint8_t*>(info.key_out_ptr);
            const size_t es         = info.key_elem_size;
            const size_t in_stride  = info.key_in_kv_len;
            const size_t out_stride = info.key_out_seq_len;

            size_t dst_col = n_past_;
            for (const auto& run : runs) {
                const size_t copy_bytes = run.count * es;
                for (size_t row = 0; row < info.key_n_rows; ++row) {
                    std::memcpy(dst + (row * in_stride + dst_col) * es,
                        src + (row * out_stride + run.src_start) * es,
                        copy_bytes);
                }
                dst_col += run.count;
            }
        }

        // Value layout: [H, 1, kv_len, hd] input ← [H, 1, seq_len, hd] output
        {
            auto*        dst        = static_cast<uint8_t*>(info.val_in_ptr);
            const auto*  src        = static_cast<const uint8_t*>(info.val_out_ptr);
            const size_t ts         = info.val_token_size;
            const size_t in_stride  = info.val_in_kv_len;
            const size_t out_stride = info.val_out_seq_len;

            size_t dst_row = n_past_;
            for (const auto& run : runs) {
                const size_t copy_bytes = run.count * ts;
                for (size_t h = 0; h < info.val_n_heads; ++h) {
                    std::memcpy(
                        dst + (h * in_stride + dst_row) * ts, src + (h * out_stride + run.src_start) * ts, copy_bytes);
                }
                dst_row += run.count;
            }
        }
    }
}

std::vector<int32_t> SSDModel::computeTreePositionIds(size_t n_past, size_t num_tokens) const {
    const size_t         fp = ssd_cfg_.forecast_prefix;
    std::vector<int32_t> pos_ids(num_tokens, 0);

    for (size_t i = 0; i < num_tokens; ++i) {
        if (attention_map_[i] < 0) {
            pos_ids[i] = static_cast<int32_t>(n_past - fp);
        } else {
            const size_t parent_idx = static_cast<size_t>(attention_map_[i]);
            pos_ids[i]              = pos_ids[parent_idx] + 1;
        }
    }

    return pos_ids;
}

void SSDModel::Rope::build(size_t head_dim, const ParsedGenieConfig& gc, float theta) {
    // Mirror llm_spec_loader's provider selection so SSD's own position ids go
    // through the same table the graph would otherwise have been fed.
    if (const auto* s = std::get_if<Llama3RopeScaling>(&gc.rope_scaling)) {
        llama3_.emplace(head_dim,
            theta,
            s->factor,
            s->low_freq_factor,
            s->high_freq_factor,
            static_cast<int>(s->original_max_position_embeddings));
        return;
    }
    plain_.emplace(head_dim, theta);
}

std::pair<std::vector<double>, std::vector<double>> SSDModel::Rope::forward(
    const std::vector<int32_t>& position_ids) const {
    if (llama3_) return llama3_->forward(position_ids);
    if (plain_) return plain_->forward(position_ids);
    throw std::runtime_error("SSDModel: RoPE table queried before it was built");
}

SSDModel::Rope& SSDModel::requireRope() {
    if (!rope_) {
        throw std::runtime_error("SSDModel: RoPE table accessed before onInitialized() built it");
    }
    return *rope_;
}

void SSDModel::runShardsWithTreeMask(
    const std::vector<int32_t>& tokens, size_t phase, size_t n_past, size_t kv_prefix_offset) {
    const size_t num_tokens = tokens.size();
    const size_t kv_len     = kvLen(/*phase=*/1, active_cl_idx_);
    const size_t seq_len    = spec_.seq_len_decode;

    auto mask               = buildTreeAttentionMask(n_past,
        num_tokens,
        seq_len,
        kvMaskWidth(/*phase=*/1, active_cl_idx_),
        kvNewBase(/*phase=*/1, active_cl_idx_, n_past),
        kv_prefix_offset);
    auto tree_pos           = computeTreePositionIds(n_past, num_tokens);
    auto [cos_vec, sin_vec] = requireRope().forward(tree_pos);

    const LLMRunContext ctx{tokens, n_past, num_tokens, phase};

    for (size_t s = 0; s < spec_.shards.size(); ++s) {
        const size_t gi = graphIndex(phase, s, active_cl_idx_);
        Graph&       g  = graph(gi);

        if (g.hasInput(spec_.attention_mask_name)) {
            g.write(spec_.attention_mask_name, mask.data(), mask.size());
        }
        writeCacheIndex(g, n_past);

        for (auto& provider : input_providers_) {
            provider->write(g, ctx);
        }

        // Override RoPE with tree-based position IDs (providers write sequential ones first).
        if (g.hasInput("position_ids_cos")) g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
        if (g.hasInput("position_ids_sin")) g.write("position_ids_sin", sin_vec.data(), sin_vec.size());

        TimeLog tl;
        if (!g.execute(tl)) {
            throw std::runtime_error(
                "SSD graph execute failed: phase=" + std::to_string(phase) + " shard=" + std::to_string(s));
        }

        if (s + 1 < spec_.shards.size()) {
            if (phase == 0) {
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            } else {
                applyConnections({decode_shard_hidden_state_[active_cl_idx_][s]});
            }
        }
    }
}

bool SSDModel::loadForecastPrefix() {
    const std::string& path = ssd_cfg_.forecast_prefix_path;
    std::ifstream      file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open forecast prefix file: %s\n", path.c_str());
        return false;
    }

    KVCacheFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || header.magic != 0xC0DE) {
        fprintf(stderr, "Invalid forecast prefix file (bad magic): %s\n", path.c_str());
        return false;
    }

    const size_t n_valid = header.update_size;
    if (n_valid != ssd_cfg_.forecast_prefix) {
        fprintf(stderr,
            "Forecast prefix size mismatch: file has %zu, config expects %zu\n",
            n_valid,
            ssd_cfg_.forecast_prefix);
        return false;
    }

    const size_t n_heads_file = header.n_heads;
    const size_t embed_dim    = header.embed_dim;
    const size_t H            = spec_.num_kv_heads;
    const size_t hd           = spec_.head_dim;

    size_t bytes_per_elem = 1;
    if (header.dtype == 1)
        bytes_per_elem = 2;
    else if (header.dtype == 2)
        bytes_per_elem = 4;

    const auto& kv_block = spec_.state_blocks[kv_state_block_idx_];

    // (shard, key_in, value_in) for every KV slot, in shard-major order — the
    // same order the prefix file was written.
    struct ShardKV {
        size_t              shard;
        const KVTensorPair* pair;
    };
    std::vector<ShardKV> all_kv;
    for (size_t s = 0; s < spec_.shards.size(); ++s) {
        for (const auto& pair : kv_block.shard_pairs[s]) {
            all_kv.push_back({s, &pair});
        }
    }

    // The file stores one tensor per LAYER holding all `n_heads_file` heads
    // (num_tensors = 2 * n_layers, keys then values). An export may instead expose
    // one tensor per (layer, head) -- the SSD w4a16 bundle names them
    // `past_nativekvcache__key_<layer>_head_<h>_in` -- so address the file by
    // (layer, head) rather than reading it sequentially, which would hand each
    // per-head tensor the wrong slice and silently corrupt every forecast logit.
    const size_t n_layers_file = header.num_tensors / 2;
    // Bytes for one head's worth of prefix. Same for keys ([hd][n_valid]) and
    // values ([n_valid][hd]).
    const size_t head_block  = embed_dim * n_valid * bytes_per_elem;
    const size_t layer_block = n_heads_file * head_block;
    const size_t key_base    = sizeof(KVCacheFileHeader);
    const size_t val_base    = key_base + n_layers_file * layer_block;

    // Layer and head index carried by the tensor name: the first integer is the
    // layer, the second (when present) the head.
    auto indicesOf = [](const std::string& name) -> std::pair<size_t, size_t> {
        std::vector<size_t> nums;
        for (size_t i = 0; i < name.size();) {
            if (std::isdigit(static_cast<unsigned char>(name[i]))) {
                size_t v = 0;
                while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) {
                    v = v * 10 + static_cast<size_t>(name[i] - '0');
                    ++i;
                }
                nums.push_back(v);
            } else {
                ++i;
            }
        }
        if (nums.empty()) return {0, 0};
        return {nums[0], nums.size() > 1 ? nums[1] : 0};
    };

    auto loadSection = [&](bool keys) -> bool {
        for (const auto& [shard, pair] : all_kv) {
            const std::string& name = keys ? pair->key_in : pair->value_in;
            const size_t       gi   = graphIndex(0, shard, active_cl_idx_);
            Graph&             g    = graph(gi);
            if (!g.hasInput(name)) continue;

            const auto [layer, head] = indicesOf(name);
            if (layer >= n_layers_file || head >= n_heads_file) {
                fprintf(stderr,
                    "Forecast prefix: tensor '%s' resolves to layer %zu head %zu, outside the file's %zu x %zu\n",
                    name.c_str(),
                    layer,
                    head,
                    n_layers_file,
                    n_heads_file);
                return false;
            }

            const TensorSpec& in_spec = g.inputSpec(name);
            const size_t      es      = in_spec.elementSize();
            // Heads packed into THIS tensor: 1 for a per-head export, n_kv for a
            // per-layer one.
            const size_t heads_here = in_spec.shape[0];
            auto*        dst        = static_cast<uint8_t*>(g.inputPtr(name));

            file.seekg(
                static_cast<std::streamoff>((keys ? key_base : val_base) + layer * layer_block + head * head_block),
                std::ios::beg);

            if (keys) {
                // key_in is [heads, 1, hd, kv_len]: one file row per (head, hd).
                const size_t in_kv_len = in_spec.shape[3];
                const size_t n_rows    = heads_here * hd;
                for (size_t row = 0; row < n_rows; ++row) {
                    file.read(reinterpret_cast<char*>(dst + row * in_kv_len * es),
                        static_cast<std::streamsize>(n_valid * es));
                }
            } else {
                // value_in is [heads, 1, kv_len, hd]: n_valid tokens per head.
                const size_t in_kv_len  = in_spec.shape[2];
                const size_t token_size = hd * es;
                for (size_t h = 0; h < heads_here; ++h) {
                    file.read(reinterpret_cast<char*>(dst + h * in_kv_len * token_size),
                        static_cast<std::streamsize>(n_valid * token_size));
                }
            }
            if (!file) {
                fprintf(stderr, "Forecast prefix: short read on '%s'\n", name.c_str());
                return false;
            }
        }
        return true;
    };

    if (!loadSection(/*keys=*/true) || !loadSection(/*keys=*/false)) return false;

    printf("Loaded forecast prefix: %zu KV entries from %s\n", n_valid, path.c_str());
    return true;
}

void SSDModel::resetKVCache() {
    LLMModel::resetKVCache();

    if (!ssd_cfg_.forecast_prefix_path.empty()) {
        loadForecastPrefix();
    }
    n_past_ = ssd_cfg_.forecast_prefix;
}

std::vector<int32_t> SSDModel::generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
    std::function<bool(int32_t)> token_callback) {
    // Prefill: prompt tokens must not attend to the forecast prefix KV [0, fp).
    const size_t fp               = ssd_cfg_.forecast_prefix;
    size_t       tokens_processed = 0;
    const size_t total_tokens     = prompt_tokens.size();
    size_t       last_chunk_size  = 0;

    while (tokens_processed < total_tokens) {
        const size_t remaining  = total_tokens - tokens_processed;
        const size_t chunk_size = std::min(remaining, spec_.seq_len_prefill);
        last_chunk_size         = chunk_size;

        const std::vector<int32_t> chunk(prompt_tokens.begin() + static_cast<ptrdiff_t>(tokens_processed),
            prompt_tokens.begin() + static_cast<ptrdiff_t>(tokens_processed + chunk_size));

        const size_t prefill_kv_len  = kvLen(/*phase=*/0, active_cl_idx_);
        const size_t prefill_seq_len = spec_.seq_len_prefill;
        const size_t row_len         = kvMaskWidth(/*phase=*/0, active_cl_idx_);
        // Where this chunk's fresh keys sit on the key axis: at the write cursor for
        // a scatter cache, after the cached region for a concat one.
        const size_t new_base = kvNewBase(/*phase=*/0, active_cl_idx_, n_past_);

        // Causal mask that skips forecast prefix [0, fp).
        std::vector<float> mask(prefill_seq_len * row_len, -1e9f);
        for (size_t row = 0; row < chunk_size; ++row) {
            float* row_ptr = mask.data() + row * row_len;
            for (size_t col = fp; col < n_past_; ++col) row_ptr[col] = 0.0f;
            for (size_t col = 0; col <= row; ++col) row_ptr[new_base + col] = 0.0f;
        }

        // Offset position IDs by -fp so prompt starts at position 0.
        const size_t        rope_n_past = n_past_ - fp;
        const LLMRunContext ctx{chunk, n_past_, chunk_size, /*phase=*/0};

        std::vector<int32_t> prefill_pos(chunk_size);
        for (size_t i = 0; i < chunk_size; ++i) prefill_pos[i] = static_cast<int32_t>(rope_n_past + i);
        auto [pf_cos, pf_sin] = requireRope().forward(prefill_pos);

        for (size_t s = 0; s < spec_.shards.size(); ++s) {
            const size_t gi = graphIndex(0, s, active_cl_idx_);
            Graph&       g  = graph(gi);

            if (g.hasInput(spec_.attention_mask_name)) {
                g.write(spec_.attention_mask_name, mask.data(), mask.size());
            }
            writeCacheIndex(g, n_past_);

            for (auto& provider : input_providers_) {
                provider->write(g, ctx);
            }

            // Override RoPE with prefix-adjusted positions.
            if (g.hasInput("position_ids_cos")) g.write("position_ids_cos", pf_cos.data(), pf_cos.size());
            if (g.hasInput("position_ids_sin")) g.write("position_ids_sin", pf_sin.data(), pf_sin.size());

            TimeLog tl;
            if (!g.execute(tl)) {
                throw std::runtime_error("SSD prefill execute failed: shard=" + std::to_string(s));
            }

            updateKV(s, /*phase=*/0, n_past_, chunk_size);
            if (s + 1 < spec_.shards.size()) {
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }
        }

        n_past_ += chunk_size;
        tokens_processed += chunk_size;
    }

    {
        // Under a scatter cache both are the full CL, so this is a no-op.
        const size_t prefill_kv = kvLen(/*phase=*/0, active_cl_idx_);
        const size_t decode_kv  = kvLen(/*phase=*/1, active_cl_idx_);
        for (size_t s = 0; s < spec_.shards.size(); ++s) reshapeKV(s, prefill_kv, decode_kv, n_past_);
    }

    const size_t last_chunk_offset = last_chunk_size - 1;
    int32_t      first_token       = sampleNextToken(/*phase=*/0, last_chunk_offset);

    for (int32_t eos_id : spec_.eos_token_ids) {
        if (first_token == eos_id) {
            const size_t decode_kv  = kvLen(/*phase=*/1, active_cl_idx_);
            const size_t prefill_kv = kvLen(/*phase=*/0, active_cl_idx_);
            for (size_t s = 0; s < spec_.shards.size(); ++s) reshapeKV(s, decode_kv, prefill_kv, n_past_);
            return {};
        }
    }

    std::vector<int32_t> output_tokens;
    output_tokens.push_back(first_token);
    bool user_stop_early = token_callback && !token_callback(first_token);

    if (!user_stop_early) {
        // Initial SSD inference: run [first_token, forecast_0, forecast_1].
        // kv_prefix_offset=1: the real token (pos 0) skips the forecast prefix;
        // forecast tokens (pos >= 1) attend to it.
        {
            std::vector<int32_t> init_tokens = {first_token};
            for (size_t i = 0; i < draft_levels_; ++i) {
                init_tokens.push_back(static_cast<int32_t>(spec_.vocab_size + i));
            }

            runShardsWithTreeMask(init_tokens,
                /*phase=*/1,
                n_past_,
                /*kv_prefix_offset=*/1);

            // Deliberately does NOT commit this pass to the KV cache. Its purpose is
            // only to produce the forecast logits that seed the first draft tree; the
            // real token is the tree ROOT, and the tree pass below both re-processes
            // it and commits it via selectiveKVUpdate (node 0 is always accepted).
            // Committing here too would place the token in the cache twice and shift
            // every subsequent position, which shows up as a dropped token in the
            // output ("Lamas" for "Llamas") -- speculative decode must reproduce plain
            // autoregressive decode exactly.
        }

        int32_t last_accepted_token = first_token;
        auto    draft_tree          = buildSampleTree(last_accepted_token, /*phase=*/1, /*start_offset=*/1);

        const auto   forecast_tokens  = genForecastTokens(num_draft_nodes_);
        const size_t total_ssd_tokens = num_draft_nodes_ + forecast_tokens.size();

        std::vector<int32_t> tokens;
        tokens.reserve(total_ssd_tokens);
        std::vector<bool> selected(total_ssd_tokens, false);

        for (int step = 0; step < gen_cfg.max_tokens; ++step) {
            if (n_past_ + total_ssd_tokens > spec_.context_lengths[active_cl_idx_]) {
                fprintf(stderr,
                    "SSD: Context limit reached (%zu + %zu > %zu)\n",
                    n_past_,
                    total_ssd_tokens,
                    spec_.context_lengths[active_cl_idx_]);
                break;
            }

            tokens.clear();
            tokens.insert(tokens.end(), draft_tree.begin(), draft_tree.end());
            tokens.insert(tokens.end(), forecast_tokens.begin(), forecast_tokens.end());

            // Draft tree positions [0, num_draft_nodes_) skip prefix;
            // forecast positions [num_draft_nodes_, ...) attend to prefix.
            runShardsWithTreeMask(tokens,
                /*phase=*/1,
                n_past_,
                /*kv_prefix_offset=*/num_draft_nodes_);

            auto [accepted_tokens, accepted_ids] = verifyDraftTree(draft_tree, /*phase=*/1);

            std::fill(selected.begin(), selected.end(), false);
            for (int32_t id : accepted_ids) {
                selected[static_cast<size_t>(id)] = true;
            }

            selectiveKVUpdate(selected, accepted_tokens.size());
            n_past_ += accepted_tokens.size();

            bool hit_eos   = false;
            bool user_stop = false;
            for (const int32_t tok : accepted_tokens) {
                for (int32_t eos_id : spec_.eos_token_ids) {
                    if (tok == eos_id) {
                        hit_eos = true;
                        break;
                    }
                }
                if (hit_eos) break;
                output_tokens.push_back(tok);
                if (token_callback && !token_callback(tok)) {
                    user_stop = true;
                    break;
                }
            }

            if (hit_eos || user_stop) break;
            if (static_cast<int>(output_tokens.size()) >= gen_cfg.max_tokens) break;

            const size_t next_draft_offset =
                num_draft_nodes_ + static_cast<size_t>(accepted_ids.back()) * draft_levels_;
            last_accepted_token = accepted_tokens.back();
            draft_tree          = buildSampleTree(last_accepted_token, /*phase=*/1, next_draft_offset);
        }
    }

    // Restore prefill stride so the next call can use AR-128 graphs.
    {
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        for (size_t s = 0; s < spec_.shards.size(); ++s) reshapeKV(s, decode_kv, prefill_kv, n_past_);
    }

    return output_tokens;
}

}  // namespace geniex

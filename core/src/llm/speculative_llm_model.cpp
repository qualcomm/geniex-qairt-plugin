// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/speculative_llm_model.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "llm/input_provider.h"
#include "llm/llm_utils.h"  // RotaryEmbedding
#include "utils.h"          // TimeLog

namespace geniex {

namespace {

// Additive attention mask [seq_len, kv_len + seq_len] for a decode/prefill
// chunk. attention_map[i] = parent index of token i within the chunk (-1 =
// root); empty ⇒ plain causal (token i attends to prior chunk tokens 0..i).
// Every token additionally attends to the whole committed cache [0, n_past).
std::vector<float> buildDecodeAttentionMask(
    const std::vector<int32_t>& attention_map, size_t n_past, size_t num_tokens, size_t seq_len, size_t kv_len) {
    // The mask holds seq_len rows but the loop writes num_tokens of them, so an
    // oversized batch (e.g. a mis-sized EAGLE verify chain) would overrun it.
    if (num_tokens > seq_len)
        throw std::runtime_error("buildDecodeAttentionMask: batch size " + std::to_string(num_tokens) +
                                 " exceeds decode width " + std::to_string(seq_len));
    const size_t       row_len = kv_len + seq_len;
    std::vector<float> mask(seq_len * row_len, -1e9f);

    for (size_t i = 0; i < num_tokens; ++i) {
        float* row = mask.data() + i * row_len;
        for (size_t j = 0; j < n_past; ++j) row[j] = 0.0f;  // committed history
        row[kv_len + i] = 0.0f;                             // self

        if (attention_map.empty()) {
            for (size_t j = 0; j <= i; ++j) row[kv_len + j] = 0.0f;  // causal
        } else {
            int32_t ancestor = attention_map[i];
            while (ancestor >= 0) {
                row[kv_len + static_cast<size_t>(ancestor)] = 0.0f;
                ancestor                                    = attention_map[static_cast<size_t>(ancestor)];
            }
        }
    }
    return mask;
}

// Additive attention mask for a speculative tree KV cache. Rows [0, n_keep) are
// the real committed sequence (always attended); rows [n_keep, n_past) are
// sibling tree branches, attended only when listed in kv_ancestors[i]. Self and
// intra-batch ancestors (attention_map) are attended as in buildDecodeAttentionMask.
std::vector<float> buildTreeAttentionMask(const std::vector<int32_t>& attention_map,
    const std::vector<std::vector<int32_t>>& kv_ancestors, size_t n_keep, size_t n_past, size_t num_tokens,
    size_t seq_len, size_t kv_len) {
    if (num_tokens > seq_len)
        throw std::runtime_error("buildTreeAttentionMask: batch size " + std::to_string(num_tokens) +
                                 " exceeds decode width " + std::to_string(seq_len));
    const size_t       row_len = kv_len + seq_len;
    std::vector<float> mask(seq_len * row_len, -1e9f);

    for (size_t i = 0; i < num_tokens; ++i) {
        float* row = mask.data() + i * row_len;
        for (size_t j = 0; j < n_keep; ++j) row[j] = 0.0f;  // real accepted history
        if (i < kv_ancestors.size())
            for (int32_t a : kv_ancestors[i])
                if (a >= 0 && static_cast<size_t>(a) < n_past) row[static_cast<size_t>(a)] = 0.0f;
        row[kv_len + i] = 0.0f;  // self

        if (!attention_map.empty()) {
            int32_t ancestor = attention_map[i];
            while (ancestor >= 0) {
                row[kv_len + static_cast<size_t>(ancestor)] = 0.0f;
                ancestor                                    = attention_map[static_cast<size_t>(ancestor)];
            }
        }
    }
    return mask;
}

}  // namespace

SpeculativeLLMModel::DecodeBatchResult SpeculativeLLMModel::decodeBatch(const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& pos_ids, const std::vector<int32_t>& attention_map, size_t n_past, float rope_theta,
    const void* feature_override, size_t feature_override_bytes, const std::string& feature_name) {
    const size_t num_tokens = tokens.size();
    const size_t kv_len     = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
    const size_t seq_len    = spec_.seq_len_decode;

    auto mask = buildDecodeAttentionMask(attention_map, n_past, num_tokens, seq_len, kv_len);
    return runDecodeForward(
        tokens, pos_ids, mask, n_past, rope_theta, feature_override, feature_override_bytes, feature_name);
}

SpeculativeLLMModel::DecodeBatchResult SpeculativeLLMModel::decodeBatchTree(const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& pos_ids, const std::vector<int32_t>& attention_map,
    const std::vector<std::vector<int32_t>>& kv_ancestors, size_t n_keep, size_t n_past, float rope_theta,
    const void* feature_override, size_t feature_override_bytes, const std::string& feature_name) {
    const size_t num_tokens = tokens.size();
    const size_t kv_len     = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
    const size_t seq_len    = spec_.seq_len_decode;

    auto mask = buildTreeAttentionMask(attention_map, kv_ancestors, n_keep, n_past, num_tokens, seq_len, kv_len);
    return runDecodeForward(
        tokens, pos_ids, mask, n_past, rope_theta, feature_override, feature_override_bytes, feature_name);
}

SpeculativeLLMModel::DecodeBatchResult SpeculativeLLMModel::runDecodeForward(const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& pos_ids, const std::vector<float>& mask, size_t n_past, float rope_theta,
    const void* feature_override, size_t feature_override_bytes, const std::string& feature_name) {
    const size_t num_tokens = tokens.size();

    RotaryEmbedding rope(spec_.head_dim, rope_theta);
    auto [cos_vec, sin_vec] = rope.forward(pos_ids);

    const LLMRunContext ctx{tokens, n_past, num_tokens, /*phase=*/1};

    for (size_t s = 0; s < shard_count_; ++s) {
        Graph& g = graph(graphIndex(/*phase=*/1, s, active_cl_idx_));

        if (g.hasInput(spec_.attention_mask_name)) {
            g.write(spec_.attention_mask_name, mask.data(), mask.size());
        }

        // Write the cross-engine feature override before the providers so an
        // embedding provider can still fill the token path.
        if (feature_override && !feature_name.empty() && g.hasInput(feature_name)) {
            g.write(feature_name, feature_override, feature_override_bytes);
        }

        for (auto& provider : input_providers_) provider->write(g, ctx);

        // Tree position ids override the sequential ones the RoPE provider wrote.
        if (g.hasInput("position_ids_cos")) g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
        if (g.hasInput("position_ids_sin")) g.write("position_ids_sin", sin_vec.data(), sin_vec.size());

        TimeLog tl;
        if (!g.execute(tl)) {
            throw std::runtime_error("decodeBatch: graph execute failed on shard " + std::to_string(s));
        }

        if (s + 1 < shard_count_) applyConnections({decode_shard_hidden_state_[active_cl_idx_][s]});
    }

    return {/*lm_head_graph=*/graphIndex(1, shard_count_ - 1, active_cl_idx_),
        /*body_graph=*/graphIndex(1, shard_count_ >= 2 ? shard_count_ - 2 : 0, active_cl_idx_),
        /*num_tokens=*/num_tokens};
}

// Copies the selected decode-output KV rows into the KV input buffers starting
// at input row `dst_base`. Does not touch n_past_ (callers own that).
void SpeculativeLLMModel::copyAcceptedKVRows(const std::vector<bool>& selected, size_t dst_base) {
    struct Run {
        size_t src_start;
        size_t count;
    };
    std::vector<Run> runs;
    for (size_t pos = 0; pos < selected.size(); ++pos) {
        if (!selected[pos]) continue;
        if (!runs.empty() && runs.back().src_start + runs.back().count == pos)
            runs.back().count++;
        else
            runs.push_back({pos, 1});
    }

    const auto& kv_block = requireKVStateBlock();
    for (size_t s = 0; s < shard_count_; ++s) {
        if (s >= kv_block.shard_pairs.size()) continue;
        Graph& g = graph(graphIndex(/*phase=*/1, s, active_cl_idx_));
        for (const auto& pair : kv_block.shard_pairs[s]) {
            size_t dst = dst_base;
            for (const auto& run : runs) {
                copyKV(g,
                    pair.key_out,
                    /*src_is_output=*/true,
                    g,
                    pair.key_in,
                    run.src_start,
                    dst,
                    run.count,
                    /*is_key=*/true);
                copyKV(g,
                    pair.value_out,
                    /*src_is_output=*/true,
                    g,
                    pair.value_in,
                    run.src_start,
                    dst,
                    run.count,
                    /*is_key=*/false);
                dst += run.count;
            }
        }
    }
}

void SpeculativeLLMModel::commitDecodeRows(const std::vector<bool>& selected, size_t n_accepted) {
    copyAcceptedKVRows(selected, n_past_);
    n_past_ += n_accepted;
}

void SpeculativeLLMModel::commitDecodeRowsAsync(const std::vector<bool>& selected, size_t n_accepted) {
    if (!decode_pool_) {
        commitDecodeRows(selected, n_accepted);
        return;
    }
    // Advance n_past_ up front so the caller's position math is correct while the
    // copy runs; the copy targets the KV INPUT rows starting at the pre-advance
    // n_past_, which nothing reads until the next decodeBatch (gated by drain).
    const size_t dst_base = n_past_;
    n_past_ += n_accepted;
    decode_pool_->enqueue([this, selected, dst_base] { copyAcceptedKVRows(selected, dst_base); });
}

void SpeculativeLLMModel::switchToDecodeStride() {
    promoteCL(n_past_, spec_.seq_len_decode, spec_.seq_len_prefill);
    const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
    const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
    for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, prefill_kv, decode_kv, n_past_);
}

void SpeculativeLLMModel::switchToPrefillStride() {
    promoteCL(n_past_, spec_.seq_len_prefill, spec_.seq_len_decode);
    const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
    const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
    for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, decode_kv, prefill_kv, n_past_);
}

}  // namespace geniex

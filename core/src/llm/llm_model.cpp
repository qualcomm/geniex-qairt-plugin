// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/llm_model.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "llm/input_provider.h"
#include "logging.h"
#include "utils.h"

namespace geniex {

namespace {
// Encoded "zero" for a KV tensor. For quantized dtypes Genie picks the
// midpoint of the unsigned range (qualla/engines/qnn-htp/KVCache/kvmanager.cpp
// :68-79 — assumes symmetric zero-point at midpoint, matches Qwen2.5-VL /
// Qwen3 exporters). For float dtypes "zero" is the literal 0 byte.
//
// uint8  -> 0x80   (= 1<<7,  zero_point=128)
// uint16 -> 0x8000 (= 1<<15, zero_point=32768) — written via fill_n
// float  -> 0x00
//
// `supported=false` => skip the fill (caller leaves allocator-zero).
struct EncodedZero {
    bool     supported = false;
    bool     wide      = false;  // true => use uint16 fill_n; false => memset
    uint8_t  byte_val  = 0;      // for memset
    uint16_t u16_val   = 0;      // for fill_n when wide
};

EncodedZero encodedZeroForDtype(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UINT_8:
            return {true, false, 0x80, 0};
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_BOOL_8:
            return {true, false, 0x00, 0};
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_UINT_16:
            return {true, true, 0x00, 0x8000};
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_FLOAT_16:
            return {true, true, 0x00, 0x0000};
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            return {true, false, 0x00, 0};
        default:
            return {};
    }
}

// Fill `dst` (n_bytes long) with the encoded-zero pattern for `dt`. No-op for
// unknown dtypes.
void fillEncodedZero(void* dst, size_t n_bytes, Qnn_DataType_t dt) {
    const auto z = encodedZeroForDtype(dt);
    if (!z.supported) return;
    if (z.wide) {
        const size_t n_elems = n_bytes / 2;
        std::fill_n(static_cast<uint16_t*>(dst), n_elems, z.u16_val);
    } else {
        std::memset(dst, z.byte_val, n_bytes);
    }
}
}  // namespace

/*static*/ std::string LLMModel::fmtPattern(const std::string& pattern, size_t layer_idx) {
    std::string       result      = pattern;
    const std::string placeholder = "{}";
    auto              pos         = result.find(placeholder);
    if (pos != std::string::npos) {
        result.replace(pos, placeholder.size(), std::to_string(layer_idx));
    }
    return result;
}

size_t LLMModel::graphIndex(size_t phase, size_t shard, size_t cl_idx) const {
    return phase * (shard_count_ * num_cl_) + shard * num_cl_ + cl_idx;
}

const StateBlockSpec& LLMModel::requireKVStateBlock() const {
    if (kv_state_block_idx_ >= spec_.state_blocks.size()) {
        throw std::runtime_error("KV state block has not been resolved");
    }
    return spec_.state_blocks[kv_state_block_idx_];
}

namespace {

template <typename Fn>
void forEachLayerInRanges(const std::vector<LayerRange>& ranges, Fn&& fn) {
    for (const auto& [begin, end] : ranges) {
        for (size_t layer = begin; layer <= end; ++layer) {
            fn(layer);
        }
    }
}

}  // namespace

LLMModel::LLMModel(LLMSpec spec) : spec_(std::move(spec)) {}

LLMModel::~LLMModel() {
    // Stop workers before the KV buffers they reference are destroyed.
    if (decode_pool_) decode_pool_->stop();
}

bool LLMModel::onInitialized() {
    shard_count_ = spec_.shards.size();
    if (shard_count_ == 0) return false;

    kv_state_block_idx_ = std::numeric_limits<size_t>::max();
    for (size_t block_idx = 0; block_idx < spec_.state_blocks.size(); ++block_idx) {
        if (spec_.state_blocks[block_idx].kind == StateBlockKind::KV) {
            kv_state_block_idx_ = block_idx;
            break;
        }
    }

    const auto& kv_block = requireKVStateBlock();
    if (kv_block.shard_layer_ranges.size() != shard_count_) {
        throw std::runtime_error("KV state block shard_layer_ranges size must match shard count");
    }

    // Discover CL / AR / phase-prefix from the loaded QNN graph names. The
    // regex tolerates an optional alphabetic prefix (Genie's `prompt_` /
    // `token_`, absent on AI Hub IoT exports).
    static const std::regex graph_name_re(R"((?:[A-Za-z]+_)?ar(\d+)_cl(\d+)_(\d+)_of_(\d+))");

    struct ParsedGraph {
        bool   ok    = false;
        size_t ar    = 0;
        size_t cl    = 0;
        size_t shard = 0;
    };
    auto parseGraphName = [&](const std::string& name) -> ParsedGraph {
        std::smatch m;
        if (!std::regex_match(name, m, graph_name_re)) return {};
        try {
            return {true, std::stoul(m[1].str()), std::stoul(m[2].str()), std::stoul(m[3].str())};
        } catch (...) {
            return {};
        }
    };

    // Every loaded graph must parse. An unmatched name would otherwise sort
    // into an arbitrary slot and trip Graph::write much later.
    std::set<size_t> cl_set;
    std::set<size_t> ar_set;
    for (const auto& g : graphs_) {
        const auto p = parseGraphName(g.name());
        if (!p.ok) {
            GENIEX_LOG_ERROR("LLMModel: graph name '{}' does not match '(<phase>_)?arN_clM_S_of_T'", g.name());
            return false;
        }
        cl_set.insert(p.cl);
        ar_set.insert(p.ar);
    }
    if (cl_set.empty() || ar_set.empty()) {
        GENIEX_LOG_ERROR("LLMModel: no graphs loaded");
        return false;
    }
    spec_.context_lengths.assign(cl_set.begin(), cl_set.end());
    spec_.seq_len_prefill = *ar_set.rbegin();
    spec_.seq_len_decode  = *ar_set.begin();
    num_cl_               = spec_.context_lengths.size();

    auto sortKey = [&](const std::string& name) -> std::tuple<int, int, int> {
        const auto p      = parseGraphName(name);
        const int  phase  = (p.ar == spec_.seq_len_prefill) ? 0 : 1;
        const int  shard  = (p.shard > 0) ? static_cast<int>(p.shard) - 1 : 0;
        int        cl_idx = 0;
        for (size_t i = 0; i < spec_.context_lengths.size(); ++i) {
            if (spec_.context_lengths[i] == p.cl) {
                cl_idx = static_cast<int>(i);
                break;
            }
        }
        return {phase, shard, cl_idx};
    };

    std::stable_sort(graphs_.begin(), graphs_.end(), [&](const Graph& a, const Graph& b) {
        return sortKey(a.name()) < sortKey(b.name());
    });
    GENIEX_LOG_DEBUG(
        "LLMModel initialized: {} shards, {} CL variants [{}], vocab={}, hidden={}",
        shard_count_,
        num_cl_,
        [&] {
            std::string s;
            for (size_t i = 0; i < num_cl_; ++i) {
                if (i) s += ',';
                s += std::to_string(spec_.context_lengths[i]);
            }
            return s;
        }(),
        spec_.vocab_size,
        spec_.hidden_size);

    discoverShardTensorNames();
    buildConnections();

    for (auto& p : input_providers_) {
        p->onInitialized(model_cfg_, spec_);
    }

    initKVBuffers();

    // Decode KV-overlap workers; env vars override the ModelConfig defaults.
    unsigned n_workers = model_cfg_.n_decode_workers;
    uint64_t cpu_mask  = model_cfg_.decode_cpu_mask;
    bool     poll      = model_cfg_.decode_poll;
    if (const char* e = std::getenv("GENIEX_DECODE_WORKERS")) n_workers = std::strtoul(e, nullptr, 10);
    if (const char* e = std::getenv("GENIEX_DECODE_CPUMASK")) cpu_mask = std::strtoull(e, nullptr, 16);
    if (const char* e = std::getenv("GENIEX_DECODE_POLL")) poll = (e[0] == '1');

    // Clock keeper: busy-spin threads that keep the CPU cluster from down-clocking
    // across the decode window. The optional GENIEX_CLOCK_KEEPER_THREADS overrides
    // the default (0 = disabled). Shares the decode cpu_mask.
    if (const char* e = std::getenv("GENIEX_CLOCK_KEEPER_THREADS"))
        clock_keeper_threads_ = std::strtoul(e, nullptr, 10);
    decode_cpu_mask_ = cpu_mask;

    // The pool hosts both the KV workers and the clock-keeper spinners, so create
    // it if either is requested.
    if (n_workers > 0 || clock_keeper_threads_ > 0) {
        GENIEX_LOG_DEBUG("decode pool: workers={} cpu_mask={:#x} poll={} clock_keeper={}",
            n_workers,
            cpu_mask,
            poll,
            clock_keeper_threads_);
        decode_pool_ = std::make_unique<ThreadPool>();
        decode_pool_->start(n_workers, cpu_mask, poll);
    }

    return true;
}

void LLMModel::discoverShardTensorNames() {
    // For each shard s:
    //   in_state_name  = first non-special input  on its representative graph
    //   out_state_name = first non-special output on its representative graph
    //   lm_head_only   = no past_key_* inputs and the output is "logits"
    // The representative graph is the prefill (phase=0) variant at the
    // smallest CL — same choice the loader used to make against metadata.json.
    if (spec_.shards.empty()) return;
    for (size_t s = 0; s < spec_.shards.size(); ++s) {
        const size_t gi = graphIndex(0, s, 0);
        const Graph& g  = graph(gi);

        std::string in_name, out_name;
        bool        has_past_key = false;
        for (const auto& t : g.inputSpecs()) {
            if (isKVTensor(t.name)) has_past_key = true;
            if (in_name.empty() && !isSpecialTensor(t.name)) in_name = t.name;
        }
        for (const auto& t : g.outputSpecs()) {
            if (out_name.empty() && !isSpecialTensor(t.name)) out_name = t.name;
        }
        if (in_name.empty() || out_name.empty()) {
            throw std::runtime_error("LLMModel: shard " + std::to_string(s + 1) + " (graph '" + g.name() +
                                     "') has no non-special input/output tensor");
        }
        spec_.shards[s].in_state_name  = in_name;
        spec_.shards[s].out_state_name = out_name;
        spec_.shards[s].lm_head_only   = !has_past_key && out_name == "logits" && s > 0;
    }
}

void LLMModel::buildConnections() {
    shard_hidden_state_.assign(num_cl_, {});
    decode_shard_hidden_state_.assign(num_cl_, {});

    for (size_t cl = 0; cl < num_cl_; ++cl) {
        for (size_t s = 0; s + 1 < shard_count_; ++s) {
            shard_hidden_state_[cl].push_back({static_cast<int>(graphIndex(0, s, cl)),
                spec_.shards[s].out_state_name,
                static_cast<int>(graphIndex(0, s + 1, cl)),
                spec_.shards[s + 1].in_state_name});
        }

        for (size_t s = 0; s + 1 < shard_count_; ++s) {
            decode_shard_hidden_state_[cl].push_back({static_cast<int>(graphIndex(1, s, cl)),
                spec_.shards[s].out_state_name,
                static_cast<int>(graphIndex(1, s + 1, cl)),
                spec_.shards[s + 1].in_state_name});
        }
    }
}

void LLMModel::runShard(size_t shard, size_t phase, size_t cl_idx, const LLMRunContext& ctx) {
    const size_t kv_len = spec_.context_lengths[cl_idx] - (phase == 0 ? spec_.seq_len_prefill : spec_.seq_len_decode);
    const size_t gi     = graphIndex(phase, shard, cl_idx);
    Graph&       g      = graph(gi);

    if (g.hasInput(spec_.attention_mask_name)) {
        const size_t seq_len = (phase == 0) ? spec_.seq_len_prefill : spec_.seq_len_decode;
        auto         mask    = get_attention_mask(ctx.n_past, ctx.curr_len, seq_len, kv_len);
        g.write(spec_.attention_mask_name, mask.data(), mask.size());
    }

    for (auto& provider : input_providers_) {
        provider->write(g, ctx);
    }

    TimeLog tl;
    if (!g.execute(tl)) {
        throw std::runtime_error("Graph execute failed: phase=" + std::to_string(phase) +
                                 " shard=" + std::to_string(shard) + " cl_idx=" + std::to_string(cl_idx) +
                                 " n_past=" + std::to_string(ctx.n_past));
    }
}

void LLMModel::copyKV(Graph& src_g, const std::string& src_name, bool src_is_output, Graph& dst_g,
    const std::string& dst_name, size_t src_off, size_t dst_off, size_t n_tok, bool is_key) {
    const TensorSpec& src_spec  = src_is_output ? src_g.outputSpec(src_name) : src_g.inputSpec(src_name);
    const TensorSpec& dst_spec  = dst_g.inputSpec(dst_name);
    const size_t      elem_size = src_spec.elementSize();

    const auto* src_buf =
        static_cast<const uint8_t*>(src_is_output ? src_g.outputPtr(src_name) : src_g.inputPtr(src_name));
    auto* dst_buf = static_cast<uint8_t*>(dst_g.inputPtr(dst_name));

    // key   [H, 1, head_dim, kv_len]: n_rows = H*head_dim, token_size = elem_size
    // value [H, 1, kv_len, head_dim]: n_rows = H, token_size = head_dim * elem_size
    const size_t H  = spec_.num_kv_heads;
    const size_t hd = spec_.head_dim;
    size_t       num_rows, src_kv_len, dst_kv_len, token_size;
    if (is_key) {
        num_rows   = H * hd;
        src_kv_len = src_spec.shape[3];
        dst_kv_len = dst_spec.shape[3];
        token_size = elem_size;
    } else {
        num_rows   = H;
        src_kv_len = src_spec.shape[2];
        dst_kv_len = dst_spec.shape[2];
        token_size = hd * elem_size;
    }

    for (size_t row = 0; row < num_rows; ++row)
        std::memcpy(dst_buf + (row * dst_kv_len + dst_off) * token_size,
            src_buf + (row * src_kv_len + src_off) * token_size,
            n_tok * token_size);
}

// Propagates freshly-computed KV outputs back into the KV input buffers so each execution sees the full context
// history.
void LLMModel::updateKV(size_t s, size_t phase, size_t dst_off, size_t n_tok) {
    const auto& kv_block     = requireKVStateBlock();
    const auto& layer_ranges = kv_block.shard_layer_ranges[s];
    if (layer_ranges.empty()) return;
    Graph& g = graph(graphIndex(phase, s, active_cl_idx_));
    forEachLayerInRanges(layer_ranges, [&](size_t l) {
        copyKV(g,
            fmtPattern(kv_block.key_out_pattern, l),
            true,
            g,
            fmtPattern(kv_block.key_in_pattern, l),
            0,
            dst_off,
            n_tok,
            true);
        copyKV(g,
            fmtPattern(kv_block.value_out_pattern, l),
            true,
            g,
            fmtPattern(kv_block.value_in_pattern, l),
            0,
            dst_off,
            n_tok,
            false);
    });
}

// Adjusts KV cache stride in-place when switching context-length variants.
// All CL/phase variants share the same physical buffer, so no reallocation is needed.
// Expanding iterates rows backward to avoid overwriting unread data; contracting goes forward.
void LLMModel::reshapeKV(size_t shard, size_t old_kv_len, size_t new_kv_len, size_t n_valid) {
    if (old_kv_len == new_kv_len) return;
    const auto& kv_block     = requireKVStateBlock();
    const auto& layer_ranges = kv_block.shard_layer_ranges[shard];
    if (layer_ranges.empty()) return;

    // Cap copy length: never read past old_kv_len or write past new_kv_len.
    const size_t copy_len = std::min(n_valid, std::min(old_kv_len, new_kv_len));

    Graph& g = graph(graphIndex(0, shard, active_cl_idx_));

    forEachLayerInRanges(layer_ranges, [&](size_t l) {
        const auto key_in = fmtPattern(kv_block.key_in_pattern, l);

        // Key: [num_kv_heads, 1, head_dim, kv_len]
        {
            const TensorSpec& spec      = g.inputSpec(key_in);
            const size_t      elem_size = spec.elementSize();
            const size_t      n_rows    = spec_.num_kv_heads * spec_.head_dim;
            auto*             buf       = static_cast<uint8_t*>(g.inputPtr(key_in));

            if (new_kv_len > old_kv_len) {
                for (ptrdiff_t row = static_cast<ptrdiff_t>(n_rows) - 1; row >= 0; --row) {
                    std::memmove(
                        buf + row * new_kv_len * elem_size, buf + row * old_kv_len * elem_size, copy_len * elem_size);
                    if (copy_len < new_kv_len)
                        fillEncodedZero(buf + (row * new_kv_len + copy_len) * elem_size,
                            (new_kv_len - copy_len) * elem_size,
                            spec.dtype);
                }
            } else {
                for (size_t row = 0; row < n_rows; ++row)
                    std::memmove(
                        buf + row * new_kv_len * elem_size, buf + row * old_kv_len * elem_size, copy_len * elem_size);
            }
        }

        // Value: [num_kv_heads, 1, kv_len, head_dim]
        {
            const auto        val_in     = fmtPattern(kv_block.value_in_pattern, l);
            const TensorSpec& spec       = g.inputSpec(val_in);
            const size_t      elem_size  = spec.elementSize();
            const size_t      n_heads    = spec_.num_kv_heads;
            const size_t      token_size = spec_.head_dim * elem_size;
            auto*             buf        = static_cast<uint8_t*>(g.inputPtr(val_in));

            if (new_kv_len > old_kv_len) {
                for (ptrdiff_t h = static_cast<ptrdiff_t>(n_heads) - 1; h >= 0; --h) {
                    std::memmove(
                        buf + h * new_kv_len * token_size, buf + h * old_kv_len * token_size, copy_len * token_size);
                    if (copy_len < new_kv_len)
                        fillEncodedZero(buf + (h * new_kv_len + copy_len) * token_size,
                            (new_kv_len - copy_len) * token_size,
                            spec.dtype);
                }
            } else {
                for (size_t h = 0; h < n_heads; ++h)
                    std::memmove(
                        buf + h * new_kv_len * token_size, buf + h * old_kv_len * token_size, copy_len * token_size);
            }
        }
    });
}

bool LLMModel::promoteCL(size_t required, size_t capacity_reserved_seq, size_t stride_reserved_seq) {
    if (num_cl_ <= 1) return false;

    size_t new_cl = active_cl_idx_;
    while (new_cl + 1 < num_cl_ && spec_.context_lengths[new_cl] - capacity_reserved_seq < required) {
        ++new_cl;
    }
    if (new_cl == active_cl_idx_) return false;

    GENIEX_LOG_DEBUG("Upgrading CL from {} to {}", active_cl_idx_, new_cl);
    const size_t old_kv = spec_.context_lengths[active_cl_idx_] - stride_reserved_seq;
    const size_t new_kv = spec_.context_lengths[new_cl] - stride_reserved_seq;
    for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, old_kv, new_kv, n_past_);
    active_cl_idx_ = new_cl;
    return true;
}

std::vector<int32_t> LLMModel::generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
    std::function<bool(int32_t)> token_callback) {
    size_t       tokens_processed = 0;
    const size_t total_tokens     = prompt_tokens.size();
    size_t       last_chunk_size  = 0;  // valid token count in the final prefill chunk

    GENIEX_LOG_DEBUG("generate: prompt_tokens={}, n_past={}, max_tokens={}", total_tokens, n_past_, gen_cfg.max_tokens);

    // (Re)build & seed the sampler. No-op when sampling is disabled — in
    // that case `sampler_` stays null and sampleNextToken() takes the greedy
    // argmax fast path.
    prepareSampler(gen_cfg, prompt_tokens);

    // Reject prompts that cannot fit in the largest available context length.
    // context_lengths is sorted ascending, so the last entry is the max CL.
    const size_t max_cl = spec_.context_lengths.back();
    if (n_past_ + total_tokens > max_cl) {
        throw ContextLengthExceededError("geniex: prompt exceeds max context length (" + std::to_string(max_cl) + ")");
    }

    while (tokens_processed < total_tokens) {
        const size_t remaining    = total_tokens - tokens_processed;
        const size_t chunk_size   = std::min(remaining, spec_.seq_len_prefill);
        last_chunk_size           = chunk_size;
        const bool is_final_chunk = (tokens_processed + chunk_size >= total_tokens);

        // Ensure the prefill KV buffer (CL - seq_len_prefill) can hold n_past + chunk_size after this chunk.
        promoteCL(/*required=*/n_past_ + chunk_size,
            /*capacity_reserved_seq=*/spec_.seq_len_prefill,
            /*stride_reserved_seq=*/spec_.seq_len_prefill);

        const std::vector<int32_t> chunk(prompt_tokens.begin() + static_cast<std::ptrdiff_t>(tokens_processed),
            prompt_tokens.begin() + static_cast<std::ptrdiff_t>(tokens_processed + chunk_size));

        GENIEX_LOG_DEBUG("prefill chunk: tokens [{}, {}) cl_idx={} final={}",
            tokens_processed,
            tokens_processed + chunk_size,
            active_cl_idx_,
            is_final_chunk);
        const LLMRunContext ctx{chunk, n_past_, chunk_size, /*phase=*/0};

        for (size_t s = 0; s < shard_count_; ++s) {
            // For non-final prefill chunks we only need the KV cache to be populated, so such shards can be skipped
            // entirely.
            if (!is_final_chunk && spec_.shards[s].lm_head_only) {
                GENIEX_LOG_DEBUG("skipping LM-head-only shard {} on non-final prefill chunk", s);
                continue;
            }
            runShard(s, /*phase=*/0, active_cl_idx_, ctx);
            updateKV(s, /*phase=*/0, n_past_, chunk_size);
            if (s + 1 < shard_count_) {
                if (!is_final_chunk && spec_.shards[s + 1].lm_head_only) {
                    continue;
                }
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }
        }

        n_past_ += chunk_size;
        tokens_processed += chunk_size;
    }

    // Switch KV stride from prefill to decode before entering the decode loop.
    {
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, prefill_kv, decode_kv, n_past_);
    }

    // Prefill output is [seq_len, vocab_size]; the last valid token is at last_chunk_size - 1.
    const size_t         last_chunk_offset = last_chunk_size - 1;
    int32_t              next_token        = sampleNextToken(/*phase=*/0, last_chunk_offset);
    std::vector<int32_t> output_tokens;

    GENIEX_LOG_DEBUG("prefill done: n_past={}, first_token={}", n_past_, next_token);

    // Keep the CPU cluster from down-clocking across the decode loop.
    if (decode_pool_ && clock_keeper_threads_ > 0)
        decode_pool_->startClockKeeper(clock_keeper_threads_, decode_cpu_mask_);

    for (int step = 0; step < gen_cfg.max_tokens; ++step) {
        bool is_eos = false;
        for (int32_t eos_id : spec_.eos_token_ids) {
            if (next_token == eos_id) {
                is_eos = true;
                break;
            }
        }
        if (is_eos) break;
        output_tokens.push_back(next_token);
        if (token_callback && !token_callback(next_token)) {
            GENIEX_LOG_DEBUG("token_callback requested stop at step {}", step);
            break;
        }

        // Stop and report when the next decode step would exceed the largest available CL.
        if (n_past_ + 1 > max_cl) {
            throw ContextLengthExceededError(
                "geniex: generation exceeds max context length (" + std::to_string(max_cl) + ")");
        }

        // KV write-back from the previous step must finish before restriding or
        // re-reading the KV buffers below.
        if (decode_pool_) decode_pool_->wait();

        // Ensure the decode KV buffer (CL - seq_len_decode) has room for the write at offset n_past_.
        promoteCL(/*required=*/n_past_ + 1,
            /*capacity_reserved_seq=*/spec_.seq_len_decode,
            /*stride_reserved_seq=*/spec_.seq_len_decode);

        const LLMRunContext ctx{{next_token}, n_past_, /*curr_len=*/1, /*phase=*/1};

        const size_t kv_dst_off = n_past_;  // n_past_ advances before the jobs run
        for (size_t s = 0; s < shard_count_; ++s) {
            runShard(s, /*phase=*/1, active_cl_idx_, ctx);
            // updateKV(s) feeds nothing in shard s+1's execute, so overlap it with the
            // next runShard; the top-of-step wait() orders it before the next read.
            if (decode_pool_) {
                decode_pool_->enqueue([this, s, kv_dst_off] { updateKV(s, /*phase=*/1, kv_dst_off, /*n_tok=*/1); });
            } else {
                updateKV(s, /*phase=*/1, kv_dst_off, /*n_tok=*/1);
            }
            if (s + 1 < shard_count_) {
                applyConnections({decode_shard_hidden_state_[active_cl_idx_][s]});
            }
        }

        n_past_++;
        next_token = sampleNextToken(/*phase=*/1, /*token_offset=*/0);
    }

    // Drain KV jobs still in flight after an early break (EOS / callback stop).
    if (decode_pool_) decode_pool_->wait();

    // Decode window is over; release the cluster back to the governor.
    if (decode_pool_) decode_pool_->stopClockKeeper();

    // Restore prefill stride so the model is ready for the next generate() call.
    // Promote first so the upcoming decode_kv → prefill_kv reshape doesn't truncate history when n_past_ > prefill_kv.
    promoteCL(/*required=*/n_past_,
        /*capacity_reserved_seq=*/spec_.seq_len_prefill,
        /*stride_reserved_seq=*/spec_.seq_len_decode);
    {
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, decode_kv, prefill_kv, n_past_);
    }

    GENIEX_LOG_DEBUG("generate done: {} output tokens", output_tokens.size());
    return output_tokens;
}

int32_t LLMModel::sampleNextToken(size_t phase, size_t token_offset) {
    std::vector<float> logits;
    readLastLogits(phase, token_offset, logits);

    // Sampler path: run the chain and accept() so penalty / DRY advance.
    if (sampler_) {
        const int32_t tok = sampler_->sample(logits);
        sampler_->accept(tok);
        return tok;
    }

    // Greedy fast path — historical behaviour when sampling is disabled.
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void LLMModel::readLastLogits(size_t phase, size_t token_offset, std::vector<float>& out) const {
    const size_t last_shard = shard_count_ - 1;
    const size_t g_idx      = graphIndex(phase, last_shard, active_cl_idx_);
    const Graph& g          = graph(g_idx);

    out.resize(spec_.vocab_size);
    g.read(spec_.shards.back().out_state_name, out.data(), spec_.vocab_size, token_offset * spec_.vocab_size);
}

namespace {

// True when two configs agree on every field that affects the sampler chain.
bool samplerCfgEqual(const GenerationConfig& a, const GenerationConfig& b) {
    return a.enable_sampling == b.enable_sampling && a.temperature == b.temperature && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.top_k == b.top_k && a.repetition_penalty == b.repetition_penalty &&
           a.presence_penalty == b.presence_penalty && a.frequency_penalty == b.frequency_penalty &&
           a.penalty_last_n == b.penalty_last_n && a.seed == b.seed && a.grammar_str == b.grammar_str &&
           a.grammar_root == b.grammar_root;
}

}  // namespace

void LLMModel::prepareSampler(const GenerationConfig& gen_cfg, const std::vector<int32_t>& prompt_tokens) {
    // Sampling off → drop any cached sampler so sampleNextToken() goes greedy.
    if (!gen_cfg.enable_sampling) {
        sampler_.reset();
        sampler_cfg_valid_ = false;
        return;
    }

    // Reuse iff config is unchanged — keeps penalty / DRY history alive
    // across multi-turn calls. `prompt_tokens` is just the new turn's prompt
    // (prior turns live in the KV cache); we append it to the running
    // sampler history below.
    const bool can_reuse = sampler_ && sampler_cfg_valid_ && samplerCfgEqual(sampler_cfg_, gen_cfg);
    if (!can_reuse) {
        geniex_sampler_params sp;
        sp.seed            = gen_cfg.seed;
        sp.temp            = gen_cfg.temperature;
        sp.top_k           = gen_cfg.top_k;
        sp.top_p           = gen_cfg.top_p;
        sp.min_p           = gen_cfg.min_p;
        sp.penalty_repeat  = gen_cfg.repetition_penalty;
        sp.penalty_freq    = gen_cfg.frequency_penalty;
        sp.penalty_present = gen_cfg.presence_penalty;
        sp.penalty_last_n  = gen_cfg.penalty_last_n;
        sp.no_perf         = true;

        // EOG tokens come from the model spec so Sampler::is_eog() works.
        sp.eog_tokens.assign(spec_.eos_token_ids.begin(), spec_.eos_token_ids.end());

        sampler_ = std::make_unique<Sampler>(sp);

        // Grammar needs a tokenizer (for the vocab interface). The pipeline
        // injects one via gen_cfg.tokenizer; if missing, warn and skip.
        if (!gen_cfg.grammar_str.empty()) {
            if (gen_cfg.tokenizer) {
                try {
                    sampler_->set_grammar(
                        std::make_unique<Grammar>(gen_cfg.grammar_str, *gen_cfg.tokenizer, gen_cfg.grammar_root));
                } catch (const std::exception& e) {
                    GENIEX_LOG_WARN("grammar init failed, continuing without grammar: {}", e.what());
                }
            } else {
                GENIEX_LOG_WARN("grammar_str set but no tokenizer provided — grammar disabled");
            }
        }

        sampler_cfg_       = gen_cfg;
        sampler_cfg_valid_ = true;
    }

    // Append this turn's prompt; subsequent generated tokens are accept()ed
    // inside sampleNextToken().
    if (!prompt_tokens.empty()) {
        sampler_->init(prompt_tokens);
    }
}

std::unordered_set<std::string> LLMModel::buildKVInputNameSet() const {
    std::unordered_set<std::string> names;
    const auto&                     kv_block = requireKVStateBlock();
    for (size_t s = 0; s < shard_count_; ++s) {
        const auto& layer_ranges = kv_block.shard_layer_ranges[s];
        forEachLayerInRanges(layer_ranges, [&](size_t l) {
            names.insert(fmtPattern(kv_block.key_in_pattern, l));
            names.insert(fmtPattern(kv_block.value_in_pattern, l));
        });
    }
    return names;
}

void LLMModel::resetKVCache() {
    n_past_        = 0;
    active_cl_idx_ = 0;

    // Drop sampler state too; penalty / DRY shouldn't leak across resets.
    sampler_.reset();
    sampler_cfg_valid_ = false;

    initKVBuffers();
}

void LLMModel::initKVBuffers() {
    const auto   kv_names     = buildKVInputNameSet();
    const size_t total_graphs = 2 * shard_count_ * num_cl_;
    for (size_t gi = 0; gi < total_graphs; ++gi) {
        Graph& g = graph(gi);
        for (const auto& spec : g.inputSpecs()) {
            if (!kv_names.count(spec.name)) continue;
            void* buf = g.inputPtr(spec.name);
            if (!buf) continue;
            fillEncodedZero(buf, spec.byteCount(), spec.dtype);
        }
    }
}

void LLMModel::saveKVCacheToFile(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveKVCacheToFile: cannot open " + path);

    f.write(reinterpret_cast<const char*>(&n_past_), sizeof(n_past_));
    f.write(reinterpret_cast<const char*>(&active_cl_idx_), sizeof(active_cl_idx_));

    const auto   kv_names       = buildKVInputNameSet();
    const size_t prefill_graphs = shard_count_ * num_cl_;
    for (size_t gi = 0; gi < prefill_graphs; ++gi) {
        const Graph& g = graph(gi);
        for (const auto& spec : g.inputSpecs()) {
            if (kv_names.count(spec.name)) {
                std::vector<uint8_t> buf(spec.byteCount());
                g.read(spec.name, buf.data(), buf.size());
                f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
            }
        }
    }
    if (!f) throw std::runtime_error("saveKVCacheToFile: write error");
}

void LLMModel::loadKVCacheFromFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadKVCacheFromFile: cannot open " + path);

    f.read(reinterpret_cast<char*>(&n_past_), sizeof(n_past_));
    f.read(reinterpret_cast<char*>(&active_cl_idx_), sizeof(active_cl_idx_));

    const auto   kv_names       = buildKVInputNameSet();
    const size_t prefill_graphs = shard_count_ * num_cl_;
    for (size_t gi = 0; gi < prefill_graphs; ++gi) {
        Graph& g = graph(gi);
        for (const auto& spec : g.inputSpecs()) {
            if (kv_names.count(spec.name)) {
                std::vector<uint8_t> buf(spec.byteCount());
                f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
                g.write(spec.name, buf.data(), buf.size());
            }
        }
    }
    if (!f) throw std::runtime_error("loadKVCacheFromFile: read error");
}

size_t LLMModel::nPast() const { return n_past_; }

void LLMModel::addInputProvider(std::unique_ptr<InputProvider> provider) {
    input_providers_.push_back(std::move(provider));
}

}  // namespace geniex

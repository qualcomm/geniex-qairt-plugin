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
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "llm/input_provider.h"
#include "llm/llm_utils.h"  // isKVTensor / isSpecialTensor
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

// True for integer QNN dtypes. A shard's token-id input (`input_ids`) is
// integer; the inter-shard hidden state is always a float/quantized-float
// tensor, so dtype separates the two regardless of tensor name.
bool isIntegerDtype(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_BOOL_8:
            return true;
        default:
            return false;
    }
}

// Compiles `pattern` into a regex whose "{}" placeholder captures an integer.
// allow_head_suffix tolerates an optional "_h<n>" before the trailing suffix,
// matching exports that split KV into one tensor per head.
std::regex patternToRegex(const std::string& pattern, bool allow_head_suffix = false) {
    const std::string ph  = "{}";
    auto              at  = pattern.find(ph);
    auto              esc = [](const std::string& s) {
        static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
        return std::regex_replace(s, special, R"(\$&)");
    };
    if (at == std::string::npos) return std::regex(esc(pattern));
    const std::string head = allow_head_suffix ? R"((?:_h\d+)?)" : "";
    return std::regex(esc(pattern.substr(0, at)) + R"((\d+))" + head + esc(pattern.substr(at + ph.size())));
}
}  // namespace

std::vector<KVTensorPair> LLMModel::discoverKVPairs(const Graph& g, const StateBlockSpec& block) {
    std::vector<KVTensorPair> pairs;

    // Each pattern is "<prefix>{}<suffix>". We match key_out outputs by their
    // static prefix + suffix and capture EVERYTHING in between ("the middle" —
    // the layer index plus any per-head-group infix like "14_h0"), then build
    // the sibling in/value tensor names by reusing that same middle. This is
    // deliberately NOT a "<prefix>(\d+)<suffix>" regex: newer exports (Gemma4
    // W4A16, QAIRT 2.45) name KV tensors `past_key_14_h0_out` — a `_h{group}`
    // infix sits between the layer index and `_out`. A digit-only capture
    // fails to match those, so discoverKVPairs found ZERO pairs, updateKV wrote
    // nothing back, and decode read an empty cache → coherent prefill but
    // garbage decode. Matching prefix+suffix and carrying the middle verbatim
    // handles both the classic (`past_key_5_out`) and infixed layouts.
    auto split = [](const std::string& p) -> std::pair<std::string, std::string> {
        const auto at = p.find("{}");
        if (at == std::string::npos) return {p, std::string{}};
        return {p.substr(0, at), p.substr(at + 2)};
    };
    const auto [ko_pre, ko_suf] = split(block.key_out_pattern);
    const auto [ki_pre, ki_suf] = split(block.key_in_pattern);
    const auto [vo_pre, vo_suf] = split(block.value_out_pattern);
    const auto [vi_pre, vi_suf] = split(block.value_in_pattern);

    for (const auto& t : g.outputSpecs()) {
        const std::string& n = t.name;
        if (n.size() <= ko_pre.size() + ko_suf.size()) continue;
        if (n.compare(0, ko_pre.size(), ko_pre) != 0) continue;
        if (n.compare(n.size() - ko_suf.size(), ko_suf.size(), ko_suf) != 0) continue;
        const std::string middle = n.substr(ko_pre.size(), n.size() - ko_pre.size() - ko_suf.size());
        // The middle must start with the layer index; guards against a shorter
        // prefix accidentally matching an unrelated tensor.
        if (middle.empty() || middle[0] < '0' || middle[0] > '9') continue;

        KVTensorPair p{
            ki_pre + middle + ki_suf, ko_pre + middle + ko_suf, vi_pre + middle + vi_suf, vo_pre + middle + vo_suf};
        // key_out matched by construction; the other three are the block's
        // independently declared patterns, so validate they resolve to real
        // tensors (a mismatched value pattern would silently drop KV state).
        if (!g.hasInput(p.key_in) || !g.hasInput(p.value_in) || !g.hasOutput(p.value_out)) {
            throw std::runtime_error("inferSpecFromGraphs: graph '" + g.name() + "' has key_out '" + p.key_out +
                                     "' but is missing a matching in/value KV tensor");
        }
        pairs.push_back(std::move(p));
    }
    return pairs;
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

LLMModel::LLMModel(LLMSpec spec, ParsedGenieConfig gc) : spec_(std::move(spec)), gc_(std::move(gc)) {}

LLMModel::~LLMModel() {
    // Stop workers before the KV buffers they reference are destroyed.
    if (decode_pool_) decode_pool_->stop();
}

void LLMModel::inferSpecFromGraphs() {
    if (shard_count_ == 0 || num_cl_ == 0) {
        throw std::runtime_error("inferSpecFromGraphs: shard_count and num_cl must be non-zero");
    }
    spec_.shards.assign(shard_count_, ShardSpec{});
    if (spec_.state_blocks.empty()) spec_.state_blocks.push_back(makeKVStateBlock());

    // Gemma3/4 second (sliding-window) KV cache: if any graph exposes swa_key_*
    // outputs and no swa block was pre-declared, add one so its layers get
    // discovered and advanced alongside the global cache.
    {
        bool has_swa = false, swa_declared = false;
        for (const auto& b : spec_.state_blocks)
            if (b.kind == StateBlockKind::SlidingWindowKV) swa_declared = true;
        for (size_t s = 0; s < shard_count_ && !has_swa; ++s) {
            const Graph& g = graph(graphIndex(0, s, 0));
            for (const auto& t : g.outputSpecs())
                if (t.name.rfind("swa_key_", 0) == 0) {
                    has_swa = true;
                    break;
                }
        }
        if (has_swa && !swa_declared) spec_.state_blocks.push_back(makeSwaKVStateBlock());
    }

    for (auto& block : spec_.state_blocks) block.shard_pairs.assign(shard_count_, {});

    for (size_t s = 0; s < shard_count_; ++s) {
        // Representative graph: prefill (phase 0), smallest CL, shard s.
        const Graph& g = graph(graphIndex(0, s, 0));

        // Hidden size is the last dim of the inter-shard hidden-state tensor.
        // That tensor is the shard's first non-special I/O that is a float
        // (rank >= 2) and not the vocab-sized `logits`. Identifying it by role
        // + dtype rather than a fixed name list covers models whose hidden
        // tensors carry arbitrary compiler-assigned names (e.g. `embedding`,
        // `add_82384`) in addition to the canonical `hidden_states` etc.
        auto hiddenDimOf = [](const TensorSpec& t) -> size_t {
            if (t.name == "logits" || t.shape.size() < 2) return 0;
            if (isIntegerDtype(t.dtype)) return 0;  // token-id input, not a hidden state
            return t.shape.back();
        };

        // The inter-shard boundary tensor carries the SAME name on both sides:
        // shard s-1's out_state_name is a verbatim input of shard s (e.g. shard 1
        // OUT add_13690 -> shard 2 IN add_13690). So for s>0 the hidden-state
        // input is the one matching the previous shard's output — NOT merely the
        // "first non-special input." Gemma4 re-feeds inputs_embeds/per_layer_inputs
        // to EVERY shard; inputs_embeds is not in the special list and has the same
        // [1,seq,hidden] shape as the real hidden state, so first-non-special would
        // silently mis-pick it and leave the true boundary tensor unfed (garbage
        // out, no shape error). Match by name to be robust to re-fed streams.
        const std::string& prev_out = (s > 0) ? spec_.shards[s - 1].out_state_name : std::string{};

        std::string in_name, out_name;
        for (const auto& t : g.inputSpecs()) {
            // For s>0, the hidden-state input is the one whose name equals the
            // previous shard's output. For s==0 (embedding entry point, no prior
            // shard) fall back to the first non-special input.
            if (s > 0) {
                if (in_name.empty() && t.name == prev_out) in_name = t.name;
            } else if (in_name.empty() && !isSpecialTensor(t.name)) {
                in_name = t.name;
            }
            if (spec_.hidden_size == 0 && !isSpecialTensor(t.name)) {
                if (size_t h = hiddenDimOf(t)) spec_.hidden_size = h;
            }
            // KV shape: [num_kv_heads, 1, head_dim, kv_len].
            if (t.name.rfind("past_key_", 0) == 0 && t.shape.size() >= 3) {
                if (spec_.num_kv_heads == 0) spec_.num_kv_heads = t.shape[0];
                if (spec_.head_dim == 0) spec_.head_dim = t.shape[2];
            }
        }
        // Fallback: if a s>0 shard has no input matching the previous output
        // (unexpected — differently-named boundary), revert to first non-special
        // so single-shard and legacy bundles still resolve.
        if (in_name.empty() && s > 0) {
            for (const auto& t : g.inputSpecs()) {
                if (!isSpecialTensor(t.name)) {
                    in_name = t.name;
                    break;
                }
            }
        }
        bool has_kv_out = false;
        for (const auto& t : g.outputSpecs()) {
            if (out_name.empty() && !isSpecialTensor(t.name)) out_name = t.name;
            // A hidden-state OUTPUT also carries hidden_size (e.g. shard 0,
            // whose only non-special input is an integer token-id tensor).
            if (spec_.hidden_size == 0 && !isSpecialTensor(t.name)) {
                if (size_t h = hiddenDimOf(t)) spec_.hidden_size = h;
            }
            if (t.name == "logits" && spec_.vocab_size == 0) spec_.vocab_size = t.shape.back();
            if (isKVTensor(t.name)) has_kv_out = true;
        }
        if (in_name.empty() || out_name.empty()) {
            throw std::runtime_error("inferSpecFromGraphs: shard " + std::to_string(s + 1) + " (graph '" + g.name() +
                                     "') has no non-special input/output tensor");
        }
        spec_.shards[s].in_state_name  = in_name;
        spec_.shards[s].out_state_name = out_name;
        spec_.shards[s].lm_head_only   = !has_kv_out && out_name == "logits" && s > 0;

        for (auto& block : spec_.state_blocks) {
            if (isKVStateBlock(block.kind)) block.shard_pairs[s] = discoverKVPairs(g, block);
        }
    }

    if (spec_.hidden_size == 0 || spec_.vocab_size == 0) {
        throw std::runtime_error("inferSpecFromGraphs: could not determine hidden_size / vocab_size from graphs");
    }

    GENIEX_LOG_DEBUG("inferSpecFromGraphs: shards={} hidden_size={} num_kv_heads={} head_dim={} vocab_size={}",
        shard_count_,
        spec_.hidden_size,
        spec_.num_kv_heads,
        spec_.head_dim,
        spec_.vocab_size);
}

bool LLMModel::onInitialized() {
    // Discover CL / AR / phase-prefix from the loaded QNN graph names. The
    // regex tolerates an optional alphabetic prefix (Genie's `prompt_` /
    // `token_`, absent on AI Hub IoT exports).
    static const std::regex graph_name_re(R"((?:[A-Za-z]+_)?ar(\d+)_cl(\d+)_(\d+)_of_(\d+))");

    struct ParsedGraph {
        bool   ok    = false;
        size_t ar    = 0;
        size_t cl    = 0;
        size_t shard = 0;
        size_t total = 0;
    };
    auto parseGraphName = [&](const std::string& name) -> ParsedGraph {
        std::smatch m;
        if (!std::regex_match(name, m, graph_name_re)) return {};
        try {
            return {
                true, std::stoul(m[1].str()), std::stoul(m[2].str()), std::stoul(m[3].str()), std::stoul(m[4].str())};
        } catch (...) {
            return {};
        }
    };

    // Every loaded graph must parse. An unmatched name would otherwise sort
    // into an arbitrary slot and trip Graph::write much later.
    std::set<size_t> cl_set;
    std::set<size_t> ar_set;
    std::set<size_t> shard_set;  // actual loaded shard numbers (may be non-contiguous)
    for (const auto& g : graphs_) {
        const auto p = parseGraphName(g.name());
        if (!p.ok) {
            GENIEX_LOG_ERROR("LLMModel: graph name '{}' does not match '(<phase>_)?arN_clM_S_of_T'", g.name());
            return false;
        }
        cl_set.insert(p.cl);
        ar_set.insert(p.ar);
        shard_set.insert(p.shard);
    }
    if (cl_set.empty() || ar_set.empty()) {
        GENIEX_LOG_ERROR("LLMModel: no graphs loaded");
        return false;
    }
    spec_.context_lengths.assign(cl_set.begin(), cl_set.end());
    spec_.seq_len_prefill = *ar_set.rbegin();
    spec_.seq_len_decode  = *ar_set.begin();
    num_cl_               = spec_.context_lengths.size();

    // shard_count_ is the number of *loaded* shards, not the "_of_T" total: a
    // bundle whose leading shard runs off-graph (e.g. a CPU-side embedding LUT,
    // so the first ctx-bin is `2_of_3`) exposes fewer graphs than T. Ranking the
    // actual shard numbers into a dense 0-based index keeps graphIndex() and the
    // per-shard buffers sized to what was really loaded.
    shard_count_ = shard_set.size();
    std::map<size_t, int> shard_rank;
    {
        int rank = 0;
        for (size_t s : shard_set) shard_rank[s] = rank++;
    }

    auto sortKey = [&](const std::string& name) -> std::tuple<int, int, int> {
        const auto p      = parseGraphName(name);
        const int  phase  = (p.ar == spec_.seq_len_prefill) ? 0 : 1;
        const int  shard  = shard_rank.at(p.shard);
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

    // Infer every tensor-derived field (shapes, shard wiring, KV pairs) from
    // the now-sorted graphs. This is the sole source of truth for hyperparameters.
    inferSpecFromGraphs();

    kv_state_block_idx_ = std::numeric_limits<size_t>::max();
    for (size_t block_idx = 0; block_idx < spec_.state_blocks.size(); ++block_idx) {
        if (spec_.state_blocks[block_idx].kind == StateBlockKind::KV) {
            kv_state_block_idx_ = block_idx;
            break;
        }
    }

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

    createInputProviders();
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
        // NOTE: every log arg is pre-stringified by logging.h's lp() before the
        // format string is applied, so only plain "{}" specifiers are valid here
        // ("{:#x}" on an already-formatted string throws "invalid format
        // specifier"). Format the hex mask ourselves and pass it as a string.
        GENIEX_LOG_DEBUG("decode pool: workers={} cpu_mask={} poll={} clock_keeper={}",
            n_workers,
            fmt::format("{:#x}", cpu_mask),
            poll,
            clock_keeper_threads_);
        decode_pool_ = std::make_unique<ThreadPool>();
        decode_pool_->start(n_workers, cpu_mask, poll);
    }

    return true;
}

void LLMModel::createInputProviders() {
    // Skip if the caller already registered providers (e.g. a subclass or a
    // manual-tensor example).
    if (!input_providers_.empty()) return;

    input_providers_.push_back(makeEmbeddingProvider(spec_.shards.front().in_state_name, gc_));

    // RoPE dimension (Option C): last dim of the cos tensor = head_dim/2.
    // The tensor may live on any shard (shard 0 is often an embedding-only LUT
    // with no position inputs), so scan all shards' prefill graphs. Its absence
    // everywhere means the graph bakes RoPE internally — no provider needed.
    //
    // Newer exports (e.g. Gemma4 W4A16 v81, QAIRT 2.45) rename the global-RoPE
    // pair from position_ids_cos/sin to position_ids_global_cos/sin, so accept
    // either and feed whichever the graph exposes.
    static constexpr const char* kGlobalRopeCos[] = {"position_ids_cos", "position_ids_global_cos"};
    static constexpr const char* kGlobalRopeSin[] = {"position_ids_sin", "position_ids_global_sin"};
    bool                         rope_found       = false;
    for (size_t s = 0; s < shard_count_ && !rope_found; ++s) {
        const Graph& g = graph(graphIndex(0, s, 0));
        for (size_t v = 0; v < 2; ++v) {
            if (g.hasInput(kGlobalRopeCos[v])) {
                const size_t half_dim = g.inputSpec(kGlobalRopeCos[v]).shape.back();
                GENIEX_LOG_INFO("llm: global RoPE provider bound to '{}' (head_dim={}) on shard {}",
                    kGlobalRopeCos[v],
                    half_dim * 2,
                    s);
                input_providers_.push_back(makeRoPEProvider(half_dim * 2, gc_, kGlobalRopeCos[v], kGlobalRopeSin[v]));
                rope_found = true;
                break;
            }
        }
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

    const size_t seq_len = (phase == 0) ? spec_.seq_len_prefill : spec_.seq_len_decode;

    if (getenv("GENIEX_DUMP_IO")) {
        static std::unordered_set<std::string> seen;
        if (seen.insert(g.name()).second) {
            for (const auto& ts : g.inputSpecs()) {
                std::string sh;
                for (auto d : ts.shape) sh += std::to_string(d) + ",";
                GENIEX_LOG_INFO("IO[{}] IN {} dtype={} scale={} offset={} shape=[{}]",
                    g.name(),
                    ts.name,
                    static_cast<int>(ts.dtype),
                    ts.quant_scale,
                    ts.quant_offset,
                    sh);
            }
            for (const auto& ts : g.outputSpecs()) {
                GENIEX_LOG_INFO("IO[{}] OUT {} dtype={} scale={} offset={}",
                    g.name(),
                    ts.name,
                    static_cast<int>(ts.dtype),
                    ts.quant_scale,
                    ts.quant_offset);
            }
        }
    }

    if (g.hasInput(spec_.attention_mask_name)) {
        auto mask = get_attention_mask(ctx.n_past, ctx.curr_len, seq_len, kv_len);
        g.write(spec_.attention_mask_name, mask.data(), mask.size());
    }

    // Gemma3/4 sliding-window mask: a second, band-limited causal mask for the
    // local-attention layers. Its kv_len is the fixed swa window (read from the
    // tensor's own last dim minus seq_len), not the global growing cache length.
    if (!spec_.swa_attention_mask_name.empty() && g.hasInput(spec_.swa_attention_mask_name)) {
        const size_t total_len  = g.inputSpec(spec_.swa_attention_mask_name).shape.back();
        const size_t swa_kv_len = total_len - seq_len;
        auto swa_mask = get_sliding_window_mask(ctx.n_past, ctx.curr_len, seq_len, swa_kv_len, spec_.swa_window);
        g.write(spec_.swa_attention_mask_name, swa_mask.data(), swa_mask.size());
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

// kv_len (token capacity) of a KV input tensor: last dim for keys
// [H,1,head_dim,kv_len], dim 2 for values [H,1,kv_len,head_dim].
size_t LLMModel::kvCapacityOf(Graph& g, const std::string& name, bool is_key) const {
    const TensorSpec& spec = g.inputSpec(name);
    return is_key ? spec.shape[3] : spec.shape[2];
}

// Shift a fixed-window KV input buffer left by `shift` tokens (dropping the
// oldest), making room to append fresh tokens at the tail. Used only by
// sliding-window caches once their window fills.
void LLMModel::shiftKVLeft(Graph& g, const std::string& name, size_t shift, bool is_key) {
    if (shift == 0) return;
    const TensorSpec& spec      = g.inputSpec(name);
    const size_t      elem_size = spec.elementSize();
    auto*             buf       = static_cast<uint8_t*>(g.inputPtr(name));
    size_t            num_rows, kv_len, token_size;
    if (is_key) {
        num_rows   = spec.shape[0] * spec.shape[2];  // H * head_dim
        kv_len     = spec.shape[3];
        token_size = elem_size;
    } else {
        num_rows   = spec.shape[0];  // H
        kv_len     = spec.shape[2];
        token_size = spec.shape[3] * elem_size;  // head_dim * elem
    }
    if (shift >= kv_len) return;
    const size_t keep = kv_len - shift;
    for (size_t row = 0; row < num_rows; ++row) {
        uint8_t* base = buf + row * kv_len * token_size;
        std::memmove(base, base + shift * token_size, keep * token_size);
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

    // Head layout is derived from the tensors themselves, not the global
    // spec_.{num_kv_heads,head_dim}: a model may own multiple KV blocks with
    // different head dims (e.g. Gemma3/4's global past_* vs sliding-window
    // swa_* caches), and copyKV must honour whichever block this pair belongs
    // to. Shapes: key [H, 1, head_dim, kv_len], value [H, 1, kv_len, head_dim].
    size_t num_rows, src_kv_len, dst_kv_len, token_size;
    if (is_key) {
        const size_t H  = src_spec.shape[0];
        const size_t hd = src_spec.shape[2];
        num_rows        = H * hd;
        src_kv_len      = src_spec.shape[3];
        dst_kv_len      = dst_spec.shape[3];
        token_size      = elem_size;
    } else {
        const size_t H  = src_spec.shape[0];
        const size_t hd = src_spec.shape[3];
        num_rows        = H;
        src_kv_len      = src_spec.shape[2];
        dst_kv_len      = dst_spec.shape[2];
        token_size      = hd * elem_size;
    }

    for (size_t row = 0; row < num_rows; ++row)
        std::memcpy(dst_buf + (row * dst_kv_len + dst_off) * token_size,
            src_buf + (row * src_kv_len + src_off) * token_size,
            n_tok * token_size);
}

// Propagates freshly-computed KV outputs back into the KV input buffers so each execution sees the full context
// history.
void LLMModel::updateKV(size_t s, size_t phase, size_t dst_off, size_t n_tok) {
    Graph& g = graph(graphIndex(phase, s, active_cl_idx_));
    // Advance EVERY KV block this shard owns, not just the primary one. A
    // sliding-window model (Gemma3/4) carries a second `swa_*` cache whose
    // layers alternate with the global `past_*` layers; both must be written
    // back after each step or the local-attention layers see stale KV.
    for (const auto& block : spec_.state_blocks) {
        if (!isKVStateBlock(block.kind)) continue;
        if (s >= block.shard_pairs.size()) continue;
        const auto& pairs = block.shard_pairs[s];
        if (pairs.empty()) continue;

        // Fixed-window caches (swa_*) never grow their kv_len across phases;
        // once the window is full the write wraps by shifting the buffer left
        // by one before appending. For dst_off < window this is a plain append.
        const size_t kv_capacity = kvCapacityOf(g, pairs.front().key_in, /*is_key=*/true);
        size_t       off         = dst_off;
        if (dst_off + n_tok > kv_capacity) {
            // Window overflow: drop the oldest (dst_off + n_tok - kv_capacity)
            // tokens so the newest n_tok land at the tail.
            const size_t shift = dst_off + n_tok - kv_capacity;
            for (const auto& p : pairs) {
                shiftKVLeft(g, p.key_in, shift, /*is_key=*/true);
                shiftKVLeft(g, p.value_in, shift, /*is_key=*/false);
            }
            off = kv_capacity - n_tok;
        }
        for (const auto& p : pairs) {
            copyKV(g, p.key_out, true, g, p.key_in, 0, off, n_tok, true);
            copyKV(g, p.value_out, true, g, p.value_in, 0, off, n_tok, false);
        }
    }
}

// Adjusts KV cache stride in-place when switching context-length variants.
// All CL/phase variants share the same physical buffer, so no reallocation is needed.
// Expanding iterates rows backward to avoid overwriting unread data; contracting goes forward.
void LLMModel::reshapeKV(size_t shard, size_t old_kv_len, size_t new_kv_len, size_t n_valid) {
    if (old_kv_len == new_kv_len) return;

    // Cap copy length: never read past old_kv_len or write past new_kv_len.
    const size_t copy_len = std::min(n_valid, std::min(old_kv_len, new_kv_len));

    Graph& g = graph(graphIndex(0, shard, active_cl_idx_));

    // Restride only the global CL-scaled KV cache; fixed-window (swa) caches
    // keep a constant kv_len across context-length variants and must not grow.
    for (const auto& block : spec_.state_blocks) {
        if (block.kind != StateBlockKind::KV) continue;
        if (shard >= block.shard_pairs.size()) continue;
        const auto& pairs = block.shard_pairs[shard];
        if (pairs.empty()) continue;

        for (const auto& p : pairs) {
            const auto& key_in = p.key_in;

            // Key: [num_kv_heads, 1, head_dim, kv_len]
            {
                const TensorSpec& spec      = g.inputSpec(key_in);
                const size_t      elem_size = spec.elementSize();
                const size_t      n_rows    = spec.shape[0] * spec.shape[2];  // H * head_dim (this block)
                auto*             buf       = static_cast<uint8_t*>(g.inputPtr(key_in));

                if (new_kv_len > old_kv_len) {
                    for (ptrdiff_t row = static_cast<ptrdiff_t>(n_rows) - 1; row >= 0; --row) {
                        std::memmove(buf + row * new_kv_len * elem_size,
                            buf + row * old_kv_len * elem_size,
                            copy_len * elem_size);
                        if (copy_len < new_kv_len)
                            fillEncodedZero(buf + (row * new_kv_len + copy_len) * elem_size,
                                (new_kv_len - copy_len) * elem_size,
                                spec.dtype);
                    }
                } else {
                    for (size_t row = 0; row < n_rows; ++row)
                        std::memmove(buf + row * new_kv_len * elem_size,
                            buf + row * old_kv_len * elem_size,
                            copy_len * elem_size);
                }
            }

            // Value: [num_kv_heads, 1, kv_len, head_dim]
            {
                const auto&       val_in     = p.value_in;
                const TensorSpec& spec       = g.inputSpec(val_in);
                const size_t      elem_size  = spec.elementSize();
                const size_t      n_heads    = spec.shape[0];              // H (this block)
                const size_t      token_size = spec.shape[3] * elem_size;  // head_dim * elem (this block)
                auto*             buf        = static_cast<uint8_t*>(g.inputPtr(val_in));

                if (new_kv_len > old_kv_len) {
                    for (ptrdiff_t h = static_cast<ptrdiff_t>(n_heads) - 1; h >= 0; --h) {
                        std::memmove(buf + h * new_kv_len * token_size,
                            buf + h * old_kv_len * token_size,
                            copy_len * token_size);
                        if (copy_len < new_kv_len)
                            fillEncodedZero(buf + (h * new_kv_len + copy_len) * token_size,
                                (new_kv_len - copy_len) * token_size,
                                spec.dtype);
                    }
                } else {
                    for (size_t h = 0; h < n_heads; ++h)
                        std::memmove(buf + h * new_kv_len * token_size,
                            buf + h * old_kv_len * token_size,
                            copy_len * token_size);
                }
            }
        }
    }
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

// Mirrors llama.cpp's context-shift heuristic: discards ~half of (n_past - n_keep), or more if
// needed to fit `n_fit`. Returns 0 when n_past <= n_keep.
/*static*/ size_t LLMModel::computeSlideDiscard(size_t n_past, size_t n_fit, size_t max_cl, size_t n_keep) {
    if (n_past <= n_keep) return 0;

    const auto past   = static_cast<std::ptrdiff_t>(n_past);
    const auto fit    = static_cast<std::ptrdiff_t>(n_fit);
    const auto cl     = static_cast<std::ptrdiff_t>(max_cl);
    const auto keep   = static_cast<std::ptrdiff_t>(n_keep);
    const auto needed = past + fit - cl + 1;

    auto discard = std::max(past / 2 - keep, needed);
    discard      = std::min(discard, past - keep);
    return discard > 0 ? static_cast<size_t>(discard) : 0;
}

// Evicts the oldest `n_discard` tokens above the anchored `n_keep` prefix, then re-prefills the
// surviving tail (token IDs recovered from token_history_) instead of relocating its cached KV --
// QAIRT's compiled graphs cache post-RoPE K/V with no facility to re-rotate cached history, so a
// byte relocation would leave survivors' RoPE rotation at an out-of-distribution position.
//
// `at_decode_stride` must be true when called mid-decode-loop; the buffer is restrided to prefill
// stride, re-prefilled, then restrided back so the caller's decode loop continues unmodified.
void LLMModel::slideWindowEvict(size_t n_discard, size_t n_keep, bool at_decode_stride) {
    if (n_discard == 0) return;

    const size_t tail_begin = n_keep + n_discard;
    const size_t tail_len   = (n_past_ > tail_begin) ? (n_past_ - tail_begin) : 0;

    GENIEX_LOG_INFO(
        "sliding window: discarding {} tokens (n_keep={}), re-prefilling {} surviving tokens, n_past {} -> {}",
        n_discard,
        n_keep,
        tail_len,
        n_past_,
        n_keep + tail_len);

    // Recover the surviving tail's token IDs before n_past_ / token_history_ are touched below.
    std::vector<int32_t> tail_tokens(token_history_.begin() + static_cast<std::ptrdiff_t>(tail_begin),
        token_history_.begin() + static_cast<std::ptrdiff_t>(tail_begin + tail_len));

    if (at_decode_stride) {
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, decode_kv, prefill_kv, n_keep);
    }

    n_past_ = n_keep;
    token_history_.resize(n_keep);

    prefillChunks(tail_tokens, /*last_chunk_size_out=*/nullptr);

    if (at_decode_stride) {
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, prefill_kv, decode_kv, n_past_);
    }
}

// Chunked prefill over `tokens`, writing fresh KV starting at the current n_past_ and advancing
// n_past_ / token_history_ as each chunk completes. Assumes the KV buffer is already strided for
// prefill; callers at decode stride must reshapeKV first (see slideWindowEvict). Extracted from
// generate()'s main prefill loop so slideWindowEvict's re-prefill can reuse the same logic.
void LLMModel::prefillChunks(
    const std::vector<int32_t>& tokens, size_t* last_chunk_size_out, std::vector<float>* all_logits_out) {
    size_t       tokens_processed = 0;
    const size_t total_tokens     = tokens.size();

    // Collecting per-position logits requires the LM head to run on every chunk,
    // not just the final one (see the lm_head_only skip below).
    const bool collect_logits = all_logits_out != nullptr;

    while (tokens_processed < total_tokens) {
        const size_t remaining      = total_tokens - tokens_processed;
        const size_t chunk_size     = std::min(remaining, spec_.seq_len_prefill);
        const bool   is_final_chunk = (tokens_processed + chunk_size >= total_tokens);
        if (last_chunk_size_out) *last_chunk_size_out = chunk_size;

        // Ensure the prefill KV buffer (CL - seq_len_prefill) can hold n_past + chunk_size after this chunk.
        promoteCL(/*required=*/n_past_ + chunk_size,
            /*capacity_reserved_seq=*/spec_.seq_len_prefill,
            /*stride_reserved_seq=*/spec_.seq_len_prefill);

        const std::vector<int32_t> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(tokens_processed),
            tokens.begin() + static_cast<std::ptrdiff_t>(tokens_processed + chunk_size));

        GENIEX_LOG_DEBUG("prefill chunk: tokens [{}, {}) cl_idx={} final={}",
            tokens_processed,
            tokens_processed + chunk_size,
            active_cl_idx_,
            is_final_chunk);
        const LLMRunContext ctx{chunk, n_past_, chunk_size, /*phase=*/0};

        // Force the LM head to run when the caller wants logits for this chunk.
        const bool run_lm_head = is_final_chunk || collect_logits;

        for (size_t s = 0; s < shard_count_; ++s) {
            // For non-final prefill chunks we only need the KV cache to be populated, so the
            // LM-head shard can be skipped entirely -- unless the caller asked for per-chunk logits.
            if (!run_lm_head && spec_.shards[s].lm_head_only) {
                GENIEX_LOG_DEBUG("skipping LM-head-only shard {} on non-final prefill chunk", s);
                continue;
            }
            runShard(s, /*phase=*/0, active_cl_idx_, ctx);
            updateKV(s, /*phase=*/0, n_past_, chunk_size);
            if (s + 1 < shard_count_) {
                if (!run_lm_head && spec_.shards[s + 1].lm_head_only) {
                    continue;
                }
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }
        }

        // Append this chunk's logits row-major ([chunk_size, vocab_size]) once the LM head has run.
        if (collect_logits) {
            const size_t vocab = spec_.vocab_size;
            const size_t base  = all_logits_out->size();
            all_logits_out->resize(base + chunk_size * vocab);
            const Graph& lm_head = graph(graphIndex(/*phase=*/0, /*shard=*/shard_count_ - 1, active_cl_idx_));
            lm_head.read(spec_.shards.back().out_state_name,
                all_logits_out->data() + base,
                chunk_size * vocab,
                /*elem_offset=*/0);
        }

        token_history_.insert(token_history_.end(), chunk.begin(), chunk.end());
        n_past_ += chunk_size;
        tokens_processed += chunk_size;
    }
}

bool LLMModel::isEndOfGeneration(int32_t token, const GenerationConfig& gen_cfg) const {
    for (int32_t eos_id : spec_.eos_token_ids)
        if (token == eos_id) return true;
    // The bundle config often lists only <eos>; chat models end turns on a
    // separate token (e.g. Gemma's <turn|>) that the tokenizer's EOG set covers.
    return gen_cfg.tokenizer && gen_cfg.tokenizer->is_eog(token);
}

std::vector<int32_t> LLMModel::generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
    std::function<bool(int32_t)> token_callback) {
    const size_t total_tokens    = prompt_tokens.size();
    size_t       last_chunk_size = 0;  // valid token count in the final prefill chunk

    GENIEX_LOG_DEBUG("generate: prompt_tokens={}, n_past={}, max_tokens={}", total_tokens, n_past_, gen_cfg.max_tokens);

    // (Re)build & seed the sampler. No-op when sampling is disabled — in
    // that case `sampler_` stays null and sampleNextToken() takes the greedy
    // argmax fast path.
    prepareSampler(gen_cfg, prompt_tokens);

    // Reject prompts that cannot fit in the largest available context length, unless sliding_window
    // is opted in, in which case evict the oldest tokens above n_keep to make room first.
    // context_lengths is sorted ascending, so the last entry is the max CL.
    const size_t max_cl = spec_.context_lengths.back();
    const size_t n_keep =
        gen_cfg.sliding_window
            ? std::min<size_t>(static_cast<size_t>(std::max<int32_t>(gen_cfg.sliding_window_n_keep, 0)), max_cl)
            : 0;

    // Attempts to evict enough of the oldest tokens (above n_keep) to fit `n_fit` more.
    // `at_decode_stride` must reflect whether the KV buffer is currently strided for decode (true)
    // or prefill (false) at the call site -- see slideWindowEvict. Returns false (no-op) when
    // sliding_window is disabled or eviction alone cannot make the requested room.
    auto try_slide = [&](size_t n_fit, bool at_decode_stride) -> bool {
        if (!gen_cfg.sliding_window) return false;
        const size_t n_discard = computeSlideDiscard(n_past_, n_fit, max_cl, n_keep);
        if (n_discard == 0 || (n_past_ - n_discard) + n_fit > max_cl) return false;
        slideWindowEvict(n_discard, n_keep, at_decode_stride);
        return true;
    };

    if (n_past_ + total_tokens > max_cl) {
        // Entering generate(), the KV buffer is left at the prefill stride by the previous call's
        // cleanup reshape (or at initKVBuffers()'s default on the very first call).
        try_slide(total_tokens, /*at_decode_stride=*/false);
        if (n_past_ + total_tokens > max_cl) {
            throw ContextLengthExceededError(
                "geniex: prompt exceeds max context length (" + std::to_string(max_cl) + ")");
        }
    }

    prefillChunks(prompt_tokens, &last_chunk_size);

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
    startClockKeeper();

    for (int step = 0; step < gen_cfg.max_tokens; ++step) {
        if (isEndOfGeneration(next_token, gen_cfg)) break;
        output_tokens.push_back(next_token);
        if (token_callback && !token_callback(next_token)) {
            GENIEX_LOG_DEBUG("token_callback requested stop at step {}", step);
            break;
        }

        // KV write-back from the previous step must finish before restriding, evicting, or
        // re-reading the KV buffers below.
        if (decode_pool_) {
            decode_pool_->wait();
        }

        // Stop and report when the next decode step would exceed the largest available CL.
        // With sliding_window enabled, evict the oldest tokens above n_keep to make room first.
        // The KV buffer is at the decode stride throughout this loop (switched right before it).
        if (n_past_ + 1 > max_cl) {
            try_slide(1, /*at_decode_stride=*/true);
            if (n_past_ + 1 > max_cl) {
                throw ContextLengthExceededError(
                    "geniex: generation exceeds max context length (" + std::to_string(max_cl) + ")");
            }
        }

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
        token_history_.push_back(next_token);
        next_token = sampleNextToken(/*phase=*/1, /*token_offset=*/0);
    }

    // Drain KV jobs still in flight after an early break (EOS / callback stop).
    if (decode_pool_) decode_pool_->wait();

    // Decode window is over; release the cluster back to the governor.
    stopClockKeeper();

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

std::vector<float> LLMModel::forwardLogits(const std::vector<int32_t>& tokens, bool all_positions) {
    if (tokens.empty()) {
        throw std::invalid_argument("geniex: forwardLogits requires a non-empty token sequence");
    }

    const size_t max_cl = spec_.context_lengths.back();
    if (tokens.size() > max_cl) {
        throw ContextLengthExceededError("geniex: forwardLogits input (" + std::to_string(tokens.size()) +
                                         ") exceeds max context length (" + std::to_string(max_cl) + ")");
    }

    // Score only `tokens`: start from a clean KV cache so the result is independent
    // of any prior generate()/forwardLogits() call, and leave it clean on exit.
    resetKVCache();

    const size_t       vocab = spec_.vocab_size;
    std::vector<float> logits;
    size_t             last_chunk_size = 0;

    if (all_positions) {
        // Collect every position's logits across all prefill chunks.
        logits.reserve(tokens.size() * vocab);
        prefillChunks(tokens, &last_chunk_size, &logits);
    } else {
        // Only the final token's row is needed; let prefill skip the LM head on
        // earlier chunks (the cheap path) and read the last row afterward.
        prefillChunks(tokens, &last_chunk_size, /*all_logits_out=*/nullptr);
        readLastLogits(/*phase=*/0, /*token_offset=*/last_chunk_size - 1, logits);
    }

    // Restore prefill stride and clear state for the next caller.
    resetKVCache();
    return logits;
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
    // Flag every KV block's inputs, not just the primary cache: a sliding-window
    // model's `swa_*` inputs are KV state too and must not be treated as regular
    // graph inputs.
    for (const auto& block : spec_.state_blocks) {
        if (!isKVStateBlock(block.kind)) continue;
        for (size_t s = 0; s < shard_count_; ++s) {
            if (s >= block.shard_pairs.size()) continue;
            for (const auto& p : block.shard_pairs[s]) {
                names.insert(p.key_in);
                names.insert(p.value_in);
            }
        }
    }
    return names;
}

void LLMModel::resetKVCache() {
    n_past_        = 0;
    active_cl_idx_ = 0;
    token_history_.clear();

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

    // Persisted so slideWindowEvict() has valid history to re-prefill after a load; without this,
    // token_history_ would be empty while n_past_ is restored non-zero, corrupting the next eviction.
    const size_t history_size = token_history_.size();
    f.write(reinterpret_cast<const char*>(&history_size), sizeof(history_size));
    if (history_size > 0) {
        f.write(reinterpret_cast<const char*>(token_history_.data()),
            static_cast<std::streamsize>(history_size * sizeof(int32_t)));
    }

    const auto   kv_names       = buildKVInputNameSet();
    const size_t prefill_graphs = shard_count_ * num_cl_;
    for (size_t gi = 0; gi < prefill_graphs; ++gi) {
        const Graph& g = graph(gi);
        for (const auto& spec : g.inputSpecs()) {
            if (kv_names.count(spec.name)) {
                // KV state lives in the graph's INPUT buffer; read it raw.
                // (Graph::read targets output tensors, not inputs.)
                const void* buf = g.inputPtr(spec.name);
                if (!buf) continue;
                f.write(reinterpret_cast<const char*>(buf), static_cast<std::streamsize>(spec.byteCount()));
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

    size_t history_size = 0;
    f.read(reinterpret_cast<char*>(&history_size), sizeof(history_size));
    token_history_.assign(history_size, 0);
    if (history_size > 0) {
        f.read(reinterpret_cast<char*>(token_history_.data()),
            static_cast<std::streamsize>(history_size * sizeof(int32_t)));
    }

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

size_t LLMModel::vocabSize() const { return spec_.vocab_size; }

EmbeddingInputProvider* LLMModel::findEmbeddingProvider(const std::string& tensor_name) {
    for (auto& p : input_providers_) {
        auto* e = dynamic_cast<EmbeddingInputProvider*>(p.get());
        if (e && e->tensorName() == tensor_name) return e;
    }
    return nullptr;
}

void LLMModel::addInputProvider(std::unique_ptr<InputProvider> provider) {
    input_providers_.push_back(std::move(provider));
}

// Prefill and batched-decode primitives used by speculative decoders.

namespace {

// Additive causal attention mask [seq_len, kv_len + seq_len] for one prefill
// chunk: token i attends to the whole committed cache [0, n_past) and to chunk
// tokens 0..i. (Tree-shaped decode masks live in SpeculativeLLMModel.)
std::vector<float> buildPrefillCausalMask(size_t n_past, size_t num_tokens, size_t seq_len, size_t kv_len) {
    // The mask holds seq_len rows but the loop writes num_tokens of them, so an
    // oversized chunk would overrun it.
    if (num_tokens > seq_len)
        throw std::runtime_error("buildPrefillCausalMask: chunk size " + std::to_string(num_tokens) +
                                 " exceeds prefill width " + std::to_string(seq_len));
    const size_t       row_len = kv_len + seq_len;
    std::vector<float> mask(seq_len * row_len, -1e9f);

    for (size_t i = 0; i < num_tokens; ++i) {
        float* row = mask.data() + i * row_len;
        for (size_t j = 0; j < n_past; ++j) row[j] = 0.0f;       // committed history
        for (size_t j = 0; j <= i; ++j) row[kv_len + j] = 0.0f;  // causal (includes self)
    }
    return mask;
}

}  // namespace

void LLMModel::drainDecodePool() {
    if (decode_pool_) decode_pool_->wait();
}

void LLMModel::rewindKVCache(size_t n_past) {
    if (n_past > n_past_)
        throw std::invalid_argument(
            "rewindKVCache: cannot grow n_past (" + std::to_string(n_past) + " > " + std::to_string(n_past_) + ")");
    n_past_ = n_past;
}

const void* LLMModel::outputBytes(size_t graph_idx, const std::string& name) const {
    return graph(graph_idx).outputPtr(name);
}

const TensorSpec& LLMModel::outputTensorSpec(size_t graph_idx, const std::string& name) const {
    return graph(graph_idx).outputSpec(name);
}

void LLMModel::prefill(const std::vector<int32_t>& tokens, float rope_theta, const uint8_t* feature_rows,
    size_t feature_row_bytes, const std::string& feature_name, std::vector<uint8_t>* captured_features,
    const std::string& capture_name) {
    size_t          processed = 0;
    const size_t    total     = tokens.size();
    RotaryEmbedding rope(spec_.head_dim, rope_theta);

    if (captured_features) captured_features->clear();

    while (processed < total) {
        const size_t chunk_size = std::min(total - processed, spec_.seq_len_prefill);
        promoteCL(n_past_ + chunk_size, spec_.seq_len_prefill, spec_.seq_len_prefill);

        const std::vector<int32_t> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(processed),
            tokens.begin() + static_cast<std::ptrdiff_t>(processed + chunk_size));

        std::vector<int32_t> pos(chunk_size);
        for (size_t i = 0; i < chunk_size; ++i) pos[i] = static_cast<int32_t>(n_past_ + i);
        auto [cos_vec, sin_vec] = rope.forward(pos);

        const size_t        kv_len  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        const size_t        seq_len = spec_.seq_len_prefill;
        auto                mask    = buildPrefillCausalMask(n_past_, chunk_size, seq_len, kv_len);
        const LLMRunContext ctx{chunk, n_past_, chunk_size, /*phase=*/0};
        for (size_t s = 0; s < shard_count_; ++s) {
            Graph& g = graph(graphIndex(/*phase=*/0, s, active_cl_idx_));

            if (g.hasInput(spec_.attention_mask_name)) g.write(spec_.attention_mask_name, mask.data(), mask.size());
            if (feature_rows && !feature_name.empty() && g.hasInput(feature_name)) {
                g.write(feature_name, feature_rows + processed * feature_row_bytes, chunk_size * feature_row_bytes);
            }
            for (auto& provider : input_providers_) provider->write(g, ctx);
            if (g.hasInput("position_ids_cos")) g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
            if (g.hasInput("position_ids_sin")) g.write("position_ids_sin", sin_vec.data(), sin_vec.size());

            TimeLog tl;
            if (!g.execute(tl)) throw std::runtime_error("prefill: execute failed on shard " + std::to_string(s));

            updateKV(s, /*phase=*/0, n_past_, chunk_size);
            if (s + 1 < shard_count_) applyConnections({shard_hidden_state_[active_cl_idx_][s]});
        }

        if (captured_features && !capture_name.empty()) {
            const size_t body = graphIndex(/*phase=*/0, shard_count_ >= 2 ? shard_count_ - 2 : 0, active_cl_idx_);
            const auto&  spec = graph(body).outputSpec(capture_name);
            const size_t row  = spec.shape.back() * spec.elementSize();
            const auto*  src  = static_cast<const uint8_t*>(graph(body).outputPtr(capture_name));
            captured_features->insert(captured_features->end(), src, src + chunk_size * row);
        }

        token_history_.insert(token_history_.end(), chunk.begin(), chunk.end());
        n_past_ += chunk_size;
        processed += chunk_size;
    }
}

void LLMModel::startClockKeeper() {
    if (decode_pool_ && clock_keeper_threads_ > 0)
        decode_pool_->startClockKeeper(clock_keeper_threads_, decode_cpu_mask_);
}

void LLMModel::stopClockKeeper() {
    if (decode_pool_) decode_pool_->stopClockKeeper();
}

}  // namespace geniex

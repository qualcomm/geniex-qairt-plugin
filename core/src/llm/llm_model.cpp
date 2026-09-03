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
#include <tuple>
#include <unordered_set>

#include "llm/input_provider.h"
#include "llm/kv_layout.h"  // KV buffer layouts (flat / HMX-tiled)
#include "llm/llm_utils.h"  // isKVTensor / isSpecialTensor
#include "logging.h"
#include "utils.h"

namespace geniex {

namespace {
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

// Rewrites the primary KV block's tensor-name patterns to match what `g`
// actually exposes, when the declared patterns matched nothing. Returns true if
// the patterns changed (so the caller should re-run discovery).
//
// Exports disagree on KV naming. The default `past_key_<layer>_in` covers most,
// `past_key_<layer>_h<group>_in` is handled by discoverKVPairs' middle capture,
// but the Llama-3.2-3B SSD w4a16 bundle uses a cache-group prefix and a spelt-out
// per-head infix: `past_nativekvcache__key_<layer>_head_<h>_in`. Deriving the
// prefix from a real tensor name covers all of them without a per-model spec, and
// without it the model loads, prefills fine, and then emits gibberish because
// nothing is ever written back.
bool LLMModel::adoptKVNamingFromGraph(const Graph& g, size_t shard) {
    StateBlockSpec* primary = nullptr;
    for (auto& b : spec_.state_blocks) {
        if (b.kind == StateBlockKind::KV) {
            primary = &b;
            break;
        }
    }
    if (primary == nullptr) return false;
    if (shard < primary->shard_pairs.size() && !primary->shard_pairs[shard].empty()) return false;

    // Find a key output to learn the naming from. Anything ending "_out" whose
    // name carries "key" qualifies; the value sibling is the same name with the
    // last "key" swapped for "value".
    for (const auto& t : g.outputSpecs()) {
        const std::string& n = t.name;
        if (n.size() < 4 || n.compare(n.size() - 4, 4, "_out") != 0) continue;
        const auto key_at = n.rfind("key");
        if (key_at == std::string::npos) continue;

        // prefix ends just after "key" plus the separator the export uses.
        size_t pre_end = key_at + 3;
        if (pre_end < n.size() && n[pre_end] == '_') ++pre_end;
        const std::string prefix = n.substr(0, pre_end);
        const std::string middle = n.substr(pre_end, n.size() - pre_end - 4);
        if (middle.empty() || middle[0] < '0' || middle[0] > '9') continue;

        std::string val_prefix = prefix;
        val_prefix.replace(key_at, 3, "value");

        StateBlockSpec candidate    = *primary;
        candidate.key_in_pattern    = prefix + "{}_in";
        candidate.key_out_pattern   = prefix + "{}_out";
        candidate.value_in_pattern  = val_prefix + "{}_in";
        candidate.value_out_pattern = val_prefix + "{}_out";

        // Only adopt if all four resolve for this very tensor.
        if (!g.hasInput(candidate.key_in_pattern.substr(0, prefix.size()) + middle + "_in")) continue;
        if (!g.hasInput(val_prefix + middle + "_in")) continue;
        if (!g.hasOutput(val_prefix + middle + "_out")) continue;

        if (candidate.key_in_pattern == primary->key_in_pattern &&
            candidate.value_in_pattern == primary->value_in_pattern) {
            return false;  // already correct; discovery failed for another reason
        }

        GENIEX_LOG_INFO("llm: KV naming derived from graph '{}': '{}' / '{}' (was '{}')",
            g.name(),
            candidate.key_in_pattern,
            candidate.value_in_pattern,
            primary->key_in_pattern);
        primary->key_in_pattern    = candidate.key_in_pattern;
        primary->key_out_pattern   = candidate.key_out_pattern;
        primary->value_in_pattern  = candidate.value_in_pattern;
        primary->value_out_pattern = candidate.value_out_pattern;
        return true;
    }
    return false;
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

        // A shard that clearly owns KV tensors but matched no pattern would cache
        // nothing at all -- prefill would look fine and decode would emit
        // gibberish. Rather than fail that way, derive the naming from the graph
        // itself and retry. Exports do vary: the SSD w4a16 bundle names its cache
        // `past_nativekvcache__key_<layer>_head_<h>_in`.
        if (adoptKVNamingFromGraph(g, s)) {
            for (auto& block : spec_.state_blocks) {
                if (isKVStateBlock(block.kind)) block.shard_pairs[s] = discoverKVPairs(g, block);
            }
        }

        // KV head count and head_dim come from a resolved key input, not from a
        // hardcoded name prefix: an export that names its cache anything other than
        // `past_key_*` would otherwise leave both at 0, and a zero head_dim silently
        // produces an empty RoPE table (garbage positions, plausible-looking but
        // wrong output). Shape is [num_kv_heads, 1, head_dim, kv_len]; a
        // one-tensor-per-head export reports num_kv_heads = 1, matching LLMSpec.
        for (const auto& block : spec_.state_blocks) {
            if (!isKVStateBlock(block.kind) || s >= block.shard_pairs.size()) continue;
            if (block.shard_pairs[s].empty()) continue;
            const TensorSpec& key_in = g.inputSpec(block.shard_pairs[s].front().key_in);
            if (key_in.shape.size() < 4) continue;
            if (spec_.num_kv_heads == 0) spec_.num_kv_heads = key_in.shape[0];
            if (spec_.head_dim == 0) spec_.head_dim = key_in.shape[2];
            break;
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
    // regex tolerates an optional alphabetic prefix (`prompt_` / `token_` on
    // some exports, absent on AI Hub IoT exports).
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
    // LLMSpec has exactly two AR slots (prefill and decode) and graphIndex() lays the
    // graphs out in two phases, so a third AR has nowhere to go unless a driver opts
    // in via min_decode_seq_len (SSD's tree pass): reject it outright otherwise, per
    // the same reasoning as below.
    if (ar_set.size() > 2 && spec_.min_decode_seq_len == 0) {
        std::string ars;
        for (size_t ar : ar_set) {
            if (!ars.empty()) ars += ", ";
            ars += std::to_string(ar);
        }
        GENIEX_LOG_ERROR(
            "LLMModel: graphs expose {} distinct AR lengths ({}), but only prefill and decode are modelled",
            ar_set.size(),
            ars);
        return false;
    }
    spec_.context_lengths.assign(cl_set.begin(), cl_set.end());

    // Pick the two AR widths this runtime addresses. Prefill is always the widest.
    // Decode is the smallest that satisfies spec_.min_decode_seq_len, so a driver
    // needing a specific width (SSD's tree pass) gets it even when the bundle also
    // ships a narrower variant.
    spec_.seq_len_prefill = *ar_set.rbegin();
    spec_.seq_len_decode  = *ar_set.begin();
    if (spec_.min_decode_seq_len > 0) {
        const auto fit = ar_set.lower_bound(spec_.min_decode_seq_len);
        if (fit == ar_set.end()) {
            GENIEX_LOG_ERROR("LLMModel: decode needs at least {} token slots but the widest AR variant is {}",
                spec_.min_decode_seq_len,
                *ar_set.rbegin());
            return false;
        }
        spec_.seq_len_decode = *fit;
    }

    // Drop variants we cannot address: graphIndex() maps exactly two phases, so a
    // third AR width would collide into a phase slot and silently mis-wire. The
    // context binaries stay loaded (weights are shared), only the handles go.
    if (ar_set.size() > 2) {
        const size_t             keep_prefill = spec_.seq_len_prefill;
        const size_t             keep_decode  = spec_.seq_len_decode;
        std::vector<std::string> dropped;
        auto                     unused = [&](const Graph& g) {
            const auto p = parseGraphName(g.name());
            if (!p.ok) return false;
            if (p.ar == keep_prefill || p.ar == keep_decode) return false;
            dropped.push_back(g.name());
            return true;
        };
        graphs_.erase(std::remove_if(graphs_.begin(), graphs_.end(), unused), graphs_.end());
        GENIEX_LOG_INFO("LLMModel: using AR-{} (prefill) and AR-{} (decode); ignoring {} graph(s) of other AR widths",
            keep_prefill,
            keep_decode,
            dropped.size());

        // Re-derive the sets from what survived.
        cl_set.clear();
        shard_set.clear();
        for (const auto& g : graphs_) {
            const auto p = parseGraphName(g.name());
            cl_set.insert(p.cl);
            shard_set.insert(p.shard);
        }
    }

    spec_.context_lengths.assign(cl_set.begin(), cl_set.end());
    num_cl_ = spec_.context_lengths.size();

    // shard_count_ is the number of *loaded* shards, not the "_of_T" total: a
    // bundle whose leading shard runs off-graph (e.g. a CPU-side embedding LUT,
    // so the first ctx-bin is `2_of_3`) exposes fewer graphs than T. Ranking the
    // actual shard numbers into a dense 0-based index keeps graphIndex() and the
    // per-shard buffers sized to what was really loaded.
    shard_count_ = shard_set.size();
    // Only a LEADING shard may be absent (an off-graph embedding LUT makes the
    // first ctx-bin 2_of_3). A gap in the middle -- e.g. {1,3} of _of_3 -- would
    // be ranked into a dense {0,1} and silently mis-wire the inter-shard hidden
    // state, so reject any non-contiguous loaded set.
    if (!shard_set.empty()) {
        const size_t first = *shard_set.begin();
        const size_t last  = *shard_set.rbegin();
        if (last - first + 1 != shard_set.size()) {
            GENIEX_LOG_ERROR(
                "LLMModel: loaded shards are non-contiguous (a middle shard is missing); "
                "first={} last={} count={}",
                first,
                last,
                shard_set.size());
            return false;
        }
    }
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

    // graphIndex() addresses graphs_ as a dense phase x shard x cl grid, so every
    // combination must be present exactly once. A gap or a duplicate would silently
    // resolve to the wrong graph rather than fail, and nothing else checks it.
    //
    // A bundle whose shards ship different context-length variant sets lands here,
    // as does anything that loads only a subset of the variants a bundle contains.
    {
        // A single-AR bundle is a legitimate 1 x shards x cls grid, and
        // graphIndex(phase=0, ...) addresses it correctly.
        const size_t                        phase_count = (ar_set.size() == 1) ? 1 : 2;
        std::set<std::tuple<int, int, int>> keys;
        for (const auto& g : graphs_) keys.insert(sortKey(g.name()));

        const size_t expected = phase_count * shard_count_ * num_cl_;
        if (keys.size() != graphs_.size() || keys.size() != expected) {
            GENIEX_LOG_ERROR(
                "LLMModel: loaded graphs do not form a complete {}x{}x{} (phase x shard x "
                "context-length) grid: {} graphs occupying {} distinct slots, expected {}. "
                "Every context length must be present for every shard in every phase.",
                phase_count,
                shard_count_,
                num_cl_,
                graphs_.size(),
                keys.size(),
                expected);
            return false;
        }
    }

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

    resolveKVLayout();

    // INFO, not DEBUG: this is the ground truth for what was actually loaded --
    // derived from the graphs themselves, not from what was requested -- so it is the
    // line to check when a bundle turns out to expose fewer context-length variants
    // than expected. Max usable context is the last entry.
    GENIEX_LOG_INFO(
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
    createRoPEProviders();
}

void LLMModel::createRoPEProviders() {
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

void LLMModel::runShard(size_t shard, size_t phase, size_t cl_idx, const LLMRunContext& ctx,
    const std::function<void(Graph&)>& extra_inputs) {
    const size_t kv_len = kvLen(phase, cl_idx);
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
        auto mask = get_attention_mask(ctx.n_past,
            ctx.curr_len,
            seq_len,
            kv_len,
            kvMaskWidth(phase, cl_idx),
            kvNewBase(phase, cl_idx, ctx.n_past));
        g.write(spec_.attention_mask_name, mask.data(), mask.size());
    }

    // Must match the mask's new_base above -- both describe the same write cursor.
    writeCacheIndex(g, kvNewBase(phase, cl_idx, ctx.n_past));

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

    // Caller-supplied inputs the provider chain does not cover (speculative RoPE / feature seed).
    if (extra_inputs) extra_inputs(g);

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

size_t LLMModel::kvLen(size_t phase, size_t cl_idx) const {
    // Scatter cache: kv_len is the whole CL, phase-independent, so restride is a no-op.
    if (kv_scatter_) return spec_.context_lengths[cl_idx];

    const size_t seq_len = (phase == 0) ? spec_.seq_len_prefill : spec_.seq_len_decode;

    // Reserved tail is round32(AR) under HMX tiling, else AR itself. No-op for
    // AR 32/128.
    const size_t reserved = native_kv_ ? ((seq_len + kv::TILE_GRAIN - 1) / kv::TILE_GRAIN) * kv::TILE_GRAIN : seq_len;
    return spec_.context_lengths[cl_idx] - reserved;
}

// Shift a fixed-window KV input buffer left by `shift` tokens (dropping the
// oldest), making room to append fresh tokens at the tail. Used only by
// sliding-window caches once their window fills.
void LLMModel::shiftKVLeft(Graph& g, const std::string& name, size_t shift, bool is_key) {
    if (shift == 0) return;
    const TensorSpec& spec = g.inputSpec(name);
    const auto        geo  = kv::geometryOf(spec, is_key);
    if (shift >= geo.kv_len) return;

    // Unaligned shift of a tiled cache can't move whole 32x32 blocks, so it degrades
    // to element-wise re-tiling every step once the window fills -- warn loudly.
    if (geo.format == kv::KVFormat::HmxTiled && shift % kv::TILE_GRAIN != 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GENIEX_LOG_WARN(
                "KV '{}' is HMX_WEIGHT_LAYOUT and shifting by {} (not a multiple of {}): the window "
                "slide runs element-wise and will be slow",
                name,
                shift,
                kv::TILE_GRAIN);
        }
    }

    kv::shiftLeft(geo, static_cast<uint8_t*>(g.inputPtr(name)), shift, kv::zeroPatternFor(geo.format, spec.dtype));
}

void LLMModel::copyKV(Graph& src_g, const std::string& src_name, bool src_is_output, Graph& dst_g,
    const std::string& dst_name, size_t src_off, size_t dst_off, size_t n_tok, bool is_key) {
    const TensorSpec& src_spec = src_is_output ? src_g.outputSpec(src_name) : src_g.inputSpec(src_name);
    const TensorSpec& dst_spec = dst_g.inputSpec(dst_name);

    const auto* src_buf =
        static_cast<const uint8_t*>(src_is_output ? src_g.outputPtr(src_name) : src_g.inputPtr(src_name));
    auto* dst_buf = static_cast<uint8_t*>(dst_g.inputPtr(dst_name));

    // Derived per-tensor, not from global spec_.{num_kv_heads,head_dim}: a model may
    // own multiple KV blocks with different head dims (Gemma3/4's global vs swa_*
    // caches), and each side's own dataFormat lets a tiled dst be fed by a flat src.
    const auto src_geo = kv::geometryOf(src_spec, is_key);
    const auto dst_geo = kv::geometryOf(dst_spec, is_key);

    if (dst_off + n_tok > dst_geo.kv_len)
        throw std::runtime_error("copyKV: write [" + std::to_string(dst_off) + ", " + std::to_string(dst_off + n_tok) +
                                 ") exceeds dst '" + dst_name + "' capacity " + std::to_string(dst_geo.kv_len));

    kv::copyTokens(dst_geo, dst_buf, src_geo, src_buf, src_off, dst_off, n_tok, kvRebaseFor(dst_name));
}

// Write cursor for this pass's fresh KV: n_past for scatter, round32(n_past)
// for a native/tiled scatter cache (block-granular scatter-write), kv_len
// (i.e. after the cached region) for concat.
size_t LLMModel::kvNewBase(size_t phase, size_t cl_idx, size_t n_past) const {
    if (!kv_scatter_) return kvLen(phase, cl_idx);
    if (!native_kv_) return n_past;
    return ((n_past + kv::TILE_GRAIN - 1) / kv::TILE_GRAIN) * kv::TILE_GRAIN;
}

size_t LLMModel::kvMaskWidth(size_t phase, size_t cl_idx) const {
    const size_t seq_len = (phase == 0) ? spec_.seq_len_prefill : spec_.seq_len_decode;
    return kv_scatter_ ? kvLen(phase, cl_idx) : kvLen(phase, cl_idx) + seq_len;
}

void LLMModel::writeCacheIndex(Graph& g, size_t index) const {
    if (spec_.cache_index_name.empty() || !g.hasInput(spec_.cache_index_name)) return;
    const size_t         n   = g.inputSpec(spec_.cache_index_name).elementCount();
    const auto           val = static_cast<int32_t>(index);
    std::vector<int32_t> buf(n, val);
    g.write(spec_.cache_index_name, buf.data(), buf.size());
}

int LLMModel::kvRebaseFor(const std::string& kv_in_name) const {
    const auto it = kv_rebase_.find(kv_in_name);
    return it == kv_rebase_.end() ? 0 : it->second;  // resolved once per tensor at init
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
        // once the window is full each write shifts the buffer left by n_tok
        // before appending. For dst_off < window this is a plain append.
        const size_t kv_capacity = kvCapacityOf(g, pairs.front().key_in, /*is_key=*/true);
        // dst_off is absolute and keeps growing; the buffer never holds more
        // than kv_capacity, so the shift must come from the occupancy.
        const size_t occupancy = std::min(dst_off, kv_capacity);
        size_t       off       = dst_off;
        if (occupancy + n_tok > kv_capacity) {
            // Window overflow: drop the oldest (occupancy + n_tok - kv_capacity)
            // tokens so the newest n_tok land at the tail.
            const size_t shift = occupancy + n_tok - kv_capacity;
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

        // Key: [num_kv_heads, 1, head_dim, kv_len]
        // Value: [num_kv_heads, 1, kv_len, head_dim]
        // kv::restride handles both layouts; for a tiled cache it is still pure
        // memmove (see its comment), so the prefill<->decode stride switch costs
        // the same as it does for a flat cache.
        auto restrideOne = [&](const std::string& name, bool is_key) {
            const TensorSpec& spec = g.inputSpec(name);
            const auto        geo  = kv::geometryOf(spec, is_key);
            kv::restride(geo,
                static_cast<uint8_t*>(g.inputPtr(name)),
                old_kv_len,
                new_kv_len,
                copy_len,
                kv::zeroPatternFor(geo.format, spec.dtype));
        };
        for (const auto& p : pairs) {
            restrideOne(p.key_in, /*is_key=*/true);
            restrideOne(p.value_in, /*is_key=*/false);
        }
    }
}

bool LLMModel::promoteCL(size_t required, size_t capacity_phase, size_t stride_phase) {
    if (num_cl_ <= 1) return false;

    size_t new_cl = active_cl_idx_;
    while (new_cl + 1 < num_cl_ && kvLen(capacity_phase, new_cl) < required) {
        ++new_cl;
    }
    if (new_cl == active_cl_idx_) return false;

    GENIEX_LOG_DEBUG("Upgrading CL from {} to {}", active_cl_idx_, new_cl);
    const size_t old_kv = kvLen(stride_phase, active_cl_idx_);
    const size_t new_kv = kvLen(stride_phase, new_cl);
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
        for (size_t s = 0; s < shard_count_; ++s)
            reshapeKV(s, kvLen(/*phase=*/1, active_cl_idx_), kvLen(/*phase=*/0, active_cl_idx_), n_keep);
    }

    n_past_ = n_keep;
    token_history_.resize(n_keep);

    prefillChunks(tail_tokens, /*last_chunk_size_out=*/nullptr);

    if (at_decode_stride) {
        for (size_t s = 0; s < shard_count_; ++s)
            reshapeKV(s, kvLen(/*phase=*/0, active_cl_idx_), kvLen(/*phase=*/1, active_cl_idx_), n_past_);
    }
}

// Chunked prefill over `tokens`, writing fresh KV starting at the current n_past_ and advancing
// n_past_ / token_history_ as each chunk completes. Assumes the KV buffer is already strided for
// prefill; callers at decode stride must reshapeKV first (see slideWindowEvict). Extracted from
// generate()'s main prefill loop so slideWindowEvict's re-prefill can reuse the same logic.
void LLMModel::prefillChunks(
    const std::vector<int32_t>& tokens, size_t* last_chunk_size_out, std::vector<float>* all_logits_out) {
    // Collecting per-position logits requires the LM head to run on every chunk.
    const bool collect_logits = all_logits_out != nullptr;

    PrefillHooks hooks;
    hooks.run_lm_head_every_chunk = collect_logits;
    if (collect_logits) {
        // Append this chunk's logits row-major ([chunk_size, vocab_size]) once the LM head has run.
        hooks.on_chunk_done = [this, all_logits_out](size_t chunk_size) {
            const size_t vocab = spec_.vocab_size;
            const size_t base  = all_logits_out->size();
            all_logits_out->resize(base + chunk_size * vocab);
            const Graph& lm_head = graph(graphIndex(/*phase=*/0, /*shard=*/shard_count_ - 1, active_cl_idx_));
            lm_head.read(spec_.shards.back().out_state_name,
                all_logits_out->data() + base,
                chunk_size * vocab,
                /*elem_offset=*/0);
        };
    }

    prefillLoop(tokens, hooks, last_chunk_size_out);
}

void LLMModel::prefillLoop(const std::vector<int32_t>& tokens, const PrefillHooks& hooks, size_t* last_chunk_size_out) {
    size_t       tokens_processed = 0;
    const size_t total_tokens     = tokens.size();

    while (tokens_processed < total_tokens) {
        const size_t remaining      = total_tokens - tokens_processed;
        const size_t chunk_size     = std::min(remaining, spec_.seq_len_prefill);
        const bool   is_final_chunk = (tokens_processed + chunk_size >= total_tokens);
        if (last_chunk_size_out) *last_chunk_size_out = chunk_size;

        // Ensure the prefill KV buffer (CL - seq_len_prefill) can hold n_past + chunk_size after this chunk.
        promoteCL(/*required=*/n_past_ + chunk_size, /*capacity_phase=*/0, /*stride_phase=*/0);

        const std::vector<int32_t> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(tokens_processed),
            tokens.begin() + static_cast<std::ptrdiff_t>(tokens_processed + chunk_size));

        GENIEX_LOG_DEBUG("prefill chunk: tokens [{}, {}) cl_idx={} final={}",
            tokens_processed,
            tokens_processed + chunk_size,
            active_cl_idx_,
            is_final_chunk);
        const LLMRunContext ctx{chunk, n_past_, chunk_size, /*phase=*/0};

        // Non-final chunks only need the KV cache populated, so the LM-head-only shard can be
        // skipped -- unless a hook needs its output (e.g. per-chunk logits) on every chunk.
        const bool run_lm_head = is_final_chunk || hooks.run_lm_head_every_chunk;

        // Bind the hook's extra shard inputs (speculative RoPE / feature seed) to this chunk's offset.
        std::function<void(Graph&)> extra_inputs;
        if (hooks.write_shard_inputs) {
            extra_inputs = [&hooks, &ctx, tokens_processed](
                               Graph& g) { hooks.write_shard_inputs(g, ctx, tokens_processed); };
        }

        for (size_t s = 0; s < shard_count_; ++s) {
            // For non-final prefill chunks the LM-head shard is pure output and can be skipped,
            // avoiding a TTFT-inflating full LM head on every chunk.
            if (!run_lm_head && spec_.shards[s].lm_head_only) {
                GENIEX_LOG_DEBUG("skipping LM-head-only shard {} on non-final prefill chunk", s);
                continue;
            }
            runShard(s, /*phase=*/0, active_cl_idx_, ctx, extra_inputs);
            updateKV(s, /*phase=*/0, n_past_, chunk_size);
            if (s + 1 < shard_count_) {
                if (!run_lm_head && spec_.shards[s + 1].lm_head_only) {
                    continue;
                }
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }
        }

        if (hooks.on_chunk_done) hooks.on_chunk_done(chunk_size);

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
            throw PromptTooLongError("geniex: prompt exceeds max context length (" + std::to_string(max_cl) + ")");
        }
    }

    prefillChunks(prompt_tokens, &last_chunk_size);

    // Switch KV stride from prefill to decode before entering the decode loop.
    {
        for (size_t s = 0; s < shard_count_; ++s)
            reshapeKV(s, kvLen(/*phase=*/0, active_cl_idx_), kvLen(/*phase=*/1, active_cl_idx_), n_past_);
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
        promoteCL(/*required=*/n_past_ + 1, /*capacity_phase=*/1, /*stride_phase=*/1);

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
    promoteCL(/*required=*/n_past_, /*capacity_phase=*/0, /*stride_phase=*/1);
    {
        for (size_t s = 0; s < shard_count_; ++s)
            reshapeKV(s, kvLen(/*phase=*/1, active_cl_idx_), kvLen(/*phase=*/0, active_cl_idx_), n_past_);
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

void LLMModel::resolveKVLayout() {
    kv_rebase_.clear();
    native_kv_  = false;
    kv_scatter_ = false;

    // A scatter cache announces itself two ways at once: the body graphs take a
    // cache_index input, and their kv_in spans a whole context length rather than
    // CL - AR. Require both, so a bundle that merely happens to expose a
    // cache_index is not mistaken for one.
    if (kv_state_block_idx_ < spec_.state_blocks.size()) {
        const auto& block = spec_.state_blocks[kv_state_block_idx_];
        for (size_t s = 0; s < shard_count_ && !kv_scatter_; ++s) {
            if (s >= block.shard_pairs.size() || block.shard_pairs[s].empty()) continue;
            const Graph& g = graph(graphIndex(0, s, 0));
            if (!g.hasInput(spec_.cache_index_name)) continue;
            const auto& key_in = g.inputSpec(block.shard_pairs[s].front().key_in);
            if (key_in.shape.size() < 4) continue;
            const size_t declared = key_in.shape[3];
            for (size_t cl : spec_.context_lengths) {
                if (declared == cl) {
                    kv_scatter_ = true;
                    break;
                }
            }
        }
    }
    if (kv_scatter_) {
        GENIEX_LOG_INFO(
            "KV cache mode: SCATTER (kv_in spans the full context length; fresh KV placed at '{}'). "
            "Prefill/decode share one stride, so no restride between phases.",
            spec_.cache_index_name);
    }

    size_t n_tiled = 0, n_flat = 0;

    for (const auto& block : spec_.state_blocks) {
        if (!isKVStateBlock(block.kind)) continue;
        for (size_t s = 0; s < shard_count_; ++s) {
            if (s >= block.shard_pairs.size()) continue;
            for (const auto& p : block.shard_pairs[s]) {
                // Check EVERY (phase, CL) variant: they are separate compiled
                // graphs and a recipe could disagree between them, which would
                // corrupt the shared cache buffer the moment the stride switched.
                for (size_t gi = 0; gi < graphCount(); ++gi) {
                    const Graph& g = graph(gi);
                    if (!g.hasInput(p.key_in) || !g.hasOutput(p.key_out)) continue;

                    const TensorSpec& key_in  = g.inputSpec(p.key_in);
                    const TensorSpec& val_in  = g.inputSpec(p.value_in);
                    const TensorSpec& key_out = g.outputSpec(p.key_out);

                    kv::validateGeometry(kv::geometryOf(key_in, /*is_key=*/true), p.key_in);
                    kv::validateGeometry(kv::geometryOf(val_in, /*is_key=*/false), p.value_in);

                    const auto fmt = kv::formatOf(key_in);
                    if (kv::formatOf(val_in) != fmt) {
                        throw std::runtime_error("LLMModel: KV pair '" + p.key_in + "' / '" + p.value_in +
                                                 "' disagree on dataFormat in graph '" + g.name() +
                                                 "'; key and value caches must share a layout");
                    }
                    if (fmt == kv::KVFormat::HmxTiled) {
                        ++n_tiled;
                        // kvLen()'s reserved tail becomes 32-granular once the
                        // PRIMARY cache is tiled; a tiled swa_* block alone must
                        // not change the global cache's stride arithmetic.
                        if (block.kind == StateBlockKind::KV) native_kv_ = true;
                    } else {
                        ++n_flat;
                    }

                    const int rebase = kv::deriveRebase(key_in, key_out);
                    for (const auto* name : {&p.key_in, &p.value_in}) {
                        const auto [it, inserted] = kv_rebase_.emplace(*name, rebase);
                        if (!inserted && it->second != rebase) {
                            throw std::runtime_error("LLMModel: KV tensor '" + *name +
                                                     "' needs conflicting rebases across graph variants (" +
                                                     std::to_string(it->second) + " vs " + std::to_string(rebase) +
                                                     "); the KV output dataFormat/dtype is inconsistent");
                        }
                    }
                }
            }
        }
    }

    if (n_tiled == 0) {
        GENIEX_LOG_DEBUG("KV cache layout: FLAT_BUFFER ({} tensors)", n_flat);
        return;
    }
    if (n_flat != 0) {
        // Mixing layouts across variants of the SAME tensor is caught above; this
        // is a model with two KV blocks that disagree, which we support but should
        // not see by accident.
        GENIEX_LOG_WARN("KV cache layout is mixed: {} HMX_WEIGHT_LAYOUT tensors, {} FLAT_BUFFER", n_tiled, n_flat);
    }
    // Native KV changes how every cache byte is addressed, so say so at INFO: a
    // bundle that silently ran the flat path would produce plausible garbage.
    int rebase = 0;
    for (const auto& [_, r] : kv_rebase_) rebase = r;
    GENIEX_LOG_INFO(
        "KV cache layout: HMX_WEIGHT_LAYOUT (native KV) on {} tensors, output rebase={}, clear=0x00", n_tiled, rebase);
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
            // A tiled cache clears to 0x00 rather than the dtype midpoint -- HMX
            // applies no zero-point offset to a native KV operand.
            kv::fillZero(buf, spec.byteCount(), kv::zeroPatternFor(kv::formatOf(spec), spec.dtype));
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
    const size_t    total = tokens.size();
    RotaryEmbedding rope(spec_.head_dim, rope_theta);

    // Reject a prompt that cannot fit rather than overrunning the KV buffer
    // mid-chunk. The prefill KV inputs are strided to (CL - seq_len_prefill), so
    // that -- not the raw CL -- is the real capacity: a prompt landing in the top
    // seq_len_prefill band would pass a bare `> max_cl` guard and then trip
    // updateKV's window-overflow path, which silently shifts the cache left and
    // drops the oldest tokens (desyncing KV from position/EAGLE bookkeeping with
    // no diagnostic). This driver-facing prefill has no sliding-window fallback --
    // generate() owns that path.
    const size_t max_cl      = spec_.context_lengths.back();
    const size_t prefill_cap = max_cl - spec_.seq_len_prefill;
    if (n_past_ + total > prefill_cap)
        throw ContextLengthExceededError("geniex: prefill exceeds usable context length (" +
                                         std::to_string(prefill_cap) + " = max CL " + std::to_string(max_cl) +
                                         " - prefill stride " + std::to_string(spec_.seq_len_prefill) + ")");

    // This path writes a plain RotaryEmbedding straight into the graph, bypassing
    // the providers. That is correct only for EAGLE, where a quantized-embedding
    // provider suppresses the RoPE provider so nothing else fills these tables.
    // A bundle whose scheme is LongRoPE/Llama3/Partial installs a RoPE provider
    // whose scaled tables this would silently overwrite with the wrong values --
    // refuse rather than corrupt positions. (The providers run before this hook,
    // so a survivor here means the model genuinely needs non-plain RoPE.)
    for (const auto& p : input_providers_) {
        const InputProvider* raw = p.get();
        if (dynamic_cast<const LongRoPEInputProvider*>(raw) || dynamic_cast<const Llama3RoPEInputProvider*>(raw) ||
            dynamic_cast<const PartialRoPEInputProvider*>(raw))
            throw std::runtime_error(
                "geniex: LLMModel::prefill writes plain RoPE and cannot serve a bundle with a scaled RoPE provider "
                "(LongRoPE/Llama3/Partial); use the pipeline decode path instead");
    }

    // Newer exports rename the global-RoPE pair to position_ids_global_*, so feed
    // whichever the graph exposes (matches createInputProviders' provider scan).
    // EAGLE engines register a quantized-embedding provider before init, which
    // suppresses the RoPE provider, so this driver writes RoPE itself.
    static constexpr const char* kRopeCos[] = {"position_ids_cos", "position_ids_global_cos"};
    static constexpr const char* kRopeSin[] = {"position_ids_sin", "position_ids_global_sin"};

    if (captured_features) captured_features->clear();

    PrefillHooks hooks;
    hooks.write_shard_inputs = [&](Graph& g, const LLMRunContext& ctx, size_t processed) {
        std::vector<int32_t> pos(ctx.curr_len);
        for (size_t i = 0; i < ctx.curr_len; ++i) pos[i] = static_cast<int32_t>(ctx.n_past + i);
        auto [cos_vec, sin_vec] = rope.forward(pos);

        if (feature_rows && !feature_name.empty() && g.hasInput(feature_name)) {
            g.write(feature_name, feature_rows + processed * feature_row_bytes, ctx.curr_len * feature_row_bytes);
        }
        for (size_t v = 0; v < 2; ++v) {
            if (g.hasInput(kRopeCos[v])) g.write(kRopeCos[v], cos_vec.data(), cos_vec.size());
            if (g.hasInput(kRopeSin[v])) g.write(kRopeSin[v], sin_vec.data(), sin_vec.size());
        }
    };

    if (captured_features && !capture_name.empty()) {
        // The prefill output buffer only retains the final chunk, so a driver that needs every
        // position's hidden state must capture the body shard's output after each chunk.
        hooks.on_chunk_done = [&](size_t chunk_size) {
            const size_t body = graphIndex(/*phase=*/0, shard_count_ >= 2 ? shard_count_ - 2 : 0, active_cl_idx_);
            const auto&  spec = graph(body).outputSpec(capture_name);
            const size_t row  = spec.shape.back() * spec.elementSize();
            const auto*  src  = static_cast<const uint8_t*>(graph(body).outputPtr(capture_name));
            captured_features->insert(captured_features->end(), src, src + chunk_size * row);
        };
    }

    prefillLoop(tokens, hooks, /*last_chunk_size_out=*/nullptr);
}

void LLMModel::startClockKeeper() {
    if (decode_pool_ && clock_keeper_threads_ > 0)
        decode_pool_->startClockKeeper(clock_keeper_threads_, decode_cpu_mask_);
}

void LLMModel::stopClockKeeper() {
    if (decode_pool_) decode_pool_->stopClockKeeper();
}

}  // namespace geniex

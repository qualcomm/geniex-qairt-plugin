// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// CPU-reachable EAGLE decode logic. Device bring-up (EagleModel::initialize,
// which drives the target/draft QNN backend load) lives in eagle_model_init.cpp
// so this file's speculation loop can be measured by the coverage surface.

#include "llm/eagle_model.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "graph.h"
#include "logging.h"

namespace geniex {

namespace {

#ifdef GENIEX_DEBUG
// Comma-separated rendering of a small vector, for trace logs.
template <typename T>
std::string joinVec(const std::vector<T>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    return os.str();
}
#endif

}  // namespace

EagleModel::EagleModel(LLMSpec target_spec, LLMSpec draft_spec, EagleConfig cfg) : cfg_(std::move(cfg)) {
    target_ = std::make_unique<SpeculativeLLMModel>(std::move(target_spec));
    draft_  = std::make_unique<SpeculativeLLMModel>(std::move(draft_spec));
}

SpeculativeLLMModel& EagleModel::target() {
    if (!target_) throw std::runtime_error("EagleModel: target engine not constructed");
    return *target_;
}

const SpeculativeLLMModel& EagleModel::target() const {
    if (!target_) throw std::runtime_error("EagleModel: target engine not constructed");
    return *target_;
}

SpeculativeLLMModel& EagleModel::draft() {
    if (!draft_) throw std::runtime_error("EagleModel: draft engine not constructed");
    return *draft_;
}

const SpeculativeLLMModel& EagleModel::draft() const {
    if (!draft_) throw std::runtime_error("EagleModel: draft engine not constructed");
    return *draft_;
}

void EagleModel::resetKVCache() {
    target().resetKVCache();
    if (ready_) draft().resetKVCache();
    has_pending_ = false;
    carry_feat_.clear();
}

void EagleModel::saveKVCacheToFile(const std::string&) const {
    throw std::runtime_error("EagleModel: saving the KV cache is not supported");
}

void EagleModel::loadKVCacheFromFile(const std::string&) {
    throw std::runtime_error("EagleModel: loading the KV cache is not supported");
}

[[noreturn]] std::vector<float> EagleModel::forwardLogits(const std::vector<int32_t>&, bool) {
    throw std::runtime_error("EagleModel: forwardLogits is not supported");
}

size_t EagleModel::nPast() const { return target().nPast() + (has_pending_ ? 1 : 0); }

size_t EagleModel::vocabSize() const { return target().vocabSize(); }

int32_t EagleModel::argmaxTarget(size_t phase, size_t row) const {
    const LLMSpec& s     = target().spec();
    const size_t   vocab = s.vocab_size;
    // Read the graph that actually ran: prefillLoop promotes the active CL per
    // chunk, so even phase 0 can finish at active_cl_idx_ > 0 on a long prompt;
    // a hardcoded 0 would read a CL-0 LM-head buffer that never executed and
    // yield a garbage token. The decode loop likewise promotes before each verify
    // (see generate()), so the active CL tracks the growing sequence.
    const size_t cl = target().activeContextLengthIndex();
    const size_t lm = target().graphIndex(phase, s.shards.size() - 1, cl);
    return static_cast<int32_t>(target().graph(lm).argmaxOutput(s.shards.back().out_state_name, vocab, row * vocab));
}

void EagleModel::readDraftLogits(size_t phase, size_t row, std::vector<float>& out) const {
    const LLMSpec& ds = draft().spec();
    // Same rule as argmaxTarget: read the CL that actually ran this phase.
    const size_t cl = draft().activeContextLengthIndex();
    const size_t lm = draft().graphIndex(phase, ds.shards.size() - 1, cl);
    const Graph& g  = draft().graph(lm);
    // The draft's logits are a distinct tensor from its shard "state" (hidden
    // feature) output; initialize() guarantees draft_logits_name is set.
    const std::string& logits_name = cfg_.draft_logits_name;
    const auto&        os          = g.outputSpec(logits_name);
    // Row stride follows the graph's real last dim, which may be padded above
    // the true draft vocabulary (e.g. 2560 tensor slots for a 2255-token draft
    // vocab). Read the full padded row for correct striding, then expose only
    // the real vocab so padding logits can never be proposed. The draft-token
    // map (draft index -> target id) sizes the real vocab; without a map the
    // draft already emits full-vocab ids, so the tensor width is authoritative.
    const size_t tensor_vocab = os.shape.empty() ? ds.vocab_size : os.shape.back();
    const size_t vocab =
        cfg_.draft_token_map.empty() ? tensor_vocab : std::min(tensor_vocab, cfg_.draft_token_map.size());
    out.resize(tensor_vocab);
    g.read(logits_name, out.data(), tensor_vocab, row * tensor_vocab);
    out.resize(vocab);
}

void EagleModel::topKDraftWithProbs(
    size_t phase, size_t row, size_t k, std::vector<int32_t>& tokens_out, std::vector<float>& probs_out) const {
    std::vector<float> logits;
    readDraftLogits(phase, row, logits);
    const size_t vocab = logits.size();

    // Maps a draft logit index to the full-vocab token id. readDraftLogits
    // already clamps the ranked window to the map size, so this bounds check is
    // defense-in-depth against a future caller that ranks a wider window.
    auto to_token = [this, vocab](size_t raw) -> int32_t {
        if (cfg_.draft_token_map.empty()) return static_cast<int32_t>(raw);
        if (raw >= cfg_.draft_token_map.size())
            throw std::runtime_error("EagleModel: draft_token_map is smaller than the draft vocab (" +
                                     std::to_string(cfg_.draft_token_map.size()) + " < " + std::to_string(vocab) + ")");
        return cfg_.draft_token_map[raw];
    };

    k = std::min(k, vocab);
    tokens_out.clear();
    probs_out.clear();
    if (k == 0) return;

    // Single-branch decode (the default) never ranks siblings, so the calibrated
    // softmax is dead work: a plain argmax over the row is all that is needed.
    // Probability 1.0 is a harmless placeholder -- a lone child cannot be pruned.
    if (k == 1) {
        size_t best = 0;
        for (size_t i = 1; i < vocab; ++i)
            if (logits[i] > logits[best]) best = i;
        tokens_out.push_back(to_token(best));
        probs_out.push_back(1.0f);
        return;
    }

    std::vector<size_t> idx(vocab);
    std::iota(idx.begin(), idx.end(), size_t{0});
    std::partial_sort(
        idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(), [&logits](size_t a, size_t b) {
            return logits[a] > logits[b];
        });

    // Full-row softmax gives calibrated probabilities so cumulative-probability
    // pruning across sibling branches is comparable across levels. Subtract the
    // max for numerical stability. partial_sort already placed the largest logit
    // at idx[0], so reuse it as the softmax max instead of a second full-row scan.
    const float max_logit = logits[idx[0]];
    double      denom     = 0.0;
    for (float v : logits) denom += std::exp(static_cast<double>(v) - max_logit);

    tokens_out.reserve(k);
    probs_out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t raw = idx[i];
        tokens_out.push_back(to_token(raw));
        probs_out.push_back(static_cast<float>(std::exp(static_cast<double>(logits[raw]) - max_logit) / denom));
    }
}

// Grows a speculative token tree from the anchor using the draft decode graph.
//
// A level-synchronous, multi-branch, cumulative-probability-pruned expansion
// with per-level draft-KV commit.
//
//   * Level 0 decodes the anchor (seeded with the target feature) and samples
//     n_branches children, each carrying its own softmax probability.
//   * Each deeper level decodes the surviving frontier in one batched draft
//     forward whose tree attention references the ancestor KV committed by the
//     previous level, then samples n_branches children per frontier node.
//   * Children are ranked globally by cumulative probability
//     (cumProb = candProb * parentCumProb); only the top n_branches survive to
//     the next level (calculateTopKThreshold + markEligibleSequences).
//   * Surviving children's draft KV is committed so the next level can attend to
//     them; the whole speculative region is rewound off the draft KV before the
//     function returns, leaving n_past at the accepted length.
//
// Node positions are depth offsets from the anchor so target-side RoPE matches
// where each node would land if accepted. The target verifies the full tree in
// one pass (caller); the number of nodes is bounded by max_nodes (the target's
// verify width).
EagleModel::DraftTree EagleModel::buildDraftTree(SpeculativeLLMModel& drf, int32_t anchor_token,
    const uint8_t* anchor_feature, size_t row_bytes, float theta, size_t max_nodes) {
    DraftTree      tree;
    const LLMSpec& ds = drf.spec();
    // The draft decode graph processes at most seq_len_decode rows per forward,
    // so each level's frontier is capped to that width. n_branches is likewise
    // capped: a level can never keep more survivors than one draft forward can
    // re-expand next level.
    const size_t draft_w    = std::max<size_t>(1, ds.seq_len_decode);
    const size_t n_branches = std::min(std::max<size_t>(1, cfg_.n_branches), draft_w);
    // Depth is bounded by draft_len AND by the target's verify width: a path can
    // never be longer than the number of nodes the target verifies in one pass,
    // so growing deeper than max_nodes only produces nodes the prune step drops.
    const size_t depth_max = std::min(std::max<size_t>(1, cfg_.draft_len), max_nodes);
    const size_t base_past = drf.nPast();

    if (max_nodes == 0) return tree;  // target has no verify width to spare

    // The speculative tree commits its nodes to the draft KV starting at
    // base_past; the whole region must fit the draft's decode-strided KV buffer
    // (CL - seq_len_decode). Before the post-build rewind the tree writes its
    // PEAK row count -- the anchor plus the full unpruned frontier at every level
    // (n_branches per level, depth_max-1 deeper levels), not just the max_nodes
    // that survive pruning. Bound on that peak so a wide/deep draft can never
    // overrun the KV buffer even though most rows are later discarded.
    const size_t peak_tree_rows = 1 + (depth_max - 1) * n_branches;
    // Grow into a larger CL if the current one cannot hold the peak tree. Promote
    // BEFORE reading draft_kv_cap / d_body below so both address the (possibly
    // upgraded) active CL; the vanilla decode loop promotes per step the same way.
    drf.promoteDecodeCL(peak_tree_rows);
    // Decode-phase graph, so it must select the CL that actually runs this round.
    // The promote above (and prefillLoop's per-chunk promotes) can advance the
    // draft's active CL, so a hardcoded 0 would read a graph buffer that never
    // executed and silently mis-seed the draft feature (correct text, lost
    // acceptance).
    const size_t d_body =
        drf.graphIndex(1, ds.shards.size() >= 2 ? ds.shards.size() - 2 : 0, drf.activeContextLengthIndex());
    const size_t draft_kv_cap = drf.kvLen(/*phase=*/1, drf.activeContextLengthIndex());
    if (base_past + peak_tree_rows > draft_kv_cap) {
        // A bail returns an empty tree and the round silently degrades to one
        // token; warn once so an acceptance collapse near the context limit is
        // not mistaken for normal operation.
        static std::once_flag warned;
        std::call_once(warned, [&] {
            GENIEX_LOG_WARN(
                "EagleModel: draft KV cannot hold the speculation tree (need {} rows past {}, cap {}); "
                "this round and any like it fall back to single-token decode",
                peak_tree_rows,
                base_past,
                draft_kv_cap);
        });
        return tree;
    }

    // Per-node cumulative probability, tracked in parallel with tree.tokens so
    // the finished tree can be pruned to the target's verify width by global
    // cumulative probability.
    std::vector<float> node_cum_prob;

    // A live branch on the current frontier: its node index in the target tree,
    // the cumulative probability of the path ending at it, and the draft feature
    // seed its own decode uses (the draft body output of the parent that produced
    // it). kv_ancestors are the committed draft-KV rows this node attends to (its
    // ancestors' committed rows, walked from the anchor down), used to build the
    // next level's tree attention against the speculative draft cache.
    struct Branch {
        int32_t              node;          // index into tree.tokens
        float                cum_prob;      // path cumulative probability
        std::vector<uint8_t> feature;       // draft feature seed for this node's decode
        std::vector<int32_t> kv_ancestors;  // committed draft rows this node attends to
    };

    // Level 0: decode the anchor (seeded with the target feature), sample its
    // n_branches continuations. The anchor's own draft KV is committed at row
    // base_past so level-1 nodes attend to it.
    {
        std::vector<int32_t> tok = {anchor_token};
        std::vector<int32_t> pos = {static_cast<int32_t>(base_past)};
        drf.decodeBatch(
            tok, pos, /*attention_map=*/{}, base_past, theta, anchor_feature, row_bytes, cfg_.draft_feature_input);
        drf.commitDecodeRows(/*selected=*/{true}, /*n_accepted=*/1);  // anchor KV at row base_past
    }

    std::vector<uint8_t> anchor_out(row_bytes);
    std::memcpy(
        anchor_out.data(), static_cast<const uint8_t*>(drf.outputBytes(d_body, cfg_.draft_feature_output)), row_bytes);

    std::vector<int32_t> l0_tokens;
    std::vector<float>   l0_probs;
    topKDraftWithProbs(/*phase=*/1, /*row=*/0, n_branches, l0_tokens, l0_probs);

    const std::vector<int32_t> anchor_ancestor = {static_cast<int32_t>(base_past)};
    std::vector<Branch>        frontier;
    for (size_t b = 0; b < l0_tokens.size(); ++b) {
        const int32_t node = static_cast<int32_t>(tree.tokens.size());
        tree.tokens.push_back(l0_tokens[b]);
        tree.parent.push_back(-1);
        tree.depth.push_back(1);
        tree.features.insert(tree.features.end(), anchor_out.begin(), anchor_out.end());
        node_cum_prob.push_back(l0_probs[b]);
        frontier.push_back(Branch{node, l0_probs[b], anchor_out, anchor_ancestor});
    }

    // Deeper levels: expand the surviving frontier, keep the n_branches most
    // probable children (per-level cap), commit the survivors' draft KV, recurse.
    // The full tree spans all draft_len levels; it is pruned to the verify width
    // by global cumulative probability afterwards.
    for (size_t depth = 2; depth <= depth_max && !frontier.empty(); ++depth) {
        // Decode every frontier node in one draft forward against the speculative
        // draft cache: each node attends to the real history [0, base_past) plus
        // its own committed ancestor rows (kv_ancestors), never its siblings.
        const size_t                      batch_n = std::min(frontier.size(), draft_w);
        std::vector<int32_t>              batch_tokens;
        std::vector<int32_t>              batch_pos;
        std::vector<std::vector<int32_t>> batch_kv_ancestors;
        std::vector<uint8_t>              batch_feat;
        batch_feat.reserve(batch_n * row_bytes);
        for (size_t i = 0; i < batch_n; ++i) {
            batch_tokens.push_back(tree.tokens[static_cast<size_t>(frontier[i].node)]);
            batch_pos.push_back(static_cast<int32_t>(base_past + depth - 1));
            batch_kv_ancestors.push_back(frontier[i].kv_ancestors);
            batch_feat.insert(batch_feat.end(), frontier[i].feature.begin(), frontier[i].feature.end());
        }
        const size_t frontier_row0 = drf.nPast();  // committed draft-KV row of frontier[0]
        drf.decodeBatchTree(batch_tokens,
            batch_pos,
            /*attention_map=*/{},
            batch_kv_ancestors,
            /*n_keep=*/base_past,
            drf.nPast(),
            theta,
            batch_feat.data(),
            batch_feat.size(),  // one feature row per frontier node
            cfg_.draft_feature_input);
        // Commit this level's frontier KV so the next level attends to it. Each
        // frontier[i] now lives at draft row frontier_row0 + i.
        drf.commitDecodeRows(std::vector<bool>(batch_n, true), batch_n);

        // Sample n_branches children per frontier node, forming a global candidate
        // pool ranked by cumulative probability.
        struct Cand {
            int32_t              parent_node;  // target-tree index of the frontier node
            int32_t              token;
            float                cum_prob;
            std::vector<uint8_t> feature;       // this frontier node's draft output feature
            std::vector<int32_t> kv_ancestors;  // child's committed-ancestor rows
        };
        std::vector<Cand> cands;
        for (size_t i = 0; i < batch_n; ++i) {
            std::vector<int32_t> ctok;
            std::vector<float>   cprob;
            topKDraftWithProbs(/*phase=*/1, /*row=*/i, n_branches, ctok, cprob);

            std::vector<uint8_t> feat(row_bytes);
            std::memcpy(feat.data(),
                static_cast<const uint8_t*>(drf.outputBytes(d_body, cfg_.draft_feature_output)) + i * row_bytes,
                row_bytes);
            // A child of frontier[i] attends to frontier[i]'s ancestors plus
            // frontier[i]'s own committed row (frontier_row0 + i).
            std::vector<int32_t> child_anc = frontier[i].kv_ancestors;
            child_anc.push_back(static_cast<int32_t>(frontier_row0 + i));
            for (size_t c = 0; c < ctok.size(); ++c)
                cands.push_back(Cand{frontier[i].node, ctok[c], cprob[c] * frontier[i].cum_prob, feat, child_anc});
        }
        if (cands.empty()) break;

        // Keep the n_branches most probable children this level; partial_sort
        // keeps only the survivors.
        {
            const size_t keep = std::min(n_branches, cands.size());
            std::partial_sort(cands.begin(),
                cands.begin() + static_cast<std::ptrdiff_t>(keep),
                cands.end(),
                [](const Cand& a, const Cand& b) { return a.cum_prob > b.cum_prob; });

            std::vector<Branch> next_frontier;
            for (size_t c = 0; c < keep; ++c) {
                const int32_t node = static_cast<int32_t>(tree.tokens.size());
                tree.tokens.push_back(cands[c].token);
                tree.parent.push_back(cands[c].parent_node);
                tree.depth.push_back(static_cast<int32_t>(depth));
                tree.features.insert(tree.features.end(), cands[c].feature.begin(), cands[c].feature.end());
                node_cum_prob.push_back(cands[c].cum_prob);
                next_frontier.push_back(Branch{node, cands[c].cum_prob, cands[c].feature, cands[c].kv_ancestors});
            }
            frontier = std::move(next_frontier);
        }
    }

    // Rewind the speculative draft KV: only the accepted path (decided later by
    // the caller from the target verify) is re-committed against the persistent
    // draft cache. The stale rows we wrote here are overwritten on the next
    // commit, re-anchoring at the accepted draft length.
    drf.rewindKVCache(base_past);

    // Global prune to the target's verify width by cumulative probability.
    // cumProb is monotonically non-increasing down
    // the tree (each level multiplies by a probability <= 1), so keeping the
    // top-max_nodes nodes is automatically parent-closed: a kept node's parent has
    // >= cumProb and is therefore also kept. Nodes are then reindexed in their
    // original (level-order) sequence so parent/depth stay valid.
    if (tree.tokens.size() > max_nodes) tree = pruneTreeByCumProb(tree, node_cum_prob, max_nodes, row_bytes);

#ifdef GENIEX_DEBUG
    GENIEX_LOG_TRACE("eagle tree: anchor={} nodes={} tokens=[{}] parents=[{}] depths=[{}]",
        anchor_token,
        tree.tokens.size(),
        joinVec(tree.tokens),
        joinVec(tree.parent),
        joinVec(tree.depth));
#endif
    return tree;
}

// Keeps the max_nodes highest-cumProb nodes and reindexes the tree. Relies on
// the monotonic-cumProb invariant so the kept set is parent-closed.
EagleModel::DraftTree EagleModel::pruneTreeByCumProb(
    const DraftTree& in, const std::vector<float>& cum_prob, size_t max_nodes, size_t row_bytes) {
    const size_t n = in.tokens.size();

    // Rank nodes by cumProb; the top max_nodes survive. Ties broken by original
    // index so the selection is deterministic.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(max_nodes), order.end(), [&](size_t a, size_t b) {
            return cum_prob[a] != cum_prob[b] ? cum_prob[a] > cum_prob[b] : a < b;
        });

    std::vector<bool> keep(n, false);
    for (size_t i = 0; i < max_nodes; ++i) keep[order[i]] = true;

    // Reindex kept nodes in original level order so parent indices stay ordered
    // (a parent always precedes its children).
    std::vector<int32_t> remap(n, -1);
    DraftTree            out;
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        remap[i] = static_cast<int32_t>(out.tokens.size());
        out.tokens.push_back(in.tokens[i]);
        out.depth.push_back(in.depth[i]);
        const int32_t p = in.parent[i];
        out.parent.push_back(p < 0 ? -1 : remap[static_cast<size_t>(p)]);
        out.features.insert(out.features.end(),
            in.features.begin() + static_cast<std::ptrdiff_t>(i * row_bytes),
            in.features.begin() + static_cast<std::ptrdiff_t>((i + 1) * row_bytes));
    }
    return out;
}

std::vector<int32_t> EagleModel::generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
    std::function<bool(int32_t)> token_callback) {
    if (!ready_) throw std::runtime_error("EagleModel::generate called before initialize()");
    if (prompt_tokens.empty()) return {};

    SpeculativeLLMModel& tgt   = target();
    SpeculativeLLMModel& drf   = draft();
    const LLMSpec&       ts    = tgt.spec();
    const LLMSpec&       ds    = drf.spec();
    const float          theta = cfg_.rope_theta;

    stats_ = EagleStats{};

    // A first turn starts from empty caches. Later turns find the previous ones
    // in place: the target committed up to its last position, the draft one row
    // behind it, and the last emitted token still pending.
    const bool fresh = carry_feat_.empty();
    if (fresh) resetKVCache();

    // Tokens the target still has to commit: the pending token, then the prompt.
    std::vector<int32_t> stream;
    stream.reserve(prompt_tokens.size() + 1);
    if (has_pending_) stream.push_back(pending_token_);
    stream.insert(stream.end(), prompt_tokens.begin(), prompt_tokens.end());

    // The draft consumes the stream shifted by one on a first turn (see the
    // seeding below), and unshifted once it is already one row behind.
    const size_t draft_rows = fresh ? stream.size() - 1 : stream.size();

    // Refuse a prompt that cannot fit before touching any state, so a rejected
    // turn leaves the conversation intact. Prefill capacity is the context length
    // minus the prefill stride.
    if (tgt.nPast() + stream.size() > ts.context_lengths.back() - ts.seq_len_prefill ||
        drf.nPast() + draft_rows > ds.context_lengths.back() - ds.seq_len_prefill) {
        throw PromptTooLongError("geniex: prompt does not fit the remaining context length");
    }

    bool seeded        = false;  // draft seeded and carry feature set: state is consistent
    bool decode_stride = false;  // both engines are at the decode stride

    // Orders any in-flight KV commit before the caches are read or reset and
    // releases the clock keepers.
    auto quiesce = [&] {
        tgt.drainDecodePool();
        drf.drainDecodePool();
        tgt.stopClockKeeper();
        drf.stopClockKeeper();
    };
    // Leaves both engines ready for the next turn's prefill.
    auto finishTurn = [&] {
        quiesce();
        if (decode_stride) {
            tgt.switchToPrefillStride();
            drf.switchToPrefillStride();
            decode_stride = false;
        }
    };

    try {
        // Prefill the target, capturing every position's hidden-state feature: the
        // prefill buffer keeps only the final chunk, but the draft seed needs all rows.
        const size_t body_pf = tgt.graphIndex(/*phase=*/0, ts.shards.size() >= 2 ? ts.shards.size() - 2 : 0, /*cl*/ 0);
        const auto&  feat_spec = tgt.outputTensorSpec(body_pf, cfg_.target_feature_output);
        const size_t hidden    = feat_spec.shape.back();
        const size_t row_bytes = hidden * feat_spec.elementSize();

        std::vector<uint8_t> stream_feats;
        tgt.prefill(stream,
            theta,
            /*feature_rows=*/nullptr,
            /*feature_row_bytes=*/0,
            /*feature_name=*/"",
            &stream_feats,
            cfg_.target_feature_output);

        // The last stream logits row is the final prefill chunk's last valid row.
        const size_t  last_chunk_rows = stream.size() - ((stream.size() - 1) / ts.seq_len_prefill) * ts.seq_len_prefill;
        const int32_t first_token     = argmaxTarget(/*phase=*/0, /*row=*/last_chunk_rows - 1);

        // Seed the draft over the stream. EAGLE's draft predicts token[i+1] from
        // embed(token[i+1]) paired with the target hidden state that PRODUCED
        // token[i+1] -- i.e. target_feat[i]. On a first turn this is realized by
        // feeding the draft the embeddings shifted left by one (drop token[0])
        // against the unshifted feature buffer, processing exactly n-1 positions
        // so the draft KV ends at position n-1. On a later turn the draft is
        // already one row behind the target, so the stream is fed unshifted and
        // its first token is paired with the carried feature. The context reaches
        // the draft entirely through these features; feeding an unshifted 1:1
        // stream conditions it on the wrong hidden state and it proposes
        // near-constant garbage.
        {
            const auto feat_end = stream_feats.begin() + static_cast<std::ptrdiff_t>((stream.size() - 1) * row_bytes);
            std::vector<int32_t> draft_tokens;
            std::vector<uint8_t> seeds;
            if (fresh) {
                draft_tokens.assign(stream.begin() + 1, stream.end());
            } else {
                draft_tokens = stream;
                seeds        = carry_feat_;
            }
            seeds.insert(seeds.end(), stream_feats.begin(), feat_end);
            // prefill() reads one feature row per token without a bound, so a
            // mismatch would read past the buffer rather than fail.
            if (seeds.size() != draft_tokens.size() * row_bytes)
                throw std::logic_error("EagleModel: draft seed rows do not match the draft tokens");
            if (!draft_tokens.empty())
                drf.prefill(draft_tokens, theta, seeds.data(), row_bytes, cfg_.draft_feature_input);
        }

        // The hidden state at the last committed position produced the next token
        // and conditions the draft's first proposal.
        carry_feat_.assign(stream_feats.end() - static_cast<std::ptrdiff_t>(row_bytes), stream_feats.end());
        has_pending_ = false;
        seeded       = true;

        for (int32_t eos : ts.eos_token_ids)
            if (first_token == eos) return {};  // stats_.generated_tokens stays 0: nothing emitted

        std::vector<int32_t> output;
        output.push_back(first_token);
        pending_token_ = first_token;
        has_pending_   = true;
        // A token the callback rejects is not part of the conversation, as in plain
        // decoding where the loop breaks before committing it.
        bool user_stop = false;
        if (token_callback && !token_callback(first_token)) user_stop = true;
        if (user_stop || gen_cfg.max_tokens <= 1) {
            if (user_stop) has_pending_ = false;
            stats_.generated_tokens = output.size();
            return output;
        }

        tgt.switchToDecodeStride();
        drf.switchToDecodeStride();
        decode_stride = true;

        // Keep both clusters from down-clocking across the speculative decode window;
        // the target verify and draft forwards run back-to-back here just like the
        // vanilla decode loop, which brackets itself the same way.
        tgt.startClockKeeper();
        drf.startClockKeeper();

        // The pending token is the next round's anchor, and the carried feature is
        // the target hidden state that produced it.
        int32_t&              last_token = pending_token_;
        std::vector<uint8_t>& last_feat  = carry_feat_;

        // Verify batch width: the target decode graph processes seq_len_decode rows
        // per forward, and row 0 is the anchor. max_nodes tree tokens plus the anchor
        // must fit. When the width is 1 the tree is empty and each round is a plain
        // single-token target decode -- still driven by the persistent in-place KV.
        const size_t verify_cap = std::min<size_t>(cfg_.max_verify_tokens, ts.seq_len_decode);
        size_t       max_nodes  = verify_cap > 1 ? verify_cap - 1 : 0;  // reserve row 0 for the anchor

        for (int step = 0; step < gen_cfg.max_tokens && static_cast<int>(output.size()) < gen_cfg.max_tokens; ++step) {
            ++stats_.iterations;

            // 1. Draft proposes a speculative tree from (last_token, last_feat).
            DraftTree tree = buildDraftTree(drf, last_token, last_feat.data(), row_bytes, theta, max_nodes);

            // 2. Target verifies [anchor, tree nodes...] in one batched forward.
            //    Row i predicts the continuation of verify token i; tree.parent maps
            //    node j (verify index j+1) to its parent verify index (anchor = 0).
            std::vector<int32_t> verify;
            std::vector<int32_t> attn;
            std::vector<int32_t> pos;
            size_t               n_verify = 0;
            {
                verify.reserve(tree.tokens.size() + 1);
                verify.push_back(last_token);
                verify.insert(verify.end(), tree.tokens.begin(), tree.tokens.end());
                n_verify = verify.size();
                attn.assign(n_verify, -1);  // anchor is root
                pos.resize(n_verify);
                pos[0] = static_cast<int32_t>(tgt.nPast());
                for (size_t j = 0; j < tree.tokens.size(); ++j) {
                    const size_t v = j + 1;
                    attn[v]        = tree.parent[j] < 0 ? 0 : tree.parent[j] + 1;  // +1: shift past the anchor row
                    pos[v]         = static_cast<int32_t>(tgt.nPast() + static_cast<size_t>(tree.depth[j]));
                }
            }
            // Verify commits into KV input buffers strided to (CL - seq_len_decode),
            // the same capacity buildDraftTree checks against. Guarding on the raw CL
            // would let the last seq_len_decode positions slip past here and fail
            // later in copyKV as a bare capacity runtime_error from another frame;
            // subtracting the stride keeps this the single, precise exhaustion signal.
            //
            // Grow into a larger CL first when the current one cannot hold this verify
            // batch, mirroring the per-step promoteCL() the vanilla decode loop runs.
            // Promote BEFORE reading verify_cl / v_body so both address the (possibly
            // upgraded) active CL.
            tgt.promoteDecodeCL(n_verify);
            const size_t verify_cl = tgt.kvLen(/*phase=*/1, tgt.activeContextLengthIndex());
            if (tgt.nPast() + n_verify > verify_cl) {
                // Already at the largest CL and the verify batch still overflows: this
                // is genuine context exhaustion. Fail loudly with the same signal as
                // plain decoding rather than silently returning a short, complete-
                // looking result.
                throw ContextLengthExceededError("geniex: EAGLE verify batch exceeds usable context length (" +
                                                 std::to_string(verify_cl) +
                                                 "); the largest available context length is exhausted");
            }
            // Order the previous round's async KV commit before this verify reads the
            // target KV inputs. On the first round this is a no-op.
            tgt.drainDecodePool();
#ifdef GENIEX_DEBUG
            GENIEX_LOG_TRACE("eagle verify: round={} n_past={} tokens=[{}] attn_parents=[{}] positions=[{}]",
                stats_.iterations,
                tgt.nPast(),
                joinVec(verify),
                joinVec(attn),
                joinVec(pos));
#endif
            tgt.decodeBatch(verify, pos, attn, tgt.nPast(), theta, /*feature_override=*/nullptr, 0, "");

            // 3. Greedy accept: walk the tree following the target's argmax at each
            //    matched node. The anchor's continuation (row 0) is always emitted;
            //    a child is accepted only when its token equals the target argmax of
            //    its parent -- so the emitted sequence is identical to plain target
            //    greedy decoding. selected[] marks the verify rows to commit.
            std::vector<int32_t> accepted;  // emitted tokens
            std::vector<bool>    selected(n_verify, false);
            std::vector<int32_t> accepted_rows;  // verify rows in accept order
            size_t               cur_row = 0;    // current matched verify row
            selected[0]                  = true;
            accepted_rows.push_back(0);
            while (true) {
                const int32_t tgt_next = argmaxTarget(/*phase=*/1, /*row=*/cur_row);
                accepted.push_back(tgt_next);
                // Find a child of cur_row whose proposed token equals tgt_next.
                int32_t next_row = -1;
                for (size_t j = 0; j < tree.tokens.size(); ++j) {
                    const size_t parent_row = tree.parent[j] < 0 ? 0 : static_cast<size_t>(tree.parent[j]) + 1;
                    if (parent_row == cur_row && tree.tokens[j] == tgt_next) {
                        next_row = static_cast<int32_t>(j + 1);
                        break;
                    }
                }
                if (next_row < 0) break;  // no matching proposal: stop, tgt_next is the bonus token
                cur_row           = static_cast<size_t>(next_row);
                selected[cur_row] = true;
                accepted_rows.push_back(static_cast<int32_t>(cur_row));
            }
            const size_t n_accept = accepted_rows.size();
#ifdef GENIEX_DEBUG
            GENIEX_LOG_TRACE("eagle accept: round={} count={} tokens=[{}] verify_rows=[{}]",
                stats_.iterations,
                n_accept,
                joinVec(accepted),
                joinVec(accepted_rows));
#endif

            // 4a. Commit the accepted target rows in place (advances target KV).
            //     Async: n_past_ advances now, the KV buffer copy runs on the decode
            //     pool and overlaps the CPU tail below (feature read, emit, replay-seed
            //     build). The next round's verify drains the pool before reading KV.
            //     The copy touches only KV INPUT rows; v_feat below is a decode OUTPUT
            //     buffer the copy never reads, so the overlap is race-free.
            tgt.commitDecodeRowsAsync(selected, n_accept);
            // Target hidden feature of each accepted row (row-major), used to advance
            // the draft KV. commitDecodeRows does not disturb the decode outputs.
            const size_t v_body = tgt.graphIndex(
                /*phase=*/1, ts.shards.size() >= 2 ? ts.shards.size() - 2 : 0, tgt.activeContextLengthIndex());
            const auto* v_feat = static_cast<const uint8_t*>(tgt.outputBytes(v_body, cfg_.target_feature_output));

            // Emit accepted tokens, stopping on EOS / callback / max-tokens. n_emit
            // counts the tokens that left this round; an EOS is not one of them.
            size_t n_emit = 0;
            bool   stop   = false;
            for (size_t i = 0; i < n_accept && !stop; ++i) {
                const int32_t tok = accepted[i];
                for (int32_t e : ts.eos_token_ids)
                    if (tok == e) stop = true;
                if (stop) break;
                output.push_back(tok);
                ++n_emit;
                if (token_callback && !token_callback(tok)) {
                    stop      = true;
                    user_stop = true;
                }
                if (static_cast<int>(output.size()) >= gen_cfg.max_tokens) stop = true;
            }

            // A round that stopped early committed more rows than were emitted. The
            // caches keep exactly the emitted prefix: the last emitted token stays
            // pending, so the next turn resumes from it.
            if (n_emit < n_accept) tgt.rewindKVCache(tgt.nPast() - (n_accept - n_emit));

            // 4b. Advance the draft KV in place: feed the emitted path
            //     [anchor, accepted proposals...] through the draft decode graph
            //     seeded with each token's target feature, committing every row. No
            //     reset, no wide re-prefill -- the draft KV tracks the committed
            //     sequence exactly, as a persistent draft-kv-cache does. The replay is
            //     chunked to the draft decode width so each forward is legal.
            if (n_emit > 0) {
                // EAGLE draft-KV advance with the embedding LEFT-SHIFT the draft was
                // trained on. buildDraftTree rewinds the draft KV to base_past (the
                // anchor row is discarded), so this replay re-establishes the cache
                // from that row using the same rule the prompt seeding and anchor use:
                //   draft row k = embed(t_{k+1}) paired with feat_k, the target hidden
                //   that PRODUCED t_{k+1}.
                // With base_past == draft nPast, row j (position base_past+j) must be:
                //   slot 0: embed(last_token) + last_feat   (re-seats the anchor row)
                //   slot j: embed(accepted[j-1]) + feat that produced accepted[j-1]
                //           == v_feat row accepted_rows[j-1].
                // Committing embed(accepted[j]) here instead advances the token index
                // one past its feature, a pairing the draft never saw in training, so
                // the next anchor is mis-conditioned and acceptance drops. The seed
                // array is already built for this pairing; the token stream is shifted
                // right by one (prepend last_token, drop the final emitted token,
                // which the NEXT round re-seats as its own anchor).
                std::vector<int32_t> replay;
                std::vector<uint8_t> replay_seeds;
                {
                    replay.reserve(n_emit);
                    replay.push_back(last_token);
                    replay.insert(
                        replay.end(), accepted.begin(), accepted.begin() + static_cast<std::ptrdiff_t>(n_emit) - 1);
                    replay_seeds.resize(n_emit * row_bytes);
                    std::memcpy(replay_seeds.data(), last_feat.data(), row_bytes);
                    for (size_t j = 1; j < n_emit; ++j)
                        std::memcpy(replay_seeds.data() + j * row_bytes,
                            v_feat + static_cast<size_t>(accepted_rows[j - 1]) * row_bytes,
                            row_bytes);
                }

                const size_t draft_w = std::max<size_t>(1, drf.spec().seq_len_decode);
                for (size_t off = 0; off < n_emit; off += draft_w) {
                    const size_t chunk = std::min(draft_w, n_emit - off);
                    // The emitted path advances the draft KV by n_emit rows this
                    // round, which can cross a CL boundary; promote per chunk like the
                    // vanilla decode loop before writing at drf.nPast().
                    drf.promoteDecodeCL(chunk);
                    std::vector<int32_t> ctok(replay.begin() + static_cast<std::ptrdiff_t>(off),
                        replay.begin() + static_cast<std::ptrdiff_t>(off + chunk));
                    std::vector<int32_t> dpos(chunk);
                    for (size_t i = 0; i < chunk; ++i) dpos[i] = static_cast<int32_t>(drf.nPast() + i);
#ifdef GENIEX_DEBUG
                    GENIEX_LOG_TRACE("eagle draft replay: round={} tokens=[{}] positions=[{}]",
                        stats_.iterations,
                        joinVec(ctok),
                        joinVec(dpos));
#endif
                    drf.decodeBatch(ctok,
                        dpos,
                        /*attention_map=*/{},
                        drf.nPast(),
                        theta,
                        replay_seeds.data() + off * row_bytes,
                        chunk * row_bytes,  // one feature row per replayed token
                        cfg_.draft_feature_input);
                    drf.commitDecodeRows(std::vector<bool>(chunk, true), chunk);
                }

                // The last emitted token is the next anchor, and the target hidden
                // state of the row that produced it is its feature (its own buffer
                // offset, tree order).
                last_token = accepted[n_emit - 1];
                std::memcpy(
                    last_feat.data(), v_feat + static_cast<size_t>(accepted_rows[n_emit - 1]) * row_bytes, row_bytes);
            }
            if (stop) break;
        }

        // Drain any async KV commit still in flight after the loop exits (EOS /
        // callback / max-tokens) and restride both engines for the next prefill.
        finishTurn();
        if (user_stop) has_pending_ = false;
        stats_.generated_tokens = output.size();
        return output;
    } catch (const ContextLengthExceededError&) {
        // Exhaustion is detected at a round boundary, so a seeded conversation is
        // intact and can be left as it stands.
        if (seeded) {
            finishTurn();
        } else {
            quiesce();
            resetKVCache();
        }
        throw;
    } catch (...) {
        // The caches may be mid-update: drop the conversation rather than resume
        // from a state that no longer matches its tokens.
        quiesce();
        resetKVCache();
        throw;
    }
}

}  // namespace geniex

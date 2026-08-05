// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/eagle_model.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "graph.h"
#include "llm/input_provider.h"
#include "logging.h"

namespace geniex {

namespace {

// Binds a quantized embedding provider to `embed_name`, replacing the default
// provider (which assumes an unquantized table). Registered before initialize()
// so createInputProviders() leaves it in place. An explicit table path overrides
// ModelConfig::embedding_path -- the draft needs its own embedding weights.
void registerQuantizedEmbedding(
    LLMModel& m, const std::string& embed_name, const EagleConfig& cfg, const std::string& table_path) {
    auto provider = table_path.empty() ? std::make_unique<EmbeddingInputProvider>(embed_name)
                                       : std::make_unique<EmbeddingInputProvider>(embed_name,
                                             table_path,
                                             /*row_hidden_size=*/0,
                                             /*pad_token_override=*/-1);
    provider->setQuantization(cfg.embedding_quant);
    m.addInputProvider(std::move(provider));
}

}  // namespace

EagleModel::EagleModel(LLMSpec target_spec, LLMSpec draft_spec, EagleConfig cfg) : cfg_(std::move(cfg)) {
    target_ = std::make_unique<LLMModel>(std::move(target_spec));
    draft_  = std::make_unique<LLMModel>(std::move(draft_spec));
}

LLMModel& EagleModel::target() {
    if (!target_) throw std::runtime_error("EagleModel: target engine not constructed");
    return *target_;
}

const LLMModel& EagleModel::target() const {
    if (!target_) throw std::runtime_error("EagleModel: target engine not constructed");
    return *target_;
}

LLMModel& EagleModel::draft() {
    if (!draft_) throw std::runtime_error("EagleModel: draft engine not constructed");
    return *draft_;
}

const LLMModel& EagleModel::draft() const {
    if (!draft_) throw std::runtime_error("EagleModel: draft engine not constructed");
    return *draft_;
}

bool EagleModel::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& target_cfg) {
    registerQuantizedEmbedding(target(), cfg_.target_embed_name, cfg_, /*table_path=*/"");
    if (!target().initialize(runtime_cfg, target_cfg)) {
        GENIEX_LOG_ERROR("EagleModel: target engine initialize() failed");
        return false;
    }

    ModelConfig draft_cfg = target_cfg;
    draft_cfg.model_paths = cfg_.draft_model_paths;
    registerQuantizedEmbedding(draft(), cfg_.draft_embed_name, cfg_, cfg_.draft_embedding_path);
    if (!draft().initialize(runtime_cfg, draft_cfg)) {
        GENIEX_LOG_ERROR("EagleModel: draft engine initialize() failed");
        return false;
    }

    ready_       = true;
    initialized_ = true;
    return true;
}

void EagleModel::resetKVCache() {
    target().resetKVCache();
    if (ready_) draft().resetKVCache();
}

void EagleModel::readTargetLogits(size_t phase, size_t row, std::vector<float>& out) const {
    const LLMSpec& s     = target().spec();
    const size_t   vocab = s.vocab_size;
    const size_t   lm    = target().graphIndex(phase, s.shards.size() - 1, /*cl*/ 0);
    out.resize(vocab);
    target().graph(lm).read(s.shards.back().out_state_name, out.data(), vocab, row * vocab);
}

int32_t EagleModel::argmaxTarget(size_t phase, size_t row) const {
    // Greedy accept only needs the argmax index, not the values. Reading the
    // full vocab into floats then argmaxing dequantises ~150K logits per accepted
    // row; argmaxOutput scans the raw output buffer once with no allocation and
    // no dequant (scale-offset / fp16 order matches value order). This is the
    // hot per-round CPU op on the accept path.
    const LLMSpec& s     = target().spec();
    const size_t   vocab = s.vocab_size;
    const size_t   lm    = target().graphIndex(phase, s.shards.size() - 1, /*cl*/ 0);
    return static_cast<int32_t>(target().graph(lm).argmaxOutput(s.shards.back().out_state_name, vocab, row * vocab));
}

int32_t EagleModel::argmaxTargetInPlace(size_t phase, size_t row) const { return argmaxTarget(phase, row); }

void EagleModel::readDraftLogits(size_t phase, size_t row, std::vector<float>& out) const {
    const LLMSpec& ds = draft().spec();
    const size_t   lm = draft().graphIndex(phase, ds.shards.size() - 1, /*cl*/ 0);
    const Graph&   g  = draft().graph(lm);
    // The draft's designated shard "state" output is its hidden feature
    // (last_hidden_states), not its vocabulary head. Its logits live in a
    // separate "logits" tensor whose width is the draft's (small) vocabulary --
    // reading out_state_name here would argmax over the 2560-dim hidden state
    // and produce a near-constant junk proposal set. When no explicit logits
    // tensor is configured, fall back to the shard's state output.
    const std::string& logits_name =
        cfg_.draft_logits_name.empty() ? ds.shards.back().out_state_name : cfg_.draft_logits_name;
    const auto&  os    = g.outputSpec(logits_name);
    const size_t vocab = os.shape.empty() ? ds.vocab_size : os.shape.back();
    out.resize(vocab);
    g.read(logits_name, out.data(), vocab, row * vocab);
}

int32_t EagleModel::argmaxDraft(size_t phase, size_t row) const {
    std::vector<float> logits;
    readDraftLogits(phase, row, logits);
    const size_t best = static_cast<size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    // validate() guarantees a non-empty map is exactly draft-vocab wide, so
    // `best` always indexes it; an empty map means the draft already emits ids.
    if (cfg_.draft_token_map.empty()) return static_cast<int32_t>(best);
    return cfg_.draft_token_map[best];
}

std::vector<int32_t> EagleModel::topKDraft(size_t phase, size_t row, size_t k) const {
    std::vector<float> logits;
    readDraftLogits(phase, row, logits);
    const size_t vocab = logits.size();

    k = std::min(k, vocab);
    std::vector<size_t> idx(vocab);
    std::iota(idx.begin(), idx.end(), size_t{0});
    std::partial_sort(
        idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(), [&logits](size_t a, size_t b) {
            return logits[a] > logits[b];
        });

    std::vector<int32_t> out;
    out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t raw = idx[i];
        out.push_back(cfg_.draft_token_map.empty() ? static_cast<int32_t>(raw) : cfg_.draft_token_map[raw]);
    }
    return out;
}

void EagleModel::topKDraftWithProbs(
    size_t phase, size_t row, size_t k, std::vector<int32_t>& tokens_out, std::vector<float>& probs_out) const {
    std::vector<float> logits;
    readDraftLogits(phase, row, logits);
    const size_t vocab = logits.size();

    k = std::min(k, vocab);
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

    tokens_out.clear();
    probs_out.clear();
    tokens_out.reserve(k);
    probs_out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t raw = idx[i];
        tokens_out.push_back(cfg_.draft_token_map.empty() ? static_cast<int32_t>(raw) : cfg_.draft_token_map[raw]);
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
EagleModel::DraftTree EagleModel::buildDraftTree(LLMModel& drf, int32_t anchor_token, const uint8_t* anchor_feature,
    size_t row_bytes, float theta, size_t max_nodes) {
    DraftTree      tree;
    const LLMSpec& ds = drf.spec();
    // The draft decode graph processes at most seq_len_decode rows per forward,
    // so each level's frontier is capped to that width. n_branches is likewise
    // capped: a level can never keep more survivors than one draft forward can
    // re-expand next level.
    const size_t draft_w    = std::max<size_t>(1, ds.seq_len_decode);
    const size_t n_branches = std::min(std::max<size_t>(1, cfg_.n_branches), draft_w);
    const size_t depth_max  = std::max<size_t>(1, cfg_.draft_len);
    const size_t d_body     = drf.graphIndex(1, ds.shards.size() >= 2 ? ds.shards.size() - 2 : 0, 0);
    const size_t base_past  = drf.nPast();

    if (max_nodes == 0) return tree;  // target has no verify width to spare

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
            row_bytes,
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

    LLMModel&      tgt   = target();
    LLMModel&      drf   = draft();
    const LLMSpec& ts    = tgt.spec();
    const float    theta = cfg_.rope_theta;

    stats_ = EagleStats{};

    resetKVCache();

    // Prefill the target, capturing every position's hidden-state feature: the
    // prefill buffer keeps only the final chunk, but the draft seed needs all rows.
    const size_t body_pf   = tgt.graphIndex(/*phase=*/0, ts.shards.size() >= 2 ? ts.shards.size() - 2 : 0, /*cl*/ 0);
    const auto&  feat_spec = tgt.outputTensorSpec(body_pf, cfg_.target_feature_output);
    const size_t hidden    = feat_spec.shape.back();
    const size_t row_bytes = hidden * feat_spec.elementSize();

    std::vector<uint8_t> prompt_feats;
    tgt.prefill(prompt_tokens,
        theta,
        /*feature_rows=*/nullptr,
        /*feature_row_bytes=*/0,
        /*feature_name=*/"",
        &prompt_feats,
        cfg_.target_feature_output);

    // The last prompt logits row is the final prefill chunk's last valid row.
    const size_t last_chunk_rows =
        prompt_tokens.size() - ((prompt_tokens.size() - 1) / ts.seq_len_prefill) * ts.seq_len_prefill;
    int32_t first_token = argmaxTarget(/*phase=*/0, /*row=*/last_chunk_rows - 1);
    for (int32_t eos : ts.eos_token_ids)
        if (first_token == eos) return {};

    std::vector<int32_t> output;
    output.push_back(first_token);
    if (token_callback && !token_callback(first_token)) return output;

    // Seed the draft over the prompt. EAGLE's draft predicts token[i+1] from
    // embed(token[i+1]) paired with the target hidden state that PRODUCED
    // token[i+1] -- i.e. target_feat[i]. This is realized by feeding the draft
    // the embeddings shifted left by one (drop token[0]) against the unshifted
    // feature buffer, processing exactly n-1 positions so the draft KV ends at
    // position n-1. The prompt context reaches the draft entirely through these
    // features; feeding an unshifted 1:1 stream conditions it on the wrong hidden
    // state and it proposes near-constant garbage.
    tgt.switchToDecodeStride();

    drf.resetKVCache();
    if (prompt_tokens.size() > 1) {
        const std::vector<int32_t> draft_prompt(prompt_tokens.begin() + 1, prompt_tokens.end());
        drf.prefill(draft_prompt, theta, prompt_feats.data(), row_bytes, cfg_.draft_feature_input);
    }
    drf.switchToDecodeStride();

    // Keep both clusters from down-clocking across the speculative decode window;
    // the target verify and draft forwards run back-to-back here just like the
    // vanilla decode loop, which brackets itself the same way.
    tgt.startClockKeeper();
    drf.startClockKeeper();

    // Feature conditioning the draft's next step: the target hidden state that
    // produced last_token. For the first round that is the last prompt position.
    std::vector<uint8_t> last_feat(row_bytes);
    std::memcpy(last_feat.data(), prompt_feats.data() + (prompt_tokens.size() - 1) * row_bytes, row_bytes);

    int32_t last_token = first_token;

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
        if (tgt.nPast() + n_verify > ts.context_lengths[tgt.activeContextLengthIndex()]) break;
        // Order the previous round's async KV commit before this verify reads the
        // target KV inputs. On the first round this is a no-op.
        tgt.drainDecodePool();
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
            const int32_t tgt_next = argmaxTargetInPlace(/*phase=*/1, /*row=*/cur_row);
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

        // Emit accepted tokens, stopping on EOS / callback / max-tokens.
        bool stop = false;
        {
            for (size_t i = 0; i < n_accept; ++i) {
                const int32_t tok = accepted[i];
                for (int32_t e : ts.eos_token_ids)
                    if (tok == e) stop = true;
                if (stop) break;
                output.push_back(tok);
                if (token_callback && !token_callback(tok)) {
                    stop = true;
                    break;
                }
                if (static_cast<int>(output.size()) >= gen_cfg.max_tokens) {
                    stop = true;
                    break;
                }
            }
        }
        if (stop) break;

        // 4b. Advance the draft KV in place: feed the accepted path
        //     [anchor, accepted proposals...] through the draft decode graph
        //     seeded with each token's target feature, committing every row. No
        //     reset, no wide re-prefill -- the draft KV tracks the committed
        //     sequence exactly, as a persistent draft-kv-cache does. The replay is
        //     chunked to the draft decode width so each forward is legal.
        {
            // EAGLE draft-KV advance with the embedding LEFT-SHIFT that the draft
            // was trained on: draft
            // position for token a_i is fed embed(a_{i+1}) paired with the target
            // hidden state feat(a_i). Committing embed(a_i)+feat(a_i) instead (no
            // shift) builds a KV history the draft never saw in training, so the
            // next anchor proposal is mis-conditioned and rarely matches the
            // target. The full committed sequence this round is
            //   a0=last_token, accepted[0], accepted[1], ..., accepted[n_accept-1]
            // and we advance the draft KV by n_accept positions so it is ready to
            // propose after accepted.back():
            //   slot j (j=0..n_accept-1): embed(accepted[j]) + feat(a_{j-1})
            //   feat(a_{-1}) == last_feat; feat(accepted[j-1]) == v_feat row that
            //   produced accepted[j-1], i.e. buffer offset accepted_rows[j-1].
            std::vector<int32_t> replay;
            std::vector<uint8_t> seeds;
            {
                replay.assign(accepted.begin(), accepted.end());
                seeds.resize(n_accept * row_bytes);
                std::memcpy(seeds.data(), last_feat.data(), row_bytes);
                for (size_t j = 1; j < n_accept; ++j)
                    std::memcpy(seeds.data() + j * row_bytes,
                        v_feat + static_cast<size_t>(accepted_rows[j - 1]) * row_bytes,
                        row_bytes);
            }

            const size_t draft_w = std::max<size_t>(1, drf.spec().seq_len_decode);
            for (size_t off = 0; off < n_accept; off += draft_w) {
                const size_t         chunk = std::min(draft_w, n_accept - off);
                std::vector<int32_t> ctok(replay.begin() + static_cast<std::ptrdiff_t>(off),
                    replay.begin() + static_cast<std::ptrdiff_t>(off + chunk));
                std::vector<int32_t> dpos(chunk);
                for (size_t i = 0; i < chunk; ++i) dpos[i] = static_cast<int32_t>(drf.nPast() + i);
                drf.decodeBatch(ctok,
                    dpos,
                    /*attention_map=*/{},
                    drf.nPast(),
                    theta,
                    seeds.data() + off * row_bytes,
                    row_bytes,
                    cfg_.draft_feature_input);
                drf.commitDecodeRows(std::vector<bool>(chunk, true), chunk);
            }
        }

        last_token = accepted.back();
        // The next round's anchor feature is the target hidden state of the last
        // accepted verify row (its own buffer offset, tree order).
        std::memcpy(last_feat.data(), v_feat + static_cast<size_t>(accepted_rows[n_accept - 1]) * row_bytes, row_bytes);
    }

    // Drain any async KV commit still in flight after the loop exits (EOS /
    // callback / max-tokens) before restriding reads the target KV buffers.
    tgt.drainDecodePool();
    tgt.stopClockKeeper();
    drf.stopClockKeeper();
    tgt.switchToPrefillStride();
    stats_.generated_tokens = output.size();
    return output;
}

}  // namespace geniex

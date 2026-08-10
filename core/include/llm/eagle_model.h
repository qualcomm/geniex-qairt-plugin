// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "geniex_export.h"
#include "llm/eagle_types.h"
#include "llm/speculative_llm_model.h"

namespace geniex {

// EAGLE speculative decoder composing a target and a draft LLMModel as peers.
//
// Each round the draft autoregressively proposes a short token chain conditioned
// on the target's last-layer hidden states, the target verifies the whole chain
// in one forward, and the longest greedily-matching prefix (plus one bonus
// target token) is accepted. Under greedy sampling the emitted sequence equals
// plain target decoding, so speculation only affects throughput.
class GENIEX_API EagleModel : public Model {
   public:
    EagleModel(LLMSpec target_spec, LLMSpec draft_spec, EagleConfig cfg);

    // Loads both engines. The target initializes first because the draft is
    // seeded from the target's hidden-state features; target_cfg::model_paths is
    // the target's bins, the draft's come from EagleConfig::draft_model_paths.
    bool initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& target_cfg);

    // Returns generated token IDs (excluding the prompt). token_callback is
    // called with each accepted token; return false to stop early.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg = {},
        std::function<bool(int32_t)> token_callback = nullptr);

    // Speculative-decoding metrics from the most recent generate() call.
    const EagleStats& lastStats() const { return stats_; }

    void resetKVCache();

   protected:
    // The two engines and readiness flag. Exposed so a test subclass can inject
    // pre-initialized engines and skip QNN bring-up, mirroring how the LLMModel
    // tests wire in CPU graph fixtures; production reaches ready via initialize().
    std::unique_ptr<SpeculativeLLMModel> target_;
    std::unique_ptr<SpeculativeLLMModel> draft_;
    bool                                 ready_ = false;

   private:
    SpeculativeLLMModel&       target();
    const SpeculativeLLMModel& target() const;
    SpeculativeLLMModel&       draft();
    const SpeculativeLLMModel& draft() const;

    // Greedy argmax over the target LM-head logits row `row` in the given phase.
    int32_t argmaxTarget(size_t phase, size_t row) const;
    void    readTargetLogits(size_t phase, size_t row, std::vector<float>& out) const;

    // Draft argmax at `row`, mapped through draft_token_map to a full-vocab id
    // (identity when the map is empty).
    int32_t argmaxDraft(size_t phase, size_t row) const;
    void    readDraftLogits(size_t phase, size_t row, std::vector<float>& out) const;

    // Top-k draft tokens at logits `row`, mapped through draft_token_map, most
    // probable first. Reads the vocab row once (no full-vocab copy per element).
    std::vector<int32_t> topKDraft(size_t phase, size_t row, size_t k) const;

    // Top-k draft tokens with their softmax probabilities over the full draft
    // vocab, most probable first. Tree pruning ranks branches by cumulative
    // probability, so the child sampler must return probs, not just ids. Softmax
    // is taken over the whole row for calibrated probabilities.
    void topKDraftWithProbs(
        size_t phase, size_t row, size_t k, std::vector<int32_t>& tokens_out, std::vector<float>& probs_out) const;

    // Argmax over the target LM-head row without materializing a full-vocab
    // std::vector; scans the graph buffer in place.
    int32_t argmaxTargetInPlace(size_t phase, size_t row) const;

    // A speculative token tree proposed by the draft and verified by the target
    // in one batched forward. Node 0 is the anchor (last committed token); nodes
    // 1..N are draft proposals. parent[i] indexes an earlier node (parent[0] =
    // -1). depth[i] is the RoPE position offset from the anchor. Row i of the
    // target verify pass predicts the continuation of node i.
    struct DraftTree {
        std::vector<int32_t> tokens;    // node token ids (full vocab)
        std::vector<int32_t> parent;    // parent node index (-1 for the anchor)
        std::vector<int32_t> depth;     // tree depth (position offset from anchor)
        std::vector<uint8_t> features;  // per-node draft feature seed, row-major
    };

    // Grows a speculative tree from the anchor (last committed token + its
    // target feature) using the draft engine's decode graph with tree attention.
    // Advances no KV (fully speculative). `anchor_feature` seeds the draft's
    // prediction of the anchor's continuation. Bounded by draft_len (depth),
    // n_branches (fan-out) and the target's verify width.
    DraftTree buildDraftTree(SpeculativeLLMModel& drf, int32_t anchor_token, const uint8_t* anchor_feature,
        size_t row_bytes, float theta, size_t max_nodes);

    // Keeps the max_nodes highest-cumulative-probability nodes of `in` and
    // reindexes parent/depth. cumProb is
    // non-increasing down the tree, so the kept set is parent-closed.
    static DraftTree pruneTreeByCumProb(
        const DraftTree& in, const std::vector<float>& cum_prob, size_t max_nodes, size_t row_bytes);

    EagleConfig cfg_;
    EagleStats  stats_;
};

}  // namespace geniex

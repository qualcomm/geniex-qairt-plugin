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
#include "llm/sampler_utils.h"
#include "llm/speculative_llm_model.h"
#include "llm/token_generator.h"

namespace geniex {

// EAGLE speculative decoder composing a target and a draft LLMModel as peers.
//
// Each round the draft autoregressively proposes a short token chain conditioned
// on the target's last-layer hidden states, the target verifies the whole chain
// in one forward, and the longest prefix matching the target's own picks (plus
// one bonus target token) is accepted. The target's pick at each verified row is
// its argmax, or a draw from the GenerationConfig sampler chain when
// enable_sampling is set. A proposal is kept only when it equals that pick, so
// the emitted sequence follows the target's distribution exactly and
// speculation only affects throughput. The draft proposes greedily either way.
//
// Known gaps: saving or loading the KV cache and forwardLogits() are not
// supported.
class GENIEX_API EagleModel : public Model, public TokenGenerator {
   public:
    EagleModel(LLMSpec target_spec, LLMSpec draft_spec, EagleConfig cfg);

    // Loads both engines. The target initializes first because the draft is
    // seeded from the target's hidden-state features; target_cfg::model_paths is
    // the target's bins, the draft's come from EagleConfig::draft_model_paths.
    bool initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& target_cfg) override;

    // Returns generated token IDs (excluding the prompt). token_callback is
    // called with each accepted token; return false to stop early.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg = {},
        std::function<bool(int32_t)> token_callback = nullptr) override;

    // Speculative-decoding metrics from the most recent generate() call.
    const EagleStats& lastStats() const { return stats_; }

    void resetKVCache() override;

    // Saving or loading the KV cache and forwardLogits() throw
    // UnsupportedOperationError: not supported yet.
    void                            saveKVCacheToFile(const std::string& path) const override;
    void                            loadKVCacheFromFile(const std::string& path) override;
    [[noreturn]] std::vector<float> forwardLogits(
        const std::vector<int32_t>& tokens, bool all_positions = false) override;

    // Committed length of the conversation: the target engine's KV cache plus the
    // last emitted token, which is only committed by the next generate().
    size_t nPast() const override;

    // Vocabulary size of the target engine; 0 before initialize().
    size_t vocabSize() const override;

    // Resolves EagleConfig's graph tensor bindings (embedding entries, feature
    // and logits tensor names) from the loaded engines' inferred specs and
    // graphs, so no export-specific names are hard-coded. Called by initialize()
    // once both engines are up; exposed for testing against graph fixtures.
    // Throws if any derived name is absent from the graphs.
    static void inferTensorBindings(SpeculativeLLMModel& target, SpeculativeLLMModel& draft, EagleConfig& cfg);

   protected:
    // The two engines and readiness flag. Exposed so a test subclass can inject
    // pre-initialized engines and skip QNN bring-up, mirroring how the LLMModel
    // tests wire in CPU graph fixtures; production reaches ready via initialize().
    std::unique_ptr<SpeculativeLLMModel> target_;
    std::unique_ptr<SpeculativeLLMModel> draft_;
    bool                                 ready_ = false;

    // A speculative token tree proposed by the draft and verified by the target
    // in one batched forward. Node 0 is the anchor (last committed token); nodes
    // 1..N are draft proposals. parent[i] indexes an earlier node (parent[0] =
    // -1). depth[i] is the RoPE position offset from the anchor. Row i of the
    // target verify pass predicts the continuation of node i.
    // Exposed (with pruneTreeByCumProb) so a test can pin the parent-closure and
    // tie-break invariants the prune step relies on, in isolation from generate().
    struct DraftTree {
        std::vector<int32_t> tokens;    // node token ids (full vocab)
        std::vector<int32_t> parent;    // parent node index (-1 for the anchor)
        std::vector<int32_t> depth;     // tree depth (position offset from anchor)
        std::vector<uint8_t> features;  // per-node draft feature seed, row-major
    };

    // Keeps the max_nodes highest-cumulative-probability nodes of `in` and
    // reindexes parent/depth. cumProb is
    // non-increasing down the tree, so the kept set is parent-closed.
    static DraftTree pruneTreeByCumProb(
        const DraftTree& in, const std::vector<float>& cum_prob, size_t max_nodes, size_t row_bytes);

   private:
    SpeculativeLLMModel&       target();
    const SpeculativeLLMModel& target() const;
    SpeculativeLLMModel&       draft();
    const SpeculativeLLMModel& draft() const;

    // Greedy argmax over the target LM-head logits row `row` in the given phase.
    int32_t argmaxTarget(size_t phase, size_t row) const;

    // The target's pick for logits row `row`: argmax, or a sampler draw (which also
    // advances the sampler's penalty / grammar history) when sampling is enabled.
    int32_t sampleTarget(size_t phase, size_t row);

    // (Re)builds the cached sampler for `gen_cfg`; drops it when sampling is off.
    // Keeps the existing one, and its history, while the config is unchanged.
    void prepareSampler(const GenerationConfig& gen_cfg, const std::vector<int32_t>& prompt_tokens);

    void readDraftLogits(size_t phase, size_t row, std::vector<float>& out) const;

    // Top-k draft tokens with their softmax probabilities over the full draft
    // vocab, most probable first. Tree pruning ranks branches by cumulative
    // probability, so the child sampler must return probs, not just ids. Softmax
    // is taken over the whole row for calibrated probabilities.
    void topKDraftWithProbs(
        size_t phase, size_t row, size_t k, std::vector<int32_t>& tokens_out, std::vector<float>& probs_out) const;

    // Grows a speculative tree from the anchor (last committed token + its
    // target feature) using the draft engine's decode graph with tree attention.
    // Commits each level's draft KV so deeper levels can attend to it, then
    // rewinds the whole speculative region before returning (net KV advance = 0).
    // `anchor_feature` seeds the draft's prediction of the anchor's continuation.
    // Depth is bounded by min(draft_len, max_nodes) and fan-out by n_branches;
    // the finished tree is pruned to max_nodes (the target's verify width). Bails
    // early (empty tree) if the speculative rows would not fit the draft KV.
    DraftTree buildDraftTree(SpeculativeLLMModel& drf, int32_t anchor_token, const uint8_t* anchor_feature,
        size_t row_bytes, float theta, size_t max_nodes);

    EagleConfig cfg_;
    EagleStats  stats_;

    // Null when sampling is disabled. Persists across turns so penalty history
    // spans the conversation; dropped by resetKVCache().
    std::unique_ptr<Sampler> sampler_;
    GenerationConfig         sampler_cfg_;
    bool                     sampler_cfg_valid_ = false;

    // Conversation state carried between generate() calls. The target KV holds
    // every token up to the last emitted one, which stays pending until the next
    // turn commits it; the draft KV trails the target by one row. carry_feat_ is
    // the target hidden state that produced the pending token (empty before the
    // first turn).
    int32_t              pending_token_ = 0;
    bool                 has_pending_   = false;
    std::vector<uint8_t> carry_feat_;
};

}  // namespace geniex

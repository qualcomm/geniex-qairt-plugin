// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/eagle_model.cpp - the EAGLE speculative decoder's
// two-engine generate() loop. Both the target and the draft are real LLMModels
// backed by CPU graph fixtures and the link-time QnnApi stub; no QNN device
// bring-up. A test-only subclass injects the fixture engines, skipping the
// device-only initialize() path.
//
// The stub writes a one-hot logits peak at stubSetNextToken() on every row, so
// the target's verification argmax is deterministic. The draft's proposal is
// that same peak remapped through EagleConfig::draft_token_map, which lets a
// test force the draft to either agree with the target (full acceptance) or
// diverge (rejection at the first row) using one global stub.

#include "llm/eagle_model.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "llm/eagle_types.h"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"
#include "testing/testable_llm_model.hpp"

namespace {

using geniex::EagleConfig;
using geniex::testing::EagleDraftFixture;
using geniex::testing::EagleTargetFixture;
using geniex::testing::EagleWideDraftFixture;
using geniex::testing::LLMFixture;
using geniex::testing::MultiCLEagleDraftFixture;
using geniex::testing::MultiCLEagleTargetFixture;
using geniex::testing::NoDecodePoolEnv;
using geniex::testing::TestableLLMModel;
using geniex::testing::TestableSpeculativeLLMModel;

// EagleModel with an injection seam: builds fixture-backed target/draft engines
// and marks itself ready, bypassing the QNN-only initialize().
class TestableEagleModel : public geniex::EagleModel {
   public:
    explicit TestableEagleModel(EagleConfig cfg)
        : geniex::EagleModel(EagleTargetFixture::makeSpec(), EagleDraftFixture::makeSpec(), std::move(cfg)) {}

    // Explicit-spec ctor: lets a test pin the base-class draft spec to a wider
    // decode fixture so buildDraftTree's batched frontier / prune path is reached.
    TestableEagleModel(geniex::LLMSpec target_spec, geniex::LLMSpec draft_spec, EagleConfig cfg)
        : geniex::EagleModel(std::move(target_spec), std::move(draft_spec), std::move(cfg)) {}

    template <typename TargetFixture, typename DraftFixture>
    bool init(TargetFixture& tfx, DraftFixture& dfx, const std::vector<int32_t>& target_eos = {}) {
        auto t = std::make_unique<TestableSpeculativeLLMModel>(TargetFixture::makeSpec());
        if (!t->initFromFixture(tfx)) return false;
        t->spec_.eos_token_ids = target_eos;

        auto d = std::make_unique<TestableSpeculativeLLMModel>(DraftFixture::makeSpec());
        if (!d->initFromFixture(dfx)) return false;

        target_      = std::move(t);
        draft_       = std::move(d);
        ready_       = true;
        initialized_ = true;
        return true;
    }

    // Forward to the protected tree-prune primitive so its parent-closure and
    // tie-break invariants can be pinned without driving a full generate().
    using DraftTree = geniex::EagleModel::DraftTree;
    static DraftTree prune(
        const DraftTree& in, const std::vector<float>& cum_prob, size_t max_nodes, size_t row_bytes) {
        return geniex::EagleModel::pruneTreeByCumProb(in, cum_prob, max_nodes, row_bytes);
    }

    // Expose the engines' active CL so a multi-CL test can assert that
    // speculation promoted past the initial context length.
    size_t targetActiveCL() const { return target_->activeContextLengthIndex(); }
    size_t draftActiveCL() const { return draft_->activeContextLengthIndex(); }
};

// draft_len = 3 makes the batched verify width 4, matching EagleTargetFixture's
// ar=4 decode graph. The binding names match the Eagle fixtures: both engines'
// first-shard input is `input_embeds`, bodies emit `last_hidden_states`, and the
// draft body takes an extra `hidden_states` feature input.
EagleConfig makeConfig(std::vector<int32_t> draft_token_map) {
    EagleConfig cfg;
    cfg.draft_len             = 3;
    cfg.draft_token_map       = std::move(draft_token_map);
    cfg.rope_theta            = 10000.0f;
    cfg.target_embed_name     = "input_embeds";
    cfg.draft_embed_name      = "input_embeds";
    cfg.target_feature_output = "last_hidden_states";
    cfg.draft_feature_input   = "hidden_states";
    cfg.draft_feature_output  = "last_hidden_states";
    cfg.draft_logits_name     = "logits";
    return cfg;
}

geniex::GenerationConfig genConfig(int32_t max_tokens) {
    geniex::GenerationConfig cfg;
    cfg.enable_sampling = false;
    cfg.max_tokens      = max_tokens;
    return cfg;
}

struct StubLogits {
    StubLogits(int32_t token, uint32_t vocab) {
        geniex::testing::stubSetVocabSize(vocab);
        geniex::testing::stubSetNextToken(token);
    }
    ~StubLogits() {
        geniex::testing::stubSetNextToken(-1);
        geniex::testing::stubSetVocabSize(0);
    }
};

// Drives the stub in position-scripted mode: the argmax at absolute sequence
// position p is script[p], so the emitted token varies with position instead of
// being one fixed id. Both a plain AR run and a speculative run over the same
// script must therefore walk the same per-position chain.
struct StubScript {
    StubScript(std::vector<int32_t> script, uint32_t vocab) {
        geniex::testing::stubSetVocabSize(vocab);
        geniex::testing::stubSetTokenScript(std::move(script));
    }
    ~StubScript() {
        geniex::testing::stubSetTokenScript({});
        geniex::testing::stubSetVocabSize(0);
    }
};

// generate() before initialize() has run must fail loudly rather than
// dereference the un-loaded engines.
TEST(EagleModel, GenerateBeforeInitializeThrows) {
    NoDecodePoolEnv    no_pool;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    EXPECT_THROW(model.generate({1, 2, 3}, genConfig(4)), std::runtime_error);
}

// The graph tensor bindings the two-engine driver relies on are inferred from
// the loaded graphs rather than hard-coded: given fixture engines, the resolved
// names must match what the fixtures actually expose. This is the CPU-reachable
// stand-in for the export-specific inference initialize() does on device.
TEST(EagleModel, InfersTensorBindingsFromGraphs) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;

    TestableSpeculativeLLMModel target{EagleTargetFixture::makeSpec()};
    TestableSpeculativeLLMModel draft{EagleDraftFixture::makeSpec()};
    ASSERT_TRUE(target.initFromFixture(tfx));
    ASSERT_TRUE(draft.initFromFixture(dfx));

    EagleConfig cfg;
    geniex::EagleModel::inferTensorBindings(target, draft, cfg);

    EXPECT_EQ(cfg.target_embed_name, "input_embeds");
    EXPECT_EQ(cfg.draft_embed_name, "input_embeds");
    EXPECT_EQ(cfg.target_feature_output, "last_hidden_states");
    EXPECT_EQ(cfg.draft_feature_output, "last_hidden_states");
    EXPECT_EQ(cfg.draft_feature_input, "hidden_states");
    EXPECT_EQ(cfg.draft_logits_name, "logits");
}

TEST(EagleModel, EmptyPromptReturnsEmpty) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    ASSERT_TRUE(model.init(tfx, dfx));
    EXPECT_TRUE(model.generate({}, genConfig(4)).empty());
}

// Empty draft_token_map: the draft proposal is the raw peak, so it always
// equals the target's verification token and the whole chain is accepted every
// round. Exercises the accept-all path with identity draft-token mapping.
TEST(EagleModel, AcceptsFullChain) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    ASSERT_TRUE(model.init(tfx, dfx));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    auto       out = model.generate({1, 2, 3}, genConfig(/*max_tokens=*/8));

    ASSERT_FALSE(out.empty());
    EXPECT_LE(out.size(), 8u);
    for (int32_t tok : out) EXPECT_EQ(tok, 5);

    // Every proposal is accepted, so each round emits more than one token:
    // mean accepted tokens/round must exceed 1.0 and match generated / iterations.
    const auto& stats = model.lastStats();
    EXPECT_EQ(stats.generated_tokens, out.size());
    EXPECT_GT(stats.iterations, 0u);
    EXPECT_GT(stats.meanAcceptedTokensPerRound(), 1.0f);
    EXPECT_FLOAT_EQ(stats.meanAcceptedTokensPerRound(),
        static_cast<float>(stats.generated_tokens) / static_cast<float>(stats.iterations));
}

// A draft_token_map that remaps the peak token to a different id makes every
// draft proposal disagree with the target, so only the bonus token is accepted
// each round. Exercises the rejection branch and the draft re-alignment.
TEST(EagleModel, RejectsMismatchedProposals) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;

    std::vector<int32_t> map(EagleTargetFixture::kVocab);
    for (int32_t i = 0; i < static_cast<int32_t>(map.size()); ++i) map[i] = i;
    map[5] = 3;  // draft proposes 3 where the target verifies 5

    TestableEagleModel model{makeConfig(std::move(map))};
    ASSERT_TRUE(model.init(tfx, dfx));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    auto       out = model.generate({1, 2, 3}, genConfig(/*max_tokens=*/5));

    ASSERT_EQ(out.size(), 5u);  // one accepted token per round
    for (int32_t tok : out) EXPECT_EQ(tok, 5);

    // The bonus token from each round plus the pre-loop first token: emitting
    // N tokens takes N-1 speculation rounds, so mean accepted tokens/round sits
    // just above 1.0 (no draft proposals matched, only the bonus token lands).
    const auto& stats = model.lastStats();
    EXPECT_EQ(stats.generated_tokens, out.size());
    EXPECT_EQ(stats.iterations, out.size() - 1);
    EXPECT_FLOAT_EQ(stats.meanAcceptedTokensPerRound(),
        static_cast<float>(stats.generated_tokens) / static_cast<float>(stats.iterations));
}

// When the very first target token is an EOS id, generate() returns nothing.
TEST(EagleModel, StopsOnEosFirstToken) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    ASSERT_TRUE(model.init(tfx, dfx, /*target_eos=*/{5}));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    EXPECT_TRUE(model.generate({1, 2, 3}, genConfig(8)).empty());
}

// A token_callback returning false halts generation; the returned sequence is
// exactly the tokens emitted up to and including the stopping one.
TEST(EagleModel, StopsWhenCallbackReturnsFalse) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    ASSERT_TRUE(model.init(tfx, dfx));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    int        seen = 0;
    auto       out  = model.generate({1, 2, 3}, genConfig(/*max_tokens=*/8), [&](int32_t) {
        ++seen;
        return seen < 2;  // stop after the second token
    });
    EXPECT_EQ(out.size(), 2u);
}

// Running past the target's context window is genuine exhaustion (the decode
// stride is fixed and never promotes mid-loop), so generate() must fail loudly
// rather than silently returning a short result. Either the KV-write bounds
// guard or the verify-batch context guard fires first depending on which limit
// (KV capacity vs. full context length) is hit -- both derive from
// std::runtime_error, and the contract under test is "throw, don't truncate".
TEST(EagleModel, ThrowsWhenVerifyBatchExceedsContext) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    ASSERT_TRUE(model.init(tfx, dfx));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    EXPECT_THROW(model.generate({1, 2, 3}, genConfig(/*max_tokens=*/1000)), std::runtime_error);
}

// A draft_token_map shorter than the draft logits tensor is the normal case for
// exports whose LM-head width is padded above the real draft vocabulary (e.g. a
// 2560-slot tensor for a 2255-token draft vocab). The map size is authoritative:
// readDraftLogits clamps the ranked window to it so padded slots can never be
// proposed. Here the stub peaks at id 5 but the 5-entry map (indices 0..4) hides
// it, so generation proceeds over the visible window instead of throwing.
// Guarding a genuinely-too-short map (shorter than the real draft vocab) is the
// bundle loader's job (qwen3_eaglet validates the map against draft-n-vocab).
TEST(EagleModel, ClampsDraftVocabToTokenMapSize) {
    NoDecodePoolEnv    no_pool;
    EagleTargetFixture tfx;
    EagleDraftFixture  dfx;

    std::vector<int32_t> short_map(5);
    for (int32_t i = 0; i < 5; ++i) short_map[i] = i;
    TestableEagleModel model{makeConfig(std::move(short_map))};
    ASSERT_TRUE(model.init(tfx, dfx));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    EXPECT_NO_THROW(model.generate({1, 2, 3}, genConfig(/*max_tokens=*/4)));
}

// pruneTreeByCumProb keeps the max_nodes highest-cumulative-probability nodes,
// breaks ties by original index, and reindexes parent pointers so the survivors
// stay in level order. cum_prob is non-increasing down the tree, so the kept set
// is parent-closed (a kept child's parent always ranks at least as high). Pinned
// directly here because the ar=1 fixture never grows a tree large enough to prune.
TEST(EagleModel, PruneTreeByCumProbKeepsParentClosedTopK) {
    // Tree: 0 is the anchor child; 1,2 are its children; 3 is a child of 1.
    //   0
    //   |- 1 -- 3
    //   |- 2
    constexpr size_t              kRowBytes = 2;  // 2 feature bytes per node
    TestableEagleModel::DraftTree in;
    in.tokens   = {10, 11, 12, 13};
    in.parent   = {-1, 0, 0, 1};
    in.depth    = {0, 1, 1, 2};
    in.features = {0, 0, 1, 1, 2, 2, 3, 3};  // row i = {i, i}

    // Node 1 and node 2 tie at 0.5; the tie-break keeps the lower index (1).
    // Node 3 (0.1) loses to both, so it is pruned along with its subtree.
    std::vector<float> cum_prob = {0.9f, 0.5f, 0.5f, 0.1f};

    auto out = TestableEagleModel::prune(in, cum_prob, /*max_nodes=*/2, kRowBytes);

    ASSERT_EQ(out.tokens.size(), 2u);
    EXPECT_EQ(out.tokens[0], 10);  // anchor child, always kept (highest prob)
    EXPECT_EQ(out.tokens[1], 11);  // node 1 wins the tie over node 2
    EXPECT_EQ(out.parent[0], -1);  // anchor child keeps its -1 parent
    EXPECT_EQ(out.parent[1], 0);   // node 1's parent (old 0) remaps to new index 0
    EXPECT_EQ(out.depth[0], 0);
    EXPECT_EQ(out.depth[1], 1);

    // Features travel with their node, reindexed in kept-order.
    ASSERT_EQ(out.features.size(), 2u * kRowBytes);
    EXPECT_EQ(out.features[0], 0);  // node 0's feature row {0,0}
    EXPECT_EQ(out.features[1], 0);
    EXPECT_EQ(out.features[2], 1);  // node 1's feature row {1,1}
    EXPECT_EQ(out.features[3], 1);
}

// A batched (ar=2) draft decode graph lets buildDraftTree grow a frontier wider
// than one row, so n_branches > 1 survives the draft_w clamp and the level-
// synchronous expansion / global candidate pool / cumulative-prob prune path
// actually runs. Every proposal still matches the target (identity draft map),
// so the round accepts more than the lone bonus token -- the payoff the tree
// search exists to deliver.
TEST(EagleModel, WideBranchTreeAcceptsMultipleTokens) {
    NoDecodePoolEnv       no_pool;
    EagleTargetFixture    tfx;
    EagleWideDraftFixture dfx;

    EagleConfig cfg = makeConfig(/*draft_token_map=*/{});
    cfg.n_branches  = 2;  // fan out; only reachable with the ar=2 draft width
    TestableEagleModel model{EagleTargetFixture::makeSpec(), EagleWideDraftFixture::makeSpec(), cfg};
    ASSERT_TRUE((model.init<EagleTargetFixture, EagleWideDraftFixture>(tfx, dfx)));

    StubLogits stub(/*token=*/5, EagleTargetFixture::kVocab);
    auto       out = model.generate({1, 2, 3}, genConfig(/*max_tokens=*/6));

    ASSERT_FALSE(out.empty());
    for (int32_t tok : out) EXPECT_EQ(tok, 5);

    const auto& stats = model.lastStats();
    EXPECT_EQ(stats.generated_tokens, out.size());
    EXPECT_GT(stats.iterations, 0u);
    EXPECT_GT(stats.meanAcceptedTokensPerRound(), 1.0f);
}

// The PR's central guarantee: under greedy sampling, EAGLE speculation emits the
// exact sequence a plain autoregressive target would. The other tests use a
// single fixed peak, so target and draft trivially agree at every position and a
// tree-mask / mis-seeded-feature / KV-desync bug would still "pass" (every row
// peaks at the same token regardless of position). This drives both a plain
// LLMModel and the EagleModel with a stub whose argmax VARIES with absolute
// position, so a divergence at any position -- exactly what those bugs cause --
// makes the two sequences differ. n_branches stays 1 (linear chain), so a verify
// row's tree depth equals its batch offset, which is what the scripted stub's
// n_past + row_offset position recovery assumes.
TEST(EagleModel, MatchesPlainAutoregressiveUnderGreedy) {
    NoDecodePoolEnv no_pool;
    ASSERT_EQ(EagleTargetFixture::kVocab, LLMFixture::kVocab);

    // A non-constant, EOS-free script covering every decoded position; the
    // largest prompt+generation window here stays well under kVocab entries.
    const std::vector<int32_t> script = {1, 2, 3, 4, 6, 7, 1, 3, 2, 6, 4, 7, 1, 2, 3, 6};
    constexpr uint32_t         kVocab = EagleTargetFixture::kVocab;
    const std::vector<int32_t> prompt = {1, 2, 3};
    const int32_t              kMax   = 6;

    std::vector<int32_t> plain_out;
    {
        StubScript       stub(script, kVocab);
        LLMFixture       fx;
        TestableLLMModel plain{LLMFixture::makeSpec()};
        ASSERT_TRUE(plain.initFromFixture(fx));
        plain_out = plain.generate(prompt, genConfig(kMax));
    }

    std::vector<int32_t> eagle_out;
    {
        StubScript         stub(script, kVocab);
        EagleTargetFixture tfx;
        EagleDraftFixture  dfx;
        TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
        ASSERT_TRUE(model.init(tfx, dfx));
        eagle_out = model.generate(prompt, genConfig(kMax));
    }

    ASSERT_FALSE(plain_out.empty());
    EXPECT_EQ(eagle_out, plain_out);
}

// Multi-CL promotion: a two-CL bundle ([cl8, cl16]) must produce the exact same
// greedy sequence as the single-CL ([cl16]) bundle of identical geometry, while
// growing into the larger CL mid-loop instead of throwing
// ContextLengthExceededError. Both engines promote per round (verify batch for
// the target, accepted-path replay for the draft). Comparing against the
// single-CL Eagle run (rather than a plain AR model) holds the prefill/decode
// widths fixed, so any divergence is attributable to the CL upgrade alone.
TEST(EagleModel, PromotesAcrossContextLengthBoundary) {
    NoDecodePoolEnv no_pool;
    ASSERT_EQ(MultiCLEagleTargetFixture::kVocab, EagleTargetFixture::kVocab);
    ASSERT_EQ(MultiCLEagleTargetFixture::kCL1, EagleTargetFixture::kContextLen);

    // A non-constant, EOS-free script. The target's batched verify reserves
    // seq_len_decode rows, so its usable window at cl16 is 16 - 4 = 12; prompt(3)
    // + kMax(6) = 9 positions clears the cl8 boundary (usable 8 - 4 = 4, so the
    // target promotes on the first verify) while staying under 12.
    const std::vector<int32_t> script = {1, 2, 3, 4, 6, 7, 1, 3, 2, 6, 4, 7, 1, 2, 3, 6};
    constexpr uint32_t         kVocab = MultiCLEagleTargetFixture::kVocab;
    const std::vector<int32_t> prompt = {1, 2, 3};
    const int32_t              kMax   = 6;

    // Reference: single-CL Eagle over the same script (already shown equal to a
    // plain AR target by MatchesPlainAutoregressiveUnderGreedy).
    std::vector<int32_t> single_cl_out;
    {
        StubScript         stub(script, kVocab);
        EagleTargetFixture tfx;
        EagleDraftFixture  dfx;
        TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
        ASSERT_TRUE(model.init(tfx, dfx));
        single_cl_out = model.generate(prompt, genConfig(kMax));
    }

    std::vector<int32_t> multi_cl_out;
    size_t               target_active_cl = 0;
    {
        StubScript                stub(script, kVocab);
        MultiCLEagleTargetFixture tfx;
        MultiCLEagleDraftFixture  dfx;
        TestableEagleModel        model{
            MultiCLEagleTargetFixture::makeSpec(), MultiCLEagleDraftFixture::makeSpec(), makeConfig({})};
        ASSERT_TRUE((model.init<MultiCLEagleTargetFixture, MultiCLEagleDraftFixture>(tfx, dfx)));

        // Must not throw ContextLengthExceededError as generation crosses cl8.
        EXPECT_NO_THROW(multi_cl_out = model.generate(prompt, genConfig(kMax)));
        target_active_cl = model.targetActiveCL();
    }

    ASSERT_FALSE(single_cl_out.empty());
    EXPECT_EQ(multi_cl_out, single_cl_out);
    EXPECT_GT(target_active_cl, 0u) << "target should have promoted past the initial context length";
}

}  // namespace

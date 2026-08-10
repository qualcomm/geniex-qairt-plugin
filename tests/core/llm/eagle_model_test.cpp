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
using geniex::testing::NoDecodePoolEnv;
using geniex::testing::TestableSpeculativeLLMModel;

// EagleModel with an injection seam: builds fixture-backed target/draft engines
// and marks itself ready, bypassing the QNN-only initialize().
class TestableEagleModel : public geniex::EagleModel {
   public:
    explicit TestableEagleModel(EagleConfig cfg)
        : geniex::EagleModel(EagleTargetFixture::makeSpec(), EagleDraftFixture::makeSpec(), std::move(cfg)) {}

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

// generate() before initialize() has run must fail loudly rather than
// dereference the un-loaded engines.
TEST(EagleModel, GenerateBeforeInitializeThrows) {
    NoDecodePoolEnv    no_pool;
    TestableEagleModel model{makeConfig(/*draft_token_map=*/{})};
    EXPECT_THROW(model.generate({1, 2, 3}, genConfig(4)), std::runtime_error);
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
// round. Exercises the accept-all path and argmaxDraft's identity branch.
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
    // acceptance rate must exceed 1.0 and match generated / iterations.
    const auto& stats = model.lastStats();
    EXPECT_EQ(stats.generated_tokens, out.size());
    EXPECT_GT(stats.iterations, 0u);
    EXPECT_GT(stats.acceptanceRate(), 1.0f);
    EXPECT_FLOAT_EQ(
        stats.acceptanceRate(), static_cast<float>(stats.generated_tokens) / static_cast<float>(stats.iterations));
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
    // N tokens takes N-1 speculation rounds, so acceptance rate sits just above
    // 1.0 (no draft proposals matched, only the guaranteed bonus token lands).
    const auto& stats = model.lastStats();
    EXPECT_EQ(stats.generated_tokens, out.size());
    EXPECT_EQ(stats.iterations, out.size() - 1);
    EXPECT_FLOAT_EQ(
        stats.acceptanceRate(), static_cast<float>(stats.generated_tokens) / static_cast<float>(stats.iterations));
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

}  // namespace

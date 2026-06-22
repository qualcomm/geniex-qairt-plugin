// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/llm_model.cpp - the prefill/decode orchestration
// loop. Drives a real LLMModel against a CPU-only graph fixture and the
// link-time QnnApi stub; no QNN device bring-up. A test-only subclass injects
// the loaded graphs and calls onInitialized() to bypass Model::initialize().

#include "llm/llm_model.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "QnnApi.hpp"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"

namespace {

using geniex::testing::LLMFixture;

// Exposes the protected Model members so a test can wire in pre-built graphs
// and run the orchestration entry points without a QNN backend.
class TestableLLMModel : public geniex::LLMModel {
   public:
    explicit TestableLLMModel(geniex::LLMSpec spec) : geniex::LLMModel(std::move(spec)) {}

    // Moves the fixture's graphs into the model and runs onInitialized(). The
    // fixture (which owns the IOTensor / QnnApi / tensor buffers the graphs
    // point at) must outlive this model. Templated so any fixture exposing
    // `.io` and `.graphs` (LLMFixture, MultiCLFixture) works.
    template <typename Fixture>
    bool initFromFixture(Fixture& fx) {
        api_       = std::make_unique<QnnApi>();
        io_tensor_ = std::shared_ptr<IOTensor>(std::shared_ptr<void>{}, &fx.io);  // non-owning alias
        for (auto& g : fx.graphs) graphs_.push_back(std::move(g));
        const bool ok = onInitialized();
        initialized_  = ok;
        return ok;
    }

    // Expose protected helpers for direct testing.
    using geniex::LLMModel::fmtPattern;
};

// Decode runs serially when no worker pool is created; the pool is only built
// when GENIEX_DECODE_WORKERS or the clock keeper is enabled. Force both off so
// updateKV happens inline and the loop is fully deterministic.
struct NoDecodePoolEnv {
    NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "0");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "0");
    }
    ~NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "");
    }
};

// Builds an initialized model over a fresh fixture. Holds both alive.
struct ModelFixture {
    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableLLMModel model{LLMFixture::makeSpec()};

    ModelFixture() { EXPECT_TRUE(model.initFromFixture(fx)); }
};

geniex::GenerationConfig greedyConfig(int32_t max_tokens) {
    geniex::GenerationConfig cfg;
    cfg.enable_sampling = false;  // argmax fast path
    cfg.max_tokens      = max_tokens;
    return cfg;
}

}  // namespace

// onInitialized derives the runtime shape from the loaded graph names.
TEST(LLMModel, InitializesFromGraphNames) {
    ModelFixture mf;
    EXPECT_EQ(mf.model.nPast(), 0u);
}

// A short prefill + single decode step emits exactly the token the stub was
// told to produce (greedy argmax, sampling disabled).
TEST(LLMModel, GreedyDecodeEmitsStubToken) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    const std::vector<int32_t> prompt = {1, 2, 3};
    auto                       out    = mf.model.generate(prompt, greedyConfig(/*max_tokens=*/1));

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 5);
    EXPECT_EQ(mf.model.nPast(), prompt.size() + 1);

    geniex::testing::stubSetNextToken(-1);
}

// Generation stops at an EOS token and excludes it from the output.
TEST(LLMModel, StopsOnEosAndExcludesIt) {
    geniex::LLMSpec spec = LLMFixture::makeSpec();
    spec.eos_token_ids   = {7};

    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableLLMModel model{spec};
    ASSERT_TRUE(model.initFromFixture(fx));

    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(7);  // first sampled token is EOS

    auto out = model.generate({1, 2}, greedyConfig(/*max_tokens=*/5));
    EXPECT_TRUE(out.empty());

    geniex::testing::stubSetNextToken(-1);
}

// token_callback returning false stops generation early.
TEST(LLMModel, CallbackStopsEarly) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(3);

    int  calls          = 0;
    auto stop_after_two = [&calls](int32_t) { return ++calls < 2; };
    auto out            = mf.model.generate({1}, greedyConfig(/*max_tokens=*/10), stop_after_two);

    EXPECT_EQ(calls, 2);
    EXPECT_EQ(out.size(), 2u);

    geniex::testing::stubSetNextToken(-1);
}

// A prompt longer than the max context length is rejected up front.
TEST(LLMModel, ThrowsWhenPromptExceedsContext) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(0);

    std::vector<int32_t> prompt(LLMFixture::kContextLen + 1, 1);
    EXPECT_THROW(mf.model.generate(prompt, greedyConfig(/*max_tokens=*/1)), geniex::ContextLengthExceededError);

    geniex::testing::stubSetNextToken(-1);
}

// fmtPattern substitutes the layer index for the "{}" placeholder.
TEST(LLMModel, FmtPattern) {
    EXPECT_EQ(TestableLLMModel::fmtPattern("past_key_{}_in", 3), "past_key_3_in");
    EXPECT_EQ(TestableLLMModel::fmtPattern("no_placeholder", 5), "no_placeholder");
}

// Multiple decode steps emit one token per step until max_tokens.
TEST(LLMModel, MultiStepDecode) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(4);

    auto out = mf.model.generate({1, 2}, greedyConfig(/*max_tokens=*/3));
    EXPECT_EQ(out.size(), 3u);
    for (int32_t t : out) EXPECT_EQ(t, 4);

    geniex::testing::stubSetNextToken(-1);
}

// enable_sampling drives the geniex-proc sampler chain (prepareSampler path)
// instead of the greedy argmax fast path.
TEST(LLMModel, SamplingPathRuns) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(2);  // one-hot peak -> sampler picks it

    geniex::GenerationConfig cfg;
    cfg.enable_sampling = true;
    cfg.temperature     = 0.0f;  // degenerates to greedy inside the chain
    cfg.max_tokens      = 2;
    auto out            = mf.model.generate({1}, cfg);

    EXPECT_EQ(out.size(), 2u);
    geniex::testing::stubSetNextToken(-1);
}

// resetKVCache returns the model to a pristine n_past, allowing re-generation.
TEST(LLMModel, ResetKVCache) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(3);

    mf.model.generate({1, 2}, greedyConfig(/*max_tokens=*/1));
    EXPECT_GT(mf.model.nPast(), 0u);

    mf.model.resetKVCache();
    EXPECT_EQ(mf.model.nPast(), 0u);

    geniex::testing::stubSetNextToken(-1);
}

// KV cache round-trips through a file: save after generation, load into a
// fresh model, n_past is restored.
TEST(LLMModel, SaveLoadKVCacheRoundTrip) {
    const auto path = (std::filesystem::temp_directory_path() / "geniex_kvcache.bin").string();

    {
        ModelFixture mf;
        geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
        geniex::testing::stubSetNextToken(3);
        mf.model.generate({1, 2}, greedyConfig(/*max_tokens=*/1));
        EXPECT_NO_THROW(mf.model.saveKVCacheToFile(path));
        geniex::testing::stubSetNextToken(-1);
    }

    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableLLMModel fresh{LLMFixture::makeSpec()};
    ASSERT_TRUE(fresh.initFromFixture(fx));
    EXPECT_EQ(fresh.nPast(), 0u);
    fresh.loadKVCacheFromFile(path);
    EXPECT_GT(fresh.nPast(), 0u);

    std::remove(path.c_str());
}

// save/load report errors on bad paths.
TEST(LLMModel, KVCacheFileErrors) {
    ModelFixture mf;
    EXPECT_THROW(mf.model.loadKVCacheFromFile("no_such_kvcache.bin"), std::runtime_error);
    EXPECT_THROW(mf.model.saveKVCacheToFile("Z:/nonexistent_dir/x/kvcache.bin"), std::runtime_error);
}

// With the decode pool enabled, the KV write-back runs through the threadpool
// path (decode_pool_ non-null) rather than inline.
TEST(LLMModel, DecodePoolPath) {
    _putenv_s("GENIEX_DECODE_WORKERS", "1");
    LLMFixture       fx;
    TestableLLMModel model{LLMFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(3);
    auto out = model.generate({1, 2}, greedyConfig(/*max_tokens=*/3));
    EXPECT_EQ(out.size(), 3u);

    geniex::testing::stubSetNextToken(-1);
    _putenv_s("GENIEX_DECODE_WORKERS", "");
}

// grammar_str set without a tokenizer takes the warn-and-skip path in
// prepareSampler (grammar disabled, generation still runs).
TEST(LLMModel, GrammarWithoutTokenizerWarns) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(2);

    geniex::GenerationConfig cfg;
    cfg.enable_sampling = true;
    cfg.grammar_str     = "root ::= \"a\"";
    cfg.tokenizer       = nullptr;  // no tokenizer -> grammar disabled
    cfg.max_tokens      = 1;
    auto out            = mf.model.generate({1}, cfg);
    EXPECT_EQ(out.size(), 1u);

    geniex::testing::stubSetNextToken(-1);
}

// A prompt that overruns the smaller CL triggers promoteCL -> reshapeKV,
// upgrading the active context length mid-prefill. Uses the 2-CL fixture.
TEST(LLMModel, PromotesContextLengthOnLongPrompt) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv  no_pool;
    MultiCLFixture   fx;
    TestableLLMModel model{MultiCLFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    geniex::testing::stubSetVocabSize(MultiCLFixture::kVocab);
    geniex::testing::stubSetNextToken(3);

    // CL0=8, ar_prefill=4 -> promotion fires when n_past+chunk > 8-4=4.
    // A 6-token prompt chunks as [4,2]; the second chunk promotes CL0 -> CL1.
    std::vector<int32_t> prompt(6, 1);
    auto                 out = model.generate(prompt, greedyConfig(/*max_tokens=*/1));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 3);

    geniex::testing::stubSetNextToken(-1);
}

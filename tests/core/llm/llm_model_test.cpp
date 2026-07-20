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
#include <fstream>
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
    using geniex::LLMModel::computeSlideDiscard;
    using geniex::LLMModel::fmtPattern;
    using geniex::LLMModel::spec_;
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

// Single-shard fixture whose first input is an integer token-id tensor
// (`input_ids`) alongside a float `input_embeds`. Exercises the isIntegerDtype
// guard: hidden_size must be read from the float embedding, never from the
// integer token-id input.
struct IntInputFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<geniex::testing::GraphInfoBuilder> builders;
    std::vector<geniex::Graph>                    graphs;

    IntInputFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addGraph("prefill_ar4_cl16_1_of_1", kArPrefill, kv_capacity);
        addGraph("token_ar1_cl16_1_of_1", kArDecode, kv_capacity);
    }

    IntInputFixture(const IntInputFixture&)            = delete;
    IntInputFixture& operator=(const IntInputFixture&) = delete;

    static geniex::LLMSpec makeSpec() {
        geniex::LLMSpec spec;
        spec.state_blocks.push_back(geniex::makeKVStateBlock());
        return spec;
    }

   private:
    void addGraph(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        using geniex::testing::TensorDesc;
        std::vector<TensorDesc> inputs{
            {"input_ids", QNN_DATATYPE_INT_32, {ar}},
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
            {"past_key_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}},
            {"past_value_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
            {"past_key_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}},
            {"past_value_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}},
        };
        builders.emplace_back(name, inputs, outputs);
        geniex::Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

}  // namespace

// onInitialized derives the runtime shape from the loaded graph names.
TEST(LLMModel, InitializesFromGraphNames) {
    ModelFixture mf;
    EXPECT_EQ(mf.model.nPast(), 0u);
}

// vocabSize() reports the value carried on LLMSpec (from the bundle's
// metadata.json logits shape at spec-build time), independent of nPast/graphs.
TEST(LLMModel, VocabSize) {
    ModelFixture mf;
    EXPECT_EQ(mf.model.vocabSize(), LLMFixture::kVocab);
}

// inferSpecFromGraphs derives every architecture shape and the KV tensor pairs
// purely from the loaded graph tensors — no metadata.json.
TEST(LLMModel, InfersShapesAndKVPairsFromGraphs) {
    ModelFixture mf;
    const auto&  spec = mf.model.spec_;
    EXPECT_EQ(spec.hidden_size, LLMFixture::kHidden);
    EXPECT_EQ(spec.num_kv_heads, LLMFixture::kKVHeads);
    EXPECT_EQ(spec.head_dim, LLMFixture::kHeadDim);
    EXPECT_EQ(spec.vocab_size, LLMFixture::kVocab);
    EXPECT_EQ(spec.seq_len_prefill, LLMFixture::kArPrefill);
    EXPECT_EQ(spec.seq_len_decode, LLMFixture::kArDecode);
    ASSERT_EQ(spec.context_lengths.size(), 1u);
    EXPECT_EQ(spec.context_lengths[0], LLMFixture::kContextLen);

    // One KV-only shard with kKVLayers resolved pairs, named by the default pattern.
    ASSERT_EQ(spec.state_blocks.size(), 1u);
    ASSERT_EQ(spec.state_blocks[0].shard_pairs.size(), 1u);
    ASSERT_EQ(spec.state_blocks[0].shard_pairs[0].size(), LLMFixture::kKVLayers);
    EXPECT_EQ(spec.state_blocks[0].shard_pairs[0][0].key_in, "past_key_0_in");
    EXPECT_EQ(spec.state_blocks[0].shard_pairs[0][0].value_out, "past_value_0_out");
}

// A multi-shard model resolves KV pairs only on the KV-owning shard; the
// lm-head-only shard gets an empty pair list.
TEST(LLMModel, InfersEmptyKVPairsForLMHeadShard) {
    using geniex::testing::MultiShardFixture;
    NoDecodePoolEnv   no_pool;
    MultiShardFixture fx;
    TestableLLMModel  model{MultiShardFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    const auto& pairs = model.spec_.state_blocks[0].shard_pairs;
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].size(), MultiShardFixture::kKVLayers);  // shard 0 owns KV
    EXPECT_TRUE(pairs[1].empty());                             // shard 1 is lm-head-only
    EXPECT_TRUE(model.spec_.shards[1].lm_head_only);
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

// computeSlideDiscard mirrors llama.cpp's context-shift heuristic: normally
// discards ~half of (n_past - n_keep), but discards at least enough to fit
// n_fit more when that alone demands more room.
TEST(LLMModel, ComputeSlideDiscardHalfWindow) {
    // n_past=4096, n_fit=1, max_cl=4096, n_keep=4:
    // needed = 4096+1-4096+1 = 2; half = 4096/2-4 = 2044; max(2044,2) = 2044.
    EXPECT_EQ(TestableLLMModel::computeSlideDiscard(4096, 1, 4096, 4), 2044u);
}

TEST(LLMModel, ComputeSlideDiscardNeededDominatesOnBigChunk) {
    // n_past=4090, n_fit=2048, max_cl=4096, n_keep=4:
    // needed = 4090+2048-4096+1 = 2043; half = 4090/2-4 = 2041; max(2041,2043) = 2043.
    EXPECT_EQ(TestableLLMModel::computeSlideDiscard(4090, 2048, 4096, 4), 2043u);
}

TEST(LLMModel, ComputeSlideDiscardNoOpWhenAtOrBelowNKeep) {
    EXPECT_EQ(TestableLLMModel::computeSlideDiscard(4, 1, 4096, 4), 0u);
    EXPECT_EQ(TestableLLMModel::computeSlideDiscard(2, 1, 4096, 4), 0u);
}

// With sliding_window enabled, a prompt that would otherwise exceed the max
// context length instead evicts the oldest tokens (above n_keep) and
// continues, rather than throwing ContextLengthExceededError.
TEST(LLMModel, SlidingWindowEvictsAndContinuesOnLongPrompt) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    // LLMFixture::kContextLen == 16. Prime n_past close to the ceiling with a
    // first turn, then send a second prompt that would overflow without eviction.
    geniex::GenerationConfig cfg = greedyConfig(/*max_tokens=*/1);
    cfg.sliding_window           = true;
    cfg.sliding_window_n_keep    = 2;

    auto out1 = mf.model.generate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, cfg);  // n_past -> 11
    ASSERT_EQ(out1.size(), 1u);
    EXPECT_EQ(mf.model.nPast(), 11u);

    // A 6-token prompt would push n_past to 11+6+1(decode)=18 > 16; without
    // sliding_window this throws (see ThrowsWhenPromptExceedsContext).
    auto out2 = mf.model.generate({11, 12, 13, 14, 15, 16}, cfg);
    ASSERT_EQ(out2.size(), 1u);
    EXPECT_EQ(out2[0], 5);
    // Some eviction must have happened: n_past after this call is strictly
    // less than what it would be without eviction (11+6+1=18), and still
    // fits within kContextLen.
    EXPECT_LE(mf.model.nPast(), LLMFixture::kContextLen);
    EXPECT_LT(mf.model.nPast(), 18u);

    geniex::testing::stubSetNextToken(-1);
}

// Without sliding_window, the same scenario as above still throws -- sliding
// eviction is strictly opt-in and doesn't change default behavior.
TEST(LLMModel, SlidingWindowDisabledByDefaultStillThrows) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    auto out1 = mf.model.generate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, greedyConfig(/*max_tokens=*/1));
    ASSERT_EQ(out1.size(), 1u);

    EXPECT_THROW(mf.model.generate({11, 12, 13, 14, 15, 16}, greedyConfig(/*max_tokens=*/1)),
        geniex::ContextLengthExceededError);

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

// A 2-shard model exercises discoverShardTensorNames (lm_head_only on shard 1),
// inter-shard hidden-state connections, and the LM-head skip on non-final
// prefill chunks (a multi-chunk prompt). The final logits come from shard 1.
TEST(LLMModel, MultiShardPrefillAndConnections) {
    using geniex::testing::MultiShardFixture;
    NoDecodePoolEnv   no_pool;
    MultiShardFixture fx;
    TestableLLMModel  model{MultiShardFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    geniex::testing::stubSetVocabSize(MultiShardFixture::kVocab);
    geniex::testing::stubSetNextToken(6);

    // 6-token prompt (> ar_prefill=4) -> chunks [4,2]; chunk 0 skips the
    // LM-head-only shard, chunk 1 runs it.
    std::vector<int32_t> prompt(6, 1);
    auto                 out = model.generate(prompt, greedyConfig(/*max_tokens=*/2));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 6);

    geniex::testing::stubSetNextToken(-1);
}

// inferSpecFromGraphs must read hidden_size from the named embedding tensor
// (`input_embeds`) and never from the token-id input (`input_ids`), even though
// input_ids is the first non-special input. in_state_name is still input_ids,
// which drives the token-id embedding provider.
TEST(LLMModel, IntegerInputSkippedForHiddenSize) {
    NoDecodePoolEnv  no_pool;
    IntInputFixture  fx;
    TestableLLMModel model{IntInputFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    EXPECT_EQ(model.spec_.hidden_size, IntInputFixture::kHidden);  // from input_embeds, not input_ids
    EXPECT_EQ(model.spec_.vocab_size, IntInputFixture::kVocab);
    EXPECT_EQ(model.spec_.shards[0].in_state_name, "input_ids");
}

namespace {

// Single-shard fixture whose hidden-state tensors carry arbitrary
// compiler-assigned names (`embedding` out) instead of a canonical
// `hidden_states` / `inputs_embeds`. This reproduces
// qualcomm/Qwen3-4B-Instruct-2507, whose graphs name the hidden state
// `embedding` / `add_82384`. hidden_size must still be inferred by tensor role
// + dtype, not by name. The first-shard input stays `input_ids` (as in the
// real model) so embedding-provider selection is unaffected.
struct ArbitraryHiddenNameFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<geniex::testing::GraphInfoBuilder> builders;
    std::vector<geniex::Graph>                    graphs;

    ArbitraryHiddenNameFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addGraph("prefill_ar4_cl16_1_of_1", kArPrefill, kv_capacity);
        addGraph("token_ar1_cl16_1_of_1", kArDecode, kv_capacity);
    }

    ArbitraryHiddenNameFixture(const ArbitraryHiddenNameFixture&)            = delete;
    ArbitraryHiddenNameFixture& operator=(const ArbitraryHiddenNameFixture&) = delete;

    static geniex::LLMSpec makeSpec() {
        geniex::LLMSpec spec;
        spec.state_blocks.push_back(geniex::makeKVStateBlock());
        return spec;
    }

   private:
    void addGraph(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        using geniex::testing::TensorDesc;
        // Token-id input; the hidden state is the arbitrarily-named float output.
        std::vector<TensorDesc> inputs{
            {"input_ids", QNN_DATATYPE_INT_32, {ar}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
            {"past_key_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}},
            {"past_value_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}},
        };
        std::vector<TensorDesc> outputs{
            {"add_82384", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
            {"past_key_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}},
            {"past_value_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}},
        };
        builders.emplace_back(name, inputs, outputs);
        geniex::Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

}  // namespace

// inferSpecFromGraphs must infer hidden_size from the hidden-state tensor's
// shape even when it has no canonical name — here the `add_82384` output, one
// of the arbitrary names qualcomm/Qwen3-4B-Instruct-2507 uses. Regression guard
// for the name-allowlist bug that failed such models.
TEST(LLMModel, InfersHiddenSizeFromArbitrarilyNamedTensor) {
    NoDecodePoolEnv            no_pool;
    ArbitraryHiddenNameFixture fx;
    TestableLLMModel           model{ArbitraryHiddenNameFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    EXPECT_EQ(model.spec_.hidden_size, ArbitraryHiddenNameFixture::kHidden);  // from add_82384 output
    EXPECT_EQ(model.spec_.vocab_size, ArbitraryHiddenNameFixture::kVocab);
}

namespace {

// Single-shard fixture carrying BOTH a global (past_*) and a sliding-window
// (swa_*) KV cache plus a swa_attention_mask, like Gemma3/4. The swa cache has a
// small fixed window so a modest prompt overflows it, exercising updateKV's
// window-wrap path (shiftKVLeft on key+value) that the global cache never hits.
// A second onInitialized-detected state block is auto-appended from swa_key_*.
struct SwaFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;
    static constexpr uint32_t kSwaWindow  = 4;  // swa cache capacity (small -> overflows)

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<geniex::testing::GraphInfoBuilder> builders;
    std::vector<geniex::Graph>                    graphs;

    SwaFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addGraph("prefill_ar4_cl16_1_of_1", kArPrefill, kv_capacity);
        addGraph("token_ar1_cl16_1_of_1", kArDecode, kv_capacity);
    }

    SwaFixture(const SwaFixture&)            = delete;
    SwaFixture& operator=(const SwaFixture&) = delete;

    static geniex::LLMSpec makeSpec() {
        geniex::LLMSpec spec;
        spec.state_blocks.push_back(geniex::makeKVStateBlock());
        // swa block is auto-detected + appended by onInitialized from swa_key_*.
        spec.swa_window = kSwaWindow;
        return spec;
    }

   private:
    void addGraph(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        using geniex::testing::TensorDesc;
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
            {"swa_attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kSwaWindow + ar}},
            {"past_key_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}},
            {"past_value_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}},
            {"swa_key_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kSwaWindow}},
            {"swa_value_0_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kSwaWindow, kHeadDim}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
            {"past_key_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}},
            {"past_value_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}},
            {"swa_key_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}},
            {"swa_value_0_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}},
        };
        builders.emplace_back(name, inputs, outputs);
        geniex::Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

}  // namespace

// onInitialized auto-detects the swa_key_* tensors and appends a second
// (sliding-window) KV state block alongside the global past_* block.
TEST(LLMModel, DetectsSlidingWindowKVBlock) {
    NoDecodePoolEnv  no_pool;
    SwaFixture       fx;
    TestableLLMModel model{SwaFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    // Two KV blocks: global past_* and the auto-appended swa_*.
    ASSERT_EQ(model.spec_.state_blocks.size(), 2u);
    EXPECT_EQ(model.spec_.state_blocks[1].key_out_pattern, "swa_key_{}_out");
    ASSERT_EQ(model.spec_.state_blocks[1].shard_pairs.size(), 1u);
    ASSERT_FALSE(model.spec_.state_blocks[1].shard_pairs[0].empty());
    EXPECT_EQ(model.spec_.state_blocks[1].shard_pairs[0][0].key_in, "swa_key_0_in");
}

// Generating past the swa window forces updateKV's overflow branch: the fixed
// swa cache (capacity kSwaWindow=4) shifts left (shiftKVLeft on key + value)
// once it fills, while the global cache just grows. Drives enough decode steps
// that the swa cache overflows, covering shiftKVLeft and the wrap path.
TEST(LLMModel, SlidingWindowKVCacheWrapsOnOverflow) {
    NoDecodePoolEnv  no_pool;
    SwaFixture       fx;
    TestableLLMModel model{SwaFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    geniex::testing::stubSetVocabSize(SwaFixture::kVocab);
    geniex::testing::stubSetNextToken(4);

    // Prompt (3) + several decode steps push total tokens well past the swa
    // window of 4, so updateKV must wrap the swa cache at least once.
    auto out = model.generate({1, 2, 3}, greedyConfig(/*max_tokens=*/6));
    ASSERT_EQ(out.size(), 6u);
    for (int32_t t : out) EXPECT_EQ(t, 4);
    EXPECT_EQ(model.nPast(), 3u + 6u);

    geniex::testing::stubSetNextToken(-1);
}

// ─────────────────────────────────────────────────────────────────────────────
// llm_spec_loader public API (JSON-sourced spec + provider factories)
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// Writes a minimal metadata.json + genie_config.json into a unique temp dir and
// removes the tree on destruction.
struct TempBundle {
    std::filesystem::path dir;

    TempBundle() {
        dir = std::filesystem::temp_directory_path() / ("geniex_loader_test_" + std::to_string(counter_++));
        std::filesystem::create_directories(dir);
        write("metadata.json", kMetadata);
        write("genie_config.json", kGenieConfig);
    }
    ~TempBundle() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    TempBundle(const TempBundle&)            = delete;
    TempBundle& operator=(const TempBundle&) = delete;

   private:
    void              write(const char* name, const char* body) const { std::ofstream(dir / name) << body; }
    static inline int counter_ = 0;

    static constexpr const char* kMetadata = R"({
        "model_id": "test_llm",
        "vision_preprocessing": {
            "image_width": 336, "image_height": 336, "patch_size": 14,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "normalize_mean": [0.5, 0.5, 0.5], "normalize_std": [0.5, 0.5, 0.5]
        },
        "model_files": {
            "ar4_cl16_1_of_1": {
                "inputs": {
                    "inputs_embeds": { "shape": [1, 4, 4] },
                    "attention_mask": { "shape": [1, 4, 16] },
                    "past_key_0_in": { "shape": [1, 1, 2, 16] }
                },
                "outputs": {
                    "logits": { "shape": [1, 4, 8] },
                    "past_key_0_out": { "shape": [1, 1, 2, 4] }
                }
            },
            "vision_encoder.bin": { "inputs": {}, "outputs": {} }
        }
    })";

    static constexpr const char* kGenieConfig = R"({
        "dialog": {
            "type": "basic",
            "context": { "bos-token": 1, "eos-token": [2, 3], "pad-token": 0 },
            "engine": {
                "model": {
                    "positional-encoding": {
                        "rope-theta": 1000000.0,
                        "rope-scaling": { "rope-type": "llama3", "factor": 8.0 }
                    }
                }
            },
            "sampler": { "seed": 42, "temp": 0.7, "top-k": 40, "top-p": 0.9 }
        }
    })";
};

}  // namespace

// parseQAIRTMetadata derives shapes and the vision block from metadata.json
// (retained for the VLM path and model_id dispatch).
TEST(LLMSpecLoader, ParsesMetadataShapesAndVision) {
    TempBundle bundle;
    const auto meta = geniex::parseQAIRTMetadata(bundle.dir);

    EXPECT_EQ(meta.model_id, "test_llm");
    EXPECT_EQ(meta.hidden_size, 4u);
    EXPECT_EQ(meta.num_kv_heads, 1u);
    EXPECT_EQ(meta.head_dim, 2u);
    EXPECT_EQ(meta.vocab_size, 8u);
    EXPECT_EQ(meta.num_hidden_layers, 1u);
    EXPECT_EQ(meta.first_shard_input_hint, "inputs_embeds");
    EXPECT_EQ(meta.vision_encoder_graph, "vision_encoder.bin");
    ASSERT_TRUE(meta.vision_preprocessing.has_value());
    EXPECT_EQ(meta.vision_preprocessing->image_width, 336);
    EXPECT_EQ(meta.vision_preprocessing->patch_size, 14);
}

// parseQAIRTMetadata throws when the bundle has no recognizable shard entries.
TEST(LLMSpecLoader, ParseMetadataThrowsWithoutShards) {
    const auto dir = std::filesystem::temp_directory_path() / "geniex_loader_empty";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "metadata.json") << R"({"model_files": {}})";
    EXPECT_THROW(geniex::parseQAIRTMetadata(dir), std::runtime_error);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// parseGenieConfig reads dialog tokens and the RoPE base/scaling.
TEST(LLMSpecLoader, ParsesGenieConfig) {
    TempBundle bundle;
    const auto gc = geniex::parseGenieConfig(bundle.dir);

    EXPECT_EQ(gc.dialog_type, "basic");
    EXPECT_EQ(gc.bos_token_id, 1);
    EXPECT_EQ(gc.pad_token_id, 0);
    ASSERT_EQ(gc.eos_token_ids.size(), 2u);
    EXPECT_EQ(gc.eos_token_ids[0], 2);
    EXPECT_EQ(gc.eos_token_ids[1], 3);
    EXPECT_FLOAT_EQ(gc.rope_theta, 1000000.0f);
    EXPECT_TRUE(std::holds_alternative<geniex::Llama3RopeScaling>(gc.rope_scaling));
}

// Gemma3/4 genie_config: proportional (partial-rotary) global RoPE, a separate
// local-positional-encoding block, and a perlayer-embedding stream. Exercises
// the Gemma-specific branches in parseGenieConfig / parseRopeScaling.
TEST(LLMSpecLoader, ParsesGemmaDualRopeAndPerLayerEmbedding) {
    const auto dir = std::filesystem::temp_directory_path() / "geniex_loader_gemma";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "genie_config.json") << R"({
        "dialog": {
            "type": "basic",
            "context": { "bos-token": 2, "eos-token": 1, "pad-token": 0 },
            "engine": {
                "model": {
                    "positional-encoding": {
                        "rope-theta": 1000000.0,
                        "rope-scaling": { "rope-type": "proportional", "partial-rotary-factor": 0.25 }
                    },
                    "local-positional-encoding": {
                        "rope-theta": 10000.0,
                        "rope-scaling": { "rope-type": "proportional", "partial-rotary-factor": 0.5 }
                    }
                }
            },
            "embedding": { "lut-path": "embedding_fp32.bin", "size": 1536 },
            "perlayer-embedding": { "lut-path": "per_layer_fp32.bin", "size": 8960 }
        }
    })";

    const auto gc = geniex::parseGenieConfig(dir);

    // Single-integer eos-token parses to a one-element list.
    ASSERT_EQ(gc.eos_token_ids.size(), 1u);
    EXPECT_EQ(gc.eos_token_ids[0], 1);
    // Global RoPE is proportional -> PartialRopeScaling.
    EXPECT_TRUE(std::holds_alternative<geniex::PartialRopeScaling>(gc.rope_scaling));
    // Local (sliding-window) RoPE block parsed with its own theta + scaling.
    EXPECT_TRUE(gc.local_positional_encoding_present);
    EXPECT_FLOAT_EQ(gc.local_rope_theta, 10000.0f);
    EXPECT_TRUE(std::holds_alternative<geniex::PartialRopeScaling>(gc.local_rope_scaling));
    // Both embedding streams resolved.
    ASSERT_TRUE(gc.embedding_lut_path.has_value());
    EXPECT_EQ(*gc.embedding_lut_path, "embedding_fp32.bin");
    ASSERT_TRUE(gc.perlayer_embedding_lut_path.has_value());
    EXPECT_EQ(*gc.perlayer_embedding_lut_path, "per_layer_fp32.bin");
    EXPECT_EQ(gc.perlayer_embedding_size, 8960u);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// parseGenieSamplerConfig reads the dialog.sampler defaults.
TEST(LLMSpecLoader, ParsesSamplerConfig) {
    TempBundle bundle;
    const auto s = geniex::parseGenieSamplerConfig(bundle.dir);

    ASSERT_TRUE(s.seed.has_value());
    EXPECT_EQ(*s.seed, 42u);
    ASSERT_TRUE(s.temperature.has_value());
    EXPECT_FLOAT_EQ(*s.temperature, 0.7f);
    ASSERT_TRUE(s.top_k.has_value());
    EXPECT_EQ(*s.top_k, 40);
}

// Missing genie_config.json yields all-default structs, never a throw.
TEST(LLMSpecLoader, MissingGenieConfigReturnsDefaults) {
    const auto dir = std::filesystem::temp_directory_path() / "geniex_loader_no_cfg";
    std::filesystem::create_directories(dir);
    const auto gc = geniex::parseGenieConfig(dir);
    EXPECT_EQ(gc.dialog_type, "basic");
    EXPECT_TRUE(gc.eos_token_ids.empty());
    EXPECT_TRUE(std::holds_alternative<geniex::StandardRope>(gc.rope_scaling));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// buildSpecSkeleton carries the JSON-sourced fields and a default KV block;
// tensor-derived fields stay zero until inference.
TEST(LLMSpecLoader, BuildsSkeletonSpec) {
    geniex::ParsedGenieConfig gc;
    gc.bos_token_id  = 1;
    gc.eos_token_ids = {2, 3};

    const auto spec = geniex::buildSpecSkeleton(gc);
    EXPECT_EQ(spec.bos_token_id, 1);
    EXPECT_EQ(spec.eos_token_ids, (std::vector<int32_t>{2, 3}));
    EXPECT_EQ(spec.hidden_size, 0u);  // filled later by inferSpecFromGraphs
    ASSERT_EQ(spec.state_blocks.size(), 1u);
    EXPECT_EQ(spec.state_blocks[0].kind, geniex::StateBlockKind::KV);
}

// makeRoPEProvider selects a non-null provider for every rope-scaling variant.
TEST(LLMSpecLoader, MakesRoPEProviderForEveryVariant) {
    constexpr size_t          kHeadDim = 64;
    geniex::ParsedGenieConfig gc;

    gc.rope_scaling = geniex::StandardRope{};
    EXPECT_NE(geniex::makeRoPEProvider(kHeadDim, gc), nullptr);

    gc.rope_scaling = geniex::Llama3RopeScaling{8.0f, 1.0f, 4.0f, 8192};
    EXPECT_NE(geniex::makeRoPEProvider(kHeadDim, gc), nullptr);

    geniex::LongRopeScaling lrs;
    lrs.long_factor                      = std::vector<float>(kHeadDim / 2, 1.0f);
    lrs.short_factor                     = std::vector<float>(kHeadDim / 2, 1.0f);
    lrs.original_max_position_embeddings = 4096;
    gc.rope_scaling                      = lrs;
    EXPECT_NE(geniex::makeRoPEProvider(kHeadDim, gc), nullptr);

    gc.rope_scaling = geniex::PartialRopeScaling{0.5f, 1.0f};
    EXPECT_NE(geniex::makeRoPEProvider(kHeadDim, gc), nullptr);

    geniex::MRopeScaling mrs;
    mrs.mrope_section = {16, 24, 24};
    gc.rope_scaling   = mrs;
    EXPECT_NE(geniex::makeRoPEProvider(kHeadDim, gc), nullptr);
}

// makeEmbeddingProvider maps the first-shard input name to a provider and
// rejects unknown names.
TEST(LLMSpecLoader, MakesEmbeddingProviderByInputName) {
    geniex::ParsedGenieConfig gc;
    gc.eos_token_ids = {2};

    EXPECT_NE(geniex::makeEmbeddingProvider("input_ids", gc), nullptr);
    EXPECT_NE(geniex::makeEmbeddingProvider("inputs_embeds", gc), nullptr);
    EXPECT_NE(geniex::makeEmbeddingProvider("input_embeds", gc), nullptr);
    EXPECT_THROW(geniex::makeEmbeddingProvider("bogus_tensor", gc), std::runtime_error);
}

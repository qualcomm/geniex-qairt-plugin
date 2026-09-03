// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/llm_model.cpp - the prefill/decode orchestration
// loop. Drives a real LLMModel against a CPU-only graph fixture and the
// link-time QnnApi stub; no QNN device bring-up. A test-only subclass injects
// the loaded graphs and calls onInitialized() to bypass Model::initialize().

#include "llm/llm_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "QnnApi.hpp"
#include "llm/kv_layout.h"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"
#include "testing/testable_llm_model.hpp"

namespace {

using geniex::testing::LLMFixture;
using geniex::testing::NoDecodePoolEnv;
using geniex::testing::TestableLLMModel;

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

// A prompt longer than the max context length is rejected up front (prefill),
// distinct from a window filled mid-generation (see ThrowsWhenGenerationExceedsContext).
TEST(LLMModel, ThrowsWhenPromptExceedsContext) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(0);

    std::vector<int32_t> prompt(LLMFixture::kContextLen + 1, 1);
    EXPECT_THROW(mf.model.generate(prompt, greedyConfig(/*max_tokens=*/1)), geniex::PromptTooLongError);

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
// eviction is strictly opt-in and doesn't change default behavior. The overflow
// is detected during prefill (the accumulated prompt doesn't fit), so it is a
// PromptTooLongError rather than a mid-generation ContextLengthExceededError.
TEST(LLMModel, SlidingWindowDisabledByDefaultStillThrows) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    auto out1 = mf.model.generate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, greedyConfig(/*max_tokens=*/1));
    ASSERT_EQ(out1.size(), 1u);

    EXPECT_THROW(
        mf.model.generate({11, 12, 13, 14, 15, 16}, greedyConfig(/*max_tokens=*/1)), geniex::PromptTooLongError);

    geniex::testing::stubSetNextToken(-1);
}

// A prompt that fits at prefill but whose generation would overflow the window
// forces eviction from *inside* the decode loop (slideWindowEvict's
// at_decode_stride=true path), not the prefill-time check above -- the KV
// buffer is at decode stride throughout and must be restrided to prefill
// stride, re-prefilled, then restrided back to decode stride in place.
TEST(LLMModel, SlidingWindowEvictsMidDecodeLoop) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    geniex::GenerationConfig cfg = greedyConfig(/*max_tokens=*/10);
    cfg.sliding_window           = true;
    cfg.sliding_window_n_keep    = 2;

    // LLMFixture::kContextLen == 16. A 10-token prompt leaves n_past == 10;
    // 6 decode steps reach n_past == 16, at which point the 7th step's
    // n_past_+1 > max_cl check must evict mid-loop to keep going.
    auto out = mf.model.generate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, cfg);
    EXPECT_EQ(out.size(), 10u);
    EXPECT_LE(mf.model.nPast(), LLMFixture::kContextLen);

    geniex::testing::stubSetNextToken(-1);
}

// A prompt that fits at prefill but whose generation fills the window mid-decode
// throws ContextLengthExceededError -- distinct from the up-front PromptTooLongError.
// With max_tokens large enough, the decode loop advances n_past past kContextLen.
TEST(LLMModel, ThrowsWhenGenerationExceedsContext) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);  // non-EOS: keeps decoding until the window fills

    // 10-token prompt prefills fine (n_past -> 11 after the first decode); asking
    // for more tokens than the remaining window forces a mid-generation overflow.
    EXPECT_THROW(mf.model.generate({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, greedyConfig(/*max_tokens=*/20)),
        geniex::ContextLengthExceededError);

    geniex::testing::stubSetNextToken(-1);
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

// graphIndex() addresses graphs_ as a dense phase x shard x cl grid. A hole in it
// used to resolve silently to a neighbouring graph rather than fail. This became
// reachable from our own code once the runtime started deliberately loading a
// subset of context-length variants, so it must be rejected at init.
TEST(LLMModel, RejectsIncompleteContextLengthGrid) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv no_pool;

    // The decode graph for the smaller CL is missing: 3 graphs for a 2x1x2 grid.
    MultiCLFixture   fx({
        {"prefill_ar4_cl8_1_of_1", MultiCLFixture::kArPrefill},
        {"prefill_ar4_cl16_1_of_1", MultiCLFixture::kArPrefill},
        {"token_ar1_cl16_1_of_1", MultiCLFixture::kArDecode},
    });
    TestableLLMModel model{MultiCLFixture::makeSpec()};
    EXPECT_FALSE(model.initFromFixture(fx));
}

// Two graphs claiming one (phase, shard, cl) slot is the other way the grid can be
// wrong, and cardinality alone would not catch it.
TEST(LLMModel, RejectsDuplicateGridSlot) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv no_pool;

    MultiCLFixture   fx({
        {"prefill_ar4_cl8_1_of_1", MultiCLFixture::kArPrefill},
        {"prefill_ar4_cl16_1_of_1", MultiCLFixture::kArPrefill},
        {"token_ar1_cl8_1_of_1", MultiCLFixture::kArDecode},
        {"token_ar1_cl16_1_of_1", MultiCLFixture::kArDecode},
        // Same phase/shard/cl as the graph above, different name.
        {"decode_ar1_cl16_1_of_1", MultiCLFixture::kArDecode},
    });
    TestableLLMModel model{MultiCLFixture::makeSpec()};
    EXPECT_FALSE(model.initFromFixture(fx));
}

// min_decode_seq_len (SSD's tree pass) opts a bundle into a third AR width,
// but only if some variant actually satisfies the requested width.
TEST(LLMModel, RejectsWhenMinDecodeSeqLenExceedsWidestArWidth) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv no_pool;

    MultiCLFixture  fx({
        {"prefill_ar4_cl16_1_of_1", MultiCLFixture::kArPrefill},
        {"token_ar1_cl16_1_of_1", MultiCLFixture::kArDecode},
    });
    geniex::LLMSpec spec    = MultiCLFixture::makeSpec();
    spec.min_decode_seq_len = 5;  // wider than the widest AR variant (4)
    TestableLLMModel model{spec};
    EXPECT_FALSE(model.initFromFixture(fx));
}

// With min_decode_seq_len satisfied, a third AR width is kept (not rejected)
// and the driver picks it as the decode width; the other two collapse the
// grid back down to a well-formed 2-phase x 1-shard x 1-CL layout.
TEST(LLMModel, AcceptsThirdArWidthWhenMinDecodeSeqLenIsSatisfied) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv no_pool;

    MultiCLFixture  fx({
        {"prefill_ar4_cl16_1_of_1", MultiCLFixture::kArPrefill},
        {"speculate_ar2_cl16_1_of_1", 2},
        {"token_ar1_cl16_1_of_1", MultiCLFixture::kArDecode},
    });
    geniex::LLMSpec spec    = MultiCLFixture::makeSpec();
    spec.min_decode_seq_len = 2;  // satisfied exactly by the ar2 variant
    TestableLLMModel model{spec};
    ASSERT_TRUE(model.initFromFixture(fx));
    EXPECT_EQ(model.spec_.seq_len_prefill, MultiCLFixture::kArPrefill);
    EXPECT_EQ(model.spec_.seq_len_decode, 2u);
}

// LLMSpec models exactly two AR lengths. A third has nowhere to go: it would be
// assigned phase 1 alongside decode and collide.
TEST(LLMModel, RejectsMoreThanTwoArLengths) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv no_pool;

    MultiCLFixture   fx({
        {"prefill_ar4_cl16_1_of_1", MultiCLFixture::kArPrefill},
        {"speculate_ar2_cl16_1_of_1", 2},
        {"token_ar1_cl16_1_of_1", MultiCLFixture::kArDecode},
    });
    TestableLLMModel model{MultiCLFixture::makeSpec()};
    EXPECT_FALSE(model.initFromFixture(fx));
}

// The well-formed grid must still be accepted — the guard above is not allowed to
// reject the shape every real bundle has.
TEST(LLMModel, AcceptsCompleteContextLengthGrid) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv  no_pool;
    MultiCLFixture   fx;
    TestableLLMModel model{MultiCLFixture::makeSpec()};
    EXPECT_TRUE(model.initFromFixture(fx));
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

// Regression: reshapeKV must physically restride the global (CL-scaled) KV
// cache when growing from a smaller to a larger context length. The Gemma3/4
// multi-block refactor guarded the restride on `kvCapacityOf(...) == old_kv_len`,
// which is false whenever the prefill graph's logical kv_len differs from the
// live n_past (always true on real multi-CL models). That skipped the primary
// block's restride, leaving rows at the old stride so decode past the first CL
// bucket read stale KV -> runaway repetition. This test seeds the key/value
// buffers at the OLD stride, calls reshapeKV, and asserts every row moved to the
// NEW stride. It fails against the buggy guard and passes with the kind-based one.
TEST(LLMModel, ReshapeKVRestridesGlobalCacheOnExpand) {
    using geniex::testing::MultiCLFixture;
    NoDecodePoolEnv  no_pool;
    MultiCLFixture   fx;
    TestableLLMModel model{MultiCLFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    // active_cl_idx_ stays 0: reshapeKV reads the phase-0 graph at the OLD cl.
    model.active_cl_idx_ = 0;
    geniex::Graph& g     = model.graph(model.graphIndex(/*phase=*/0, /*shard=*/0, /*cl_idx=*/0));

    const geniex::StateBlockSpec& kv = model.requireKVStateBlock();
    ASSERT_EQ(kv.kind, geniex::StateBlockKind::KV);
    ASSERT_FALSE(kv.shard_pairs.empty());
    ASSERT_FALSE(kv.shard_pairs[0].empty());
    const auto& pair = kv.shard_pairs[0][0];

    // Key layout [kKVHeads, 1, kHeadDim, kv_len] -> n_rows = kKVHeads * kHeadDim.
    const geniex::TensorSpec& key_spec = g.inputSpec(pair.key_in);
    const size_t              key_rows = key_spec.shape[0] * key_spec.shape[2];
    // Value layout [kKVHeads, 1, kv_len, kHeadDim] -> n_rows = kKVHeads, row width = kHeadDim.
    const geniex::TensorSpec& val_spec  = g.inputSpec(pair.value_in);
    const size_t              val_rows  = val_spec.shape[0];
    const size_t              val_width = val_spec.shape[3];

    // The physical buffer is pre-sized to the MAX stride any phase reshapes to
    // (kCL1 - kArDecode); the new stride must not exceed that per-row capacity or
    // reshapeKV (and this test's seed) would overrun the buffer. Grow into it.
    const size_t kNewKv = key_spec.shape[3];  // == kCL1 - kArDecode
    const size_t kOldKv = MultiCLFixture::kCL0 - MultiCLFixture::kArDecode;
    const size_t kValid = kOldKv;  // all old-stride tokens are live
    ASSERT_LT(kOldKv, kNewKv);

    // Seed each key row r with a unique ramp [r*100 .. r*100 + kOldKv-1] at the
    // OLD stride, then fill the whole (max-capacity) buffer with a sentinel so a
    // skipped restride can't accidentally leave the expected value at the new offset.
    auto* key_buf = static_cast<float*>(g.inputPtr(pair.key_in));
    std::fill_n(key_buf, key_rows * kNewKv, -1.0f);
    for (size_t r = 0; r < key_rows; ++r)
        for (size_t t = 0; t < kOldKv; ++t) key_buf[r * kOldKv + t] = static_cast<float>(r * 100 + t);

    auto* val_buf = static_cast<float*>(g.inputPtr(pair.value_in));
    std::fill_n(val_buf, val_rows * kNewKv * val_width, -1.0f);
    for (size_t r = 0; r < val_rows; ++r)
        for (size_t t = 0; t < kOldKv; ++t)
            for (size_t d = 0; d < val_width; ++d)
                val_buf[(r * kOldKv + t) * val_width + d] = static_cast<float>(r * 1000 + t * 10 + d);

    model.reshapeKV(/*shard=*/0, kOldKv, kNewKv, kValid);

    // After restride, row r's live tokens must be readable at the NEW stride.
    for (size_t r = 0; r < key_rows; ++r)
        for (size_t t = 0; t < kOldKv; ++t)
            EXPECT_FLOAT_EQ(key_buf[r * kNewKv + t], static_cast<float>(r * 100 + t))
                << "key row " << r << " tok " << t;

    for (size_t r = 0; r < val_rows; ++r)
        for (size_t t = 0; t < kOldKv; ++t)
            for (size_t d = 0; d < val_width; ++d)
                EXPECT_FLOAT_EQ(val_buf[(r * kNewKv + t) * val_width + d], static_cast<float>(r * 1000 + t * 10 + d))
                    << "val row " << r << " tok " << t << " dim " << d;
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

// A bundle whose shards are `2_of_3` / `3_of_3` with no `1_of_3` (the leading
// shard is an off-graph CPU embedding LUT, as in EAGLE/eaglet). onInitialized
// must count the LOADED shards (2), not the `_of_T` total (3): otherwise the
// KV/graph tables are sized to 3 and initKVBuffers indexes past the 4 loaded
// graphs (historically an out_of_range throw). A full prefill+decode then
// exercises the dense shard-rank remap end-to-end.
TEST(LLMModel, ExternalLeadingShardInitAndDecode) {
    using geniex::testing::ExternalLeadingShardFixture;
    NoDecodePoolEnv             no_pool;
    ExternalLeadingShardFixture fx;
    TestableLLMModel            model{ExternalLeadingShardFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    // Two loaded shards, ranked to dense slots 0 (body, owns KV) and 1 (lm-head).
    ASSERT_EQ(model.spec_.shards.size(), 2u);
    EXPECT_TRUE(model.spec_.shards[1].lm_head_only);
    const auto& pairs = model.spec_.state_blocks[0].shard_pairs;
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].size(), ExternalLeadingShardFixture::kKVLayers);
    EXPECT_TRUE(pairs[1].empty());

    geniex::testing::stubSetVocabSize(ExternalLeadingShardFixture::kVocab);
    geniex::testing::stubSetNextToken(6);
    auto out = model.generate({1, 2, 3}, greedyConfig(/*max_tokens=*/2));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 6);
    geniex::testing::stubSetNextToken(-1);
}

// prefill's optional capture reassembles one body-feature row per token
// across every prefill chunk. The prefill output buffer alone retains only the
// final chunk, so a >seq_len_prefill prompt must be captured incrementally —
// the EAGLE draft-seed path depends on the full per-position feature history.
TEST(LLMModel, PrefillCapturesFeaturesAcrossChunks) {
    using geniex::testing::ExternalLeadingShardFixture;
    NoDecodePoolEnv             no_pool;
    ExternalLeadingShardFixture fx;
    TestableLLMModel            model{ExternalLeadingShardFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    // ar_prefill=4, so a 6-token prompt chunks as [4, 2].
    const std::vector<int32_t> prompt(6, 1);
    std::vector<uint8_t>       feats;
    model.prefill(prompt,
        /*rope_theta=*/1000000.0f,
        /*feature_rows=*/nullptr,
        /*feature_row_bytes=*/0,
        /*feature_name=*/"",
        &feats,
        "last_hidden_states");

    const size_t body = model.graphIndex(/*phase=*/0, /*shard=*/0, /*cl=*/0);
    const auto&  spec = model.outputTensorSpec(body, "last_hidden_states");
    const size_t row  = spec.shape.back() * spec.elementSize();
    EXPECT_EQ(feats.size(), prompt.size() * row);  // one row per prompt token
    EXPECT_EQ(model.nPast(), prompt.size());
}

// forwardLogits (final-token mode) runs a single prefill pass and returns one
// vocab-sized logits row. The stub writes a one-hot peak at g_next_token on
// every row, so the returned row's argmax is that token.
TEST(LLMModel, ForwardLogitsLastPosition) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    auto logits = mf.model.forwardLogits({1, 2, 3}, /*all_positions=*/false);

    ASSERT_EQ(logits.size(), LLMFixture::kVocab);
    const auto argmax = std::max_element(logits.begin(), logits.end()) - logits.begin();
    EXPECT_EQ(argmax, 5);
    // KV cache is left clean for the next caller.
    EXPECT_EQ(mf.model.nPast(), 0u);

    geniex::testing::stubSetNextToken(-1);
}

// forwardLogits (all-positions mode) returns [n_tokens, vocab] row-major. Every
// row is the stub's one-hot peak, so each position's argmax is g_next_token.
TEST(LLMModel, ForwardLogitsAllPositions) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(2);

    const std::vector<int32_t> tokens = {1, 2, 3};
    auto                       logits = mf.model.forwardLogits(tokens, /*all_positions=*/true);

    ASSERT_EQ(logits.size(), tokens.size() * LLMFixture::kVocab);
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        const float* row    = logits.data() + pos * LLMFixture::kVocab;
        const auto   argmax = std::max_element(row, row + LLMFixture::kVocab) - row;
        EXPECT_EQ(argmax, 2) << "position " << pos;
    }
    EXPECT_EQ(mf.model.nPast(), 0u);

    geniex::testing::stubSetNextToken(-1);
}

// all-positions mode must span multiple prefill chunks: a prompt longer than
// seq_len_prefill chunks as [4, 2], and forwardLogits must collect logits for
// every position across both chunks (the LM head runs on the non-final chunk too).
TEST(LLMModel, ForwardLogitsAllPositionsSpansChunks) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(1);

    std::vector<int32_t> tokens(LLMFixture::kArPrefill + 2, 7);  // 6 tokens -> chunks [4, 2]
    auto                 logits = mf.model.forwardLogits(tokens, /*all_positions=*/true);

    ASSERT_EQ(logits.size(), tokens.size() * LLMFixture::kVocab);
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        const float* row    = logits.data() + pos * LLMFixture::kVocab;
        const auto   argmax = std::max_element(row, row + LLMFixture::kVocab) - row;
        EXPECT_EQ(argmax, 1) << "position " << pos;
    }

    geniex::testing::stubSetNextToken(-1);
}

// forwardLogits validates its input: empty tokens throw, and an over-long
// sequence throws ContextLengthExceededError (without consuming KV state).
TEST(LLMModel, ForwardLogitsRejectsBadInput) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);

    EXPECT_THROW(mf.model.forwardLogits({}, /*all_positions=*/false), std::invalid_argument);

    std::vector<int32_t> too_long(LLMFixture::kContextLen + 1, 1);
    EXPECT_THROW(mf.model.forwardLogits(too_long, /*all_positions=*/false), geniex::ContextLengthExceededError);
}

// forwardLogits repeated calls are independent: each starts from a clean KV
// cache, so a second call over the same tokens matches the first.
TEST(LLMModel, ForwardLogitsIsRepeatable) {
    ModelFixture mf;
    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(4);

    const std::vector<int32_t> tokens = {1, 2};
    auto                       first  = mf.model.forwardLogits(tokens, /*all_positions=*/false);
    auto                       second = mf.model.forwardLogits(tokens, /*all_positions=*/false);
    EXPECT_EQ(first, second);

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

// reshapeKV must skip sliding-window (swa_*) blocks: their kv_len is a fixed
// window that stays constant across CL variants, so restriding them would
// corrupt the cache. This asserts the swa block is tagged SlidingWindowKV (so
// the kind-based guard excludes it) yet is still a discoverable KV block, and
// that a reshapeKV call leaves the swa buffer byte-for-byte unchanged.
TEST(LLMModel, ReshapeKVLeavesSlidingWindowBlockUntouched) {
    NoDecodePoolEnv  no_pool;
    SwaFixture       fx;
    TestableLLMModel model{SwaFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));

    ASSERT_EQ(model.spec_.state_blocks.size(), 2u);
    const geniex::StateBlockSpec& swa = model.spec_.state_blocks[1];
    EXPECT_EQ(swa.kind, geniex::StateBlockKind::SlidingWindowKV);
    EXPECT_TRUE(geniex::isKVStateBlock(swa.kind));  // still a KV block for updateKV / name-set paths
    ASSERT_FALSE(swa.shard_pairs.empty());
    ASSERT_FALSE(swa.shard_pairs[0].empty());
    const auto& swa_pair = swa.shard_pairs[0][0];

    model.active_cl_idx_ = 0;
    geniex::Graph& g     = model.graph(model.graphIndex(/*phase=*/0, /*shard=*/0, /*cl_idx=*/0));

    auto*        swa_key = static_cast<float*>(g.inputPtr(swa_pair.key_in));
    const size_t win     = g.inputSpec(swa_pair.key_in).byteCount() / sizeof(float);
    for (size_t i = 0; i < win; ++i) swa_key[i] = static_cast<float>(i + 1);
    const std::vector<float> before(swa_key, swa_key + win);

    model.reshapeKV(/*shard=*/0, /*old_kv_len=*/8, /*new_kv_len=*/15, /*n_valid=*/8);

    const std::vector<float> after(swa_key, swa_key + win);
    EXPECT_EQ(before, after);  // swa window untouched by a global-cache restride
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

// isEndOfGeneration returns true for a token in spec_.eos_token_ids even when
// no tokenizer is present, and false for any other token.
TEST(LLMModel, IsEndOfGenerationMatchesSpecEosIds) {
    geniex::LLMSpec spec = LLMFixture::makeSpec();
    spec.eos_token_ids   = {7, 42};
    TestableLLMModel model{spec};

    geniex::GenerationConfig cfg;  // no tokenizer

    EXPECT_TRUE(model.isEndOfGeneration(7, cfg));
    EXPECT_TRUE(model.isEndOfGeneration(42, cfg));
    EXPECT_FALSE(model.isEndOfGeneration(1, cfg));
    EXPECT_FALSE(model.isEndOfGeneration(0, cfg));
}

// isEndOfGeneration returns false (not a crash) when gen_cfg.tokenizer is null,
// even for a token that is not in eos_token_ids.
TEST(LLMModel, IsEndOfGenerationNullTokenizerIsSafe) {
    ModelFixture             mf;
    geniex::GenerationConfig cfg;  // tokenizer = nullptr
    EXPECT_FALSE(mf.model.isEndOfGeneration(99, cfg));
}

// discoverKVPairs resolves per-head KV tensors whose names carry an optional
// `_h<n>` suffix (e.g. `swa_key_0_h0_out`, `swa_key_0_h1_out`). Each matched
// output becomes its own KVTensorPair; the suffix is preserved in all siblings.
TEST(LLMModel, DiscoverKVPairsHandlesPerHeadSuffix) {
    using geniex::testing::GraphInfoBuilder;
    using geniex::testing::TensorDesc;

    // Two-head SWA export: layer 0 has h0 and h1 tensors.
    static constexpr uint32_t kH  = 1;
    static constexpr uint32_t kHD = 2;
    static constexpr uint32_t kKV = 4;
    static constexpr uint32_t kAR = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::vector<TensorDesc> inputs{
        {"input_embeds", QNN_DATATYPE_FLOAT_32, {kAR, 4}},
        {"swa_key_0_h0_in", QNN_DATATYPE_FLOAT_32, {kH, 1, kHD, kKV}},
        {"swa_value_0_h0_in", QNN_DATATYPE_FLOAT_32, {kH, 1, kKV, kHD}},
        {"swa_key_0_h1_in", QNN_DATATYPE_FLOAT_32, {kH, 1, kHD, kKV}},
        {"swa_value_0_h1_in", QNN_DATATYPE_FLOAT_32, {kH, 1, kKV, kHD}},
    };
    std::vector<TensorDesc> outputs{
        {"logits", QNN_DATATYPE_FLOAT_32, {kAR, 8}},
        {"swa_key_0_h0_out", QNN_DATATYPE_FLOAT_32, {kH, 1, kHD, kAR}},
        {"swa_value_0_h0_out", QNN_DATATYPE_FLOAT_32, {kH, 1, kAR, kHD}},
        {"swa_key_0_h1_out", QNN_DATATYPE_FLOAT_32, {kH, 1, kHD, kAR}},
        {"swa_value_0_h1_out", QNN_DATATYPE_FLOAT_32, {kH, 1, kAR, kHD}},
    };

    GraphInfoBuilder builder("token_ar1_cl8_1_of_1", inputs, outputs);
    geniex::Graph    g(&builder.graphInfo(), &api, &io);
    g.setup(/*context=*/nullptr);

    geniex::StateBlockSpec block = geniex::makeSwaKVStateBlock();
    auto                   pairs = TestableLLMModel::discoverKVPairs(g, block);

    // Both heads resolved as separate pairs with the _h<n> suffix preserved.
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].key_out, "swa_key_0_h0_out");
    EXPECT_EQ(pairs[0].key_in, "swa_key_0_h0_in");
    EXPECT_EQ(pairs[0].value_out, "swa_value_0_h0_out");
    EXPECT_EQ(pairs[0].value_in, "swa_value_0_h0_in");
    EXPECT_EQ(pairs[1].key_out, "swa_key_0_h1_out");
    EXPECT_EQ(pairs[1].key_in, "swa_key_0_h1_in");
    EXPECT_EQ(pairs[1].value_out, "swa_value_0_h1_out");
    EXPECT_EQ(pairs[1].value_in, "swa_value_0_h1_in");
}

// adoptKVNamingFromGraph derives the primary KV block's naming patterns from a
// graph's own tensors when the declared default (past_key_{}_in / _out, etc.)
// matches nothing -- e.g. the Llama-3.2-3B SSD w4a16 export's
// `past_nativekvcache__key_<layer>_head_<h>_in` naming.
TEST(LLMModel, AdoptKVNamingFromGraphDerivesNonDefaultNaming) {
    using geniex::testing::GraphInfoBuilder;
    using geniex::testing::TensorDesc;

    QnnApi                  api;
    IOTensor                io{BufferAlloc::DEFAULT};
    std::vector<TensorDesc> inputs{
        {"input_embeds", QNN_DATATYPE_FLOAT_32, {1, 4}},
        // A key-ish output with no matching pattern precedes the real one, and
        // has no digit right after its "key_" prefix -- adoptKVNamingFromGraph
        // must skip it (continue) rather than adopt it.
        {"past_nativekvcache__key_0_head_0_in", QNN_DATATYPE_FLOAT_32, {1, 1, 2, 4}},
        {"past_nativekvcache__value_0_head_0_in", QNN_DATATYPE_FLOAT_32, {1, 1, 4, 2}},
    };
    std::vector<TensorDesc> outputs{
        {"logits", QNN_DATATYPE_FLOAT_32, {1, 8}},
        {"bogus_key_text_out", QNN_DATATYPE_FLOAT_32, {1, 1}},  // "key"-bearing but non-numeric middle
        {"past_nativekvcache__key_0_head_0_out", QNN_DATATYPE_FLOAT_32, {1, 1, 2, 1}},
        {"past_nativekvcache__value_0_head_0_out", QNN_DATATYPE_FLOAT_32, {1, 1, 1, 2}},
    };
    GraphInfoBuilder builder("token_ar1_cl8_1_of_1", inputs, outputs);
    geniex::Graph    g(&builder.graphInfo(), &api, &io);
    g.setup(/*context=*/nullptr);

    geniex::LLMSpec spec;
    spec.state_blocks.push_back(geniex::makeKVStateBlock());
    TestableLLMModel model{spec};

    EXPECT_TRUE(model.adoptKVNamingFromGraph(g, /*shard=*/0));
    ASSERT_EQ(model.spec_.state_blocks.size(), 1u);
    const auto& block = model.spec_.state_blocks[0];
    EXPECT_EQ(block.key_in_pattern, "past_nativekvcache__key_{}_in");
    EXPECT_EQ(block.key_out_pattern, "past_nativekvcache__key_{}_out");
    EXPECT_EQ(block.value_in_pattern, "past_nativekvcache__value_{}_in");
    EXPECT_EQ(block.value_out_pattern, "past_nativekvcache__value_{}_out");
}

// If the block's patterns already match what the graph exposes, there is
// nothing to adopt -- adoptKVNamingFromGraph must report false rather than
// signalling a (spurious) change.
TEST(LLMModel, AdoptKVNamingFromGraphNoOpWhenAlreadyCorrect) {
    using geniex::testing::GraphInfoBuilder;
    using geniex::testing::TensorDesc;

    QnnApi                  api;
    IOTensor                io{BufferAlloc::DEFAULT};
    std::vector<TensorDesc> inputs{
        {"input_embeds", QNN_DATATYPE_FLOAT_32, {1, 4}},
        {"custom_key_0_in", QNN_DATATYPE_FLOAT_32, {1, 1, 2, 4}},
        {"custom_value_0_in", QNN_DATATYPE_FLOAT_32, {1, 1, 4, 2}},
    };
    std::vector<TensorDesc> outputs{
        {"logits", QNN_DATATYPE_FLOAT_32, {1, 8}},
        {"custom_key_0_out", QNN_DATATYPE_FLOAT_32, {1, 1, 2, 1}},
        {"custom_value_0_out", QNN_DATATYPE_FLOAT_32, {1, 1, 1, 2}},
    };
    GraphInfoBuilder builder("token_ar1_cl8_1_of_1", inputs, outputs);
    geniex::Graph    g(&builder.graphInfo(), &api, &io);
    g.setup(/*context=*/nullptr);

    geniex::LLMSpec        spec;
    geniex::StateBlockSpec block = geniex::makeKVStateBlock();
    block.key_in_pattern         = "custom_key_{}_in";
    block.key_out_pattern        = "custom_key_{}_out";
    block.value_in_pattern       = "custom_value_{}_in";
    block.value_out_pattern      = "custom_value_{}_out";
    spec.state_blocks.push_back(block);
    TestableLLMModel model{spec};

    EXPECT_FALSE(model.adoptKVNamingFromGraph(g, /*shard=*/0));
}

// No key-bearing output tensor exists at all: the discovery loop runs to
// completion without adopting anything.
TEST(LLMModel, AdoptKVNamingFromGraphFalseWhenNoKeyOutputPresent) {
    using geniex::testing::GraphInfoBuilder;
    using geniex::testing::TensorDesc;

    QnnApi                  api;
    IOTensor                io{BufferAlloc::DEFAULT};
    std::vector<TensorDesc> inputs{{"input_embeds", QNN_DATATYPE_FLOAT_32, {1, 4}}};
    std::vector<TensorDesc> outputs{{"logits", QNN_DATATYPE_FLOAT_32, {1, 8}}};
    GraphInfoBuilder        builder("token_ar1_cl8_1_of_1", inputs, outputs);
    geniex::Graph           g(&builder.graphInfo(), &api, &io);
    g.setup(/*context=*/nullptr);

    geniex::LLMSpec spec;
    spec.state_blocks.push_back(geniex::makeKVStateBlock());
    TestableLLMModel model{spec};

    EXPECT_FALSE(model.adoptKVNamingFromGraph(g, /*shard=*/0));
}

// A tiled (HMX_WEIGHT_LAYOUT) cache can only shift in whole 32-token blocks;
// an unaligned shift degrades to an element-wise re-tile and logs a one-time
// warning. This drives that specific branch directly, bypassing the full
// decode loop.
TEST(LLMModel, ShiftKVLeftWarnsOnceForUnalignedTiledShift) {
    using geniex::testing::GraphInfoBuilder;
    using geniex::testing::TensorDesc;

    QnnApi     api;
    IOTensor   io{BufferAlloc::DEFAULT};
    TensorDesc key_in{"past_key_0_in", QNN_DATATYPE_UFIXED_POINT_8, {1, 1, 32, 64}};
    key_in.data_format = QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
    std::vector<TensorDesc> inputs{key_in};
    std::vector<TensorDesc> outputs{{"logits", QNN_DATATYPE_FLOAT_32, {1, 8}}};
    GraphInfoBuilder        builder("token_ar1_cl64_1_of_1", inputs, outputs);
    geniex::Graph           g(&builder.graphInfo(), &api, &io);
    g.setup(/*context=*/nullptr);

    TestableLLMModel model{geniex::LLMSpec{}};
    // Two calls: the static "warned" latch means only the first actually logs,
    // but both must run the shift itself without throwing.
    EXPECT_NO_THROW(model.shiftKVLeft(g, "past_key_0_in", /*shift=*/5, /*is_key=*/true));
    EXPECT_NO_THROW(model.shiftKVLeft(g, "past_key_0_in", /*shift=*/3, /*is_key=*/true));
}

// A model can carry both a flat global KV block and a tiled (HMX) sliding-
// window block at once (a real export could ship a tiled swa_* cache
// alongside a flat global one). The layout-detection pass at init must accept
// the mix, and the tiled swa_* block alone must not flip native_kv_ (which
// changes the *global* cache's stride arithmetic).
TEST(LLMModel, DetectsMixedFlatAndTiledKVLayout) {
    using geniex::testing::GraphInfoBuilder;
    using geniex::testing::TensorDesc;
    NoDecodePoolEnv no_pool;

    static constexpr uint32_t kSwaHeadDim = 32;
    static constexpr uint32_t kSwaWindow  = 32;

    QnnApi                       api;
    IOTensor                     io{BufferAlloc::DEFAULT};
    std::deque<GraphInfoBuilder> builders;
    std::vector<geniex::Graph>   graphs;

    auto addGraph = [&](const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, 4}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, 16}},
            {"swa_attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kSwaWindow + ar}},
            {"past_key_0_in", QNN_DATATYPE_FLOAT_32, {1, 1, 2, 15}},
            {"past_value_0_in", QNN_DATATYPE_FLOAT_32, {1, 1, 15, 2}},
        };
        TensorDesc swa_key_in{"swa_key_0_in", QNN_DATATYPE_UFIXED_POINT_8, {1, 1, kSwaHeadDim, kSwaWindow}};
        swa_key_in.data_format = QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
        TensorDesc swa_val_in{"swa_value_0_in", QNN_DATATYPE_UFIXED_POINT_8, {1, 1, kSwaWindow, kSwaHeadDim}};
        swa_val_in.data_format = QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
        inputs.push_back(swa_key_in);
        inputs.push_back(swa_val_in);

        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, 8}},
            {"past_key_0_out", QNN_DATATYPE_FLOAT_32, {1, 1, 2, ar}},
            {"past_value_0_out", QNN_DATATYPE_FLOAT_32, {1, 1, ar, 2}},
        };
        TensorDesc swa_key_out{"swa_key_0_out", QNN_DATATYPE_UFIXED_POINT_8, {1, 1, kSwaHeadDim, ar}};
        swa_key_out.data_format = QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
        TensorDesc swa_val_out{"swa_value_0_out", QNN_DATATYPE_UFIXED_POINT_8, {1, 1, ar, kSwaHeadDim}};
        swa_val_out.data_format = QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
        outputs.push_back(swa_key_out);
        outputs.push_back(swa_val_out);

        builders.emplace_back(name, inputs, outputs);
        geniex::Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    };
    addGraph("prefill_ar4_cl16_1_of_1", 4);
    addGraph("token_ar1_cl16_1_of_1", 1);

    struct Fixture {
        IOTensor&                   io;
        std::vector<geniex::Graph>& graphs;
    } fx{io, graphs};

    geniex::LLMSpec spec;
    spec.state_blocks.push_back(geniex::makeKVStateBlock());
    spec.swa_window = kSwaWindow;
    TestableLLMModel model{spec};
    ASSERT_TRUE(model.initFromFixture(fx));

    ASSERT_EQ(model.spec_.state_blocks.size(), 2u);
    geniex::Graph& g0 = model.graph(model.graphIndex(/*phase=*/0, /*shard=*/0, /*cl_idx=*/0));
    EXPECT_EQ(geniex::kv::formatOf(g0.inputSpec("past_key_0_in")), geniex::kv::KVFormat::Flat);
    EXPECT_EQ(geniex::kv::formatOf(g0.inputSpec("swa_key_0_in")), geniex::kv::KVFormat::HmxTiled);
    EXPECT_FALSE(model.native_kv_);
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

// modelConfigFromDirectory: bundles are not consistent about naming their
// dialog config genie_config.json (the Qwen3 eaglet exports ship
// `<model>_eager.json`), and a multi-engine dialog (eaglet: target + draft)
// must resolve to the target engine only. This drives the fallback scan, the
// array-engine target selection, ctx-bins, the HTP-extensions override, and
// the embedding LUT discovery all at once.
TEST(LLMSpecLoader, ModelConfigFromDirectoryScansForNonDefaultGenieConfigName) {
    const auto      dir = std::filesystem::temp_directory_path() / "geniex_loader_scan_multi_engine";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "tokenizer.json") << "{}";
    // A large, non-JSON-config file with a .json extension that must be
    // skipped by size rather than parsed (would otherwise be a multi-MB parse).
    {
        std::ofstream big(dir / "huge_vocab.json", std::ios::binary);
        big << "[";
        big.seekp((1 << 20) + 16);
        big << "]";
    }
    // A small but malformed JSON file: candidate scanning must swallow the
    // parse failure and keep looking rather than propagating it.
    std::ofstream(dir / "not_json.json") << "{ this is not valid json";

    std::ofstream(dir / "target_ctx.bin") << "ctxbin";
    std::ofstream(dir / "custom_ext.json") << R"({"devices":[{"cores":[{}, {}]}]})";
    std::ofstream(dir / "embedding_weights.raw") << "embed";

    std::ofstream(dir / "model_eager.json") << R"({
        "dialog": {
            "engine": [
                {"role": "draft", "model": {"binary": {"ctx-bins": ["draft_ctx.bin"]}}},
                {"role": "target", "model": {"binary": {"ctx-bins": ["target_ctx.bin"]}},
                 "backend": {"extensions": "custom_ext.json"}}
            ],
            "embedding": {"lut-path": "embedding_weights.raw"}
        }
    })";

    const auto cfg = geniex::modelConfigFromDirectory(dir);

    ASSERT_EQ(cfg.model_paths.size(), 1u);
    EXPECT_EQ(cfg.model_paths[0], (dir / "target_ctx.bin").string());
    EXPECT_EQ(cfg.htp_config_path, (dir / "custom_ext.json").string());
    EXPECT_EQ(cfg.num_cores, 2u);
    ASSERT_TRUE(cfg.embedding_path.has_value());
    EXPECT_EQ(*cfg.embedding_path, (dir / "embedding_weights.raw").string());

    std::filesystem::remove_all(dir, ec);
}

// A single-engine dialog (engine is an OBJECT, not an array) is the common
// case; ctx-bins resolve directly off it with no target/draft selection.
TEST(LLMSpecLoader, ModelConfigFromDirectorySingleEngineObjectResolvesCtxBins) {
    const auto      dir = std::filesystem::temp_directory_path() / "geniex_loader_single_engine";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "tokenizer.json") << "{}";
    std::ofstream(dir / "shard_ctx.bin") << "ctxbin";
    std::ofstream(dir / "genie_config.json") << R"({
        "dialog": {
            "engine": {"model": {"binary": {"ctx-bins": ["shard_ctx.bin"]}}}
        }
    })";

    const auto cfg = geniex::modelConfigFromDirectory(dir);

    ASSERT_EQ(cfg.model_paths.size(), 1u);
    EXPECT_EQ(cfg.model_paths[0], (dir / "shard_ctx.bin").string());
    EXPECT_FALSE(cfg.embedding_path.has_value());

    std::filesystem::remove_all(dir, ec);
}

// ── Native (HMX-tiled) KV cache ──────────────────────────────────────────────
//
// Modeled on the real native-kv bundle (Llama-3.2-3B-Instruct-SSD w4a16): a
// SCATTER cache (kv_len == CL, cache_index input) tiled on both KV in/out. These
// exercise LLMModel's orchestration -- detection, clearing, write-back,
// cache_index, mask geometry -- not the tiled byte order itself (that's
// kv_layout_test.cpp's job); they read back through kv::detile and compare
// against flat reference data.

namespace {

using geniex::testing::NativeKVFixture;

struct NativeModelFixture {
    NoDecodePoolEnv  no_pool;
    NativeKVFixture  fx;
    TestableLLMModel model{NativeKVFixture::makeSpec()};

    NativeModelFixture() { EXPECT_TRUE(model.initFromFixture(fx)); }
};

// The KV input buffer of the (phase, cl) graph, de-tiled to `n_tok` tokens.
std::vector<uint8_t> detiledKV(TestableLLMModel& m, size_t phase, const std::string& name, bool is_key, size_t n_tok) {
    geniex::Graph&       g   = m.graph(m.graphIndex(phase, 0, 0));
    const auto           geo = geniex::kv::geometryOf(g.inputSpec(name), is_key);
    std::vector<uint8_t> out(geo.n_heads * geo.head_dim * n_tok, 0);
    geniex::kv::detile(geo, static_cast<const uint8_t*>(g.inputPtr(name)), out.data(), n_tok);
    return out;
}

// De-tiles `n_tok` tokens out of a graph's own KV OUTPUT buffer, read back AFTER
// generate(): the stub's index-based identity copy (stub_qnnapi.cpp) overwrites
// KV outputs with unrelated input bytes, so a pre-generate snapshot wouldn't
// exercise updateKV/copyKV at all.
std::vector<uint8_t> detiledFrom(const geniex::kv::KVGeometry& geo, const void* ptr, size_t n_tok) {
    std::vector<uint8_t> out(geo.n_heads * geo.head_dim * n_tok, 0);
    geniex::kv::detile(geo, static_cast<const uint8_t*>(ptr), out.data(), n_tok);
    return out;
}

}  // namespace

// A tiled bundle is detected, and the fresh buffers are cleared to 0x00 rather
// than the ufixed8 midpoint 0x80 -- HMX applies no zero-point offset. The real
// bundle's KV OUTPUTS are tiled too, not just its inputs.
TEST(NativeKV, DetectedAndClearedToZero) {
    NativeModelFixture nf;
    const auto&        pairs = nf.model.requireKVStateBlock().shard_pairs[0];
    ASSERT_FALSE(pairs.empty());

    geniex::Graph& g = nf.model.graph(nf.model.graphIndex(0, 0, 0));
    EXPECT_EQ(geniex::kv::formatOf(g.inputSpec(pairs[0].key_in)), geniex::kv::KVFormat::HmxTiled);
    EXPECT_EQ(geniex::kv::formatOf(g.outputSpec(pairs[0].key_out)), geniex::kv::KVFormat::HmxTiled);
    EXPECT_TRUE(nf.model.kvScatter());

    nf.model.resetKVCache();
    const auto*  buf   = static_cast<const uint8_t*>(g.inputPtr(pairs[0].key_in));
    const size_t bytes = g.inputSpec(pairs[0].key_in).byteCount();
    for (size_t i = 0; i < bytes; ++i) ASSERT_EQ(buf[i], 0x00) << "byte " << i;
}

// A scatter cache's kv_len is the full context length at EVERY phase -- there
// is no reserved tail, and no restride between prefill and decode strides.
TEST(NativeKV, PhaseCapacitiesAreTheFullContextLength) {
    NativeModelFixture nf;
    EXPECT_EQ(nf.model.kvLen(0, 0), NativeKVFixture::kContextLen);
    EXPECT_EQ(nf.model.kvLen(1, 0), NativeKVFixture::kContextLen);
}

// Prefill writes the tiled KV output into the tiled cache at the scatter write
// cursor (n_past == 0 for the first chunk), with no rebase since both sides are
// tiled. Reference is read from key_out AFTER generate(), per detiledFrom's
// comment.
TEST(NativeKV, PrefillWriteBackLandsAtTheScatterCursor) {
    NativeModelFixture nf;
    nf.model.resetKVCache();

    const auto& pairs = nf.model.requireKVStateBlock().shard_pairs[0];
    ASSERT_FALSE(pairs.empty());

    geniex::Graph& pg      = nf.model.graph(nf.model.graphIndex(0, 0, 0));
    const auto     out_geo = geniex::kv::geometryOf(pg.outputSpec(pairs[0].key_out), /*is_key=*/true);
    ASSERT_EQ(out_geo.format, geniex::kv::KVFormat::HmxTiled);

    const size_t         n_tok = 3;
    std::vector<int32_t> prompt(n_tok, 1);
    nf.model.generate(prompt, greedyConfig(1));
    ASSERT_GE(nf.model.nPast(), n_tok);

    const int rebase = geniex::kv::deriveRebase(pg.inputSpec(pairs[0].key_in), pg.outputSpec(pairs[0].key_out));
    ASSERT_EQ(rebase, 0) << "a tiled cache fed by a tiled output needs no rebase";

    const auto expected = detiledFrom(out_geo, pg.outputPtr(pairs[0].key_out), n_tok);
    const auto got      = detiledKV(nf.model, 0, pairs[0].key_in, /*is_key=*/true, n_tok);
    ASSERT_EQ(got, expected);
}

// cache_index carries round32(n_past) for a native cache -- block-granular
// scatter-write, verified n_past=16 -> cache_index=32 on a real native-kv
// bundle.
TEST(NativeKV, CacheIndexTracksTheWriteCursor) {
    NativeModelFixture nf;
    nf.model.resetKVCache();

    const size_t         n_tok = 5;
    std::vector<int32_t> prompt(n_tok, 1);
    nf.model.generate(prompt, greedyConfig(1));

    // Check the DECODE graph -- prefill's cache_index (n_past=0) is now stale.
    geniex::Graph& dg  = nf.model.graph(nf.model.graphIndex(1, 0, 0));
    const auto*    idx = static_cast<const int32_t*>(dg.inputPtr("cache_index"));
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx[0], 32) << "round32(n_tok=5) == 32";
}

// Cached tokens must survive a second turn untouched: nothing about a scatter
// cache's write cursor should disturb earlier entries.
TEST(NativeKV, EarlierTokensSurviveASecondTurn) {
    NativeModelFixture nf;
    nf.model.resetKVCache();

    const auto& pairs = nf.model.requireKVStateBlock().shard_pairs[0];

    const size_t         n_tok = 5;
    std::vector<int32_t> prompt(n_tok, 1);
    nf.model.generate(prompt, greedyConfig(1));
    const auto before = detiledKV(nf.model, 0, pairs[0].value_in, /*is_key=*/false, n_tok);

    nf.model.generate(prompt, greedyConfig(1));
    const auto after = detiledKV(nf.model, 0, pairs[0].value_in, /*is_key=*/false, n_tok);
    EXPECT_EQ(before, after) << "tokens 0..n_tok must be untouched by the second turn";
}

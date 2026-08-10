// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/speculative_llm_model.cpp - the batched / tree
// decode + row-commit + stride-toggle primitives EAGLE drives on each engine.
// Runs a real SpeculativeLLMModel against a CPU-only graph fixture and the
// link-time QnnApi stub; no QNN device bring-up.

#include "llm/speculative_llm_model.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "QnnApi.hpp"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"
#include "testing/testable_llm_model.hpp"

namespace {

using geniex::testing::EagleTargetFixture;
using geniex::testing::ExternalLeadingShardFixture;
using geniex::testing::NoDecodePoolEnv;
using geniex::testing::TestableSpeculativeLLMModel;

// Builds an initialized speculative model over a fresh fixture, prefilled and
// switched to decode stride, ready to verify a decode batch. Holds all alive.
struct SpecFixture {
    NoDecodePoolEnv             no_pool;
    ExternalLeadingShardFixture fx;
    TestableSpeculativeLLMModel model{ExternalLeadingShardFixture::makeSpec()};

    SpecFixture() {
        EXPECT_TRUE(model.initFromFixture(fx));
        geniex::testing::stubSetVocabSize(ExternalLeadingShardFixture::kVocab);
    }
    ~SpecFixture() { geniex::testing::stubSetNextToken(-1); }

    void prefillAndDecodeStride(const std::vector<int32_t>& prompt) {
        model.prefill(prompt, /*rope_theta=*/1000000.0f, nullptr, 0, "");
        model.switchToDecodeStride();
    }
};

// decodeBatch runs a decode forward but leaves KV uncommitted; commitDecodeRows
// then advances n_past by exactly the accepted count regardless of how many
// rows were speculatively evaluated.
TEST(SpeculativeLLMModel, DecodeBatchCommitAdvancesByAcceptedCount) {
    SpecFixture sf;
    sf.prefillAndDecodeStride({1, 2, 3});
    ASSERT_EQ(sf.model.nPast(), 3u);

    const size_t         base = sf.model.nPast();
    std::vector<int32_t> tok  = {7};
    std::vector<int32_t> pos  = {static_cast<int32_t>(base)};
    auto                 res  = sf.model.decodeBatch(tok, pos, /*attention_map=*/{}, base, 1000000.0f, nullptr, 0, "");
    EXPECT_EQ(res.num_tokens, 1u);
    EXPECT_EQ(sf.model.nPast(), base);  // not advanced until commit

    sf.model.commitDecodeRows(std::vector<bool>{true}, /*n_accepted=*/1);
    EXPECT_EQ(sf.model.nPast(), base + 1);

    sf.model.switchToPrefillStride();  // round-trips the KV stride back
}

// A rejected row (selected=false) commits no KV, but n_past still advances by
// the caller-supplied accepted count -- the driver owns that bookkeeping.
TEST(SpeculativeLLMModel, CommitDecodeRowsHonorsAcceptedCount) {
    SpecFixture sf;
    sf.prefillAndDecodeStride({1, 2});

    const size_t         base = sf.model.nPast();
    std::vector<int32_t> tok  = {5};
    std::vector<int32_t> pos  = {static_cast<int32_t>(base)};
    sf.model.decodeBatch(tok, pos, /*attention_map=*/{}, base, 1000000.0f, nullptr, 0, "");

    // No row accepted; n_past unchanged.
    sf.model.commitDecodeRows(std::vector<bool>{false}, /*n_accepted=*/0);
    EXPECT_EQ(sf.model.nPast(), base);
}

// Without a decode pool, commitDecodeRowsAsync falls back to the synchronous
// commit and advances n_past inline.
TEST(SpeculativeLLMModel, CommitDecodeRowsAsyncFallsBackToSync) {
    SpecFixture sf;
    sf.prefillAndDecodeStride({4});

    const size_t         base = sf.model.nPast();
    std::vector<int32_t> tok  = {2};
    std::vector<int32_t> pos  = {static_cast<int32_t>(base)};
    sf.model.decodeBatch(tok, pos, /*attention_map=*/{}, base, 1000000.0f, nullptr, 0, "");

    sf.model.commitDecodeRowsAsync(std::vector<bool>{true}, /*n_accepted=*/1);
    sf.model.drainDecodePool();  // no-op without a pool
    EXPECT_EQ(sf.model.nPast(), base + 1);
}

// rewindKVCache shrinks n_past without touching KV buffers -- how tree building
// discards speculative scratch rows after a verify. Growing n_past is rejected.
TEST(SpeculativeLLMModel, RewindKVCacheShrinksNPastAndRejectsGrowth) {
    SpecFixture sf;
    sf.prefillAndDecodeStride({1, 2, 3, 4});
    const size_t base = sf.model.nPast();
    ASSERT_EQ(base, 4u);

    sf.model.rewindKVCache(base - 2);
    EXPECT_EQ(sf.model.nPast(), base - 2);

    EXPECT_THROW(sf.model.rewindKVCache(base), std::invalid_argument);
    EXPECT_EQ(sf.model.nPast(), base - 2);  // unchanged after the rejected grow
}

// With a real decode pool, commitDecodeRowsAsync offloads the KV copy to a
// worker but advances n_past synchronously; drainDecodePool orders the copy
// before the next read. Exercises the threadpool branch (not the sync fallback).
TEST(SpeculativeLLMModel, CommitDecodeRowsAsyncUsesDecodePool) {
    _putenv_s("GENIEX_DECODE_WORKERS", "1");
    ExternalLeadingShardFixture fx;
    TestableSpeculativeLLMModel model{ExternalLeadingShardFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));
    geniex::testing::stubSetVocabSize(ExternalLeadingShardFixture::kVocab);

    model.prefill({1, 2, 3}, 1000000.0f, nullptr, 0, "");
    model.switchToDecodeStride();
    const size_t         base = model.nPast();
    std::vector<int32_t> tok  = {7};
    std::vector<int32_t> pos  = {static_cast<int32_t>(base)};
    model.decodeBatch(tok, pos, /*attention_map=*/{}, base, 1000000.0f, nullptr, 0, "");

    model.commitDecodeRowsAsync(std::vector<bool>{true}, /*n_accepted=*/1);
    EXPECT_EQ(model.nPast(), base + 1);  // n_past advances up front
    model.drainDecodePool();             // orders the offloaded copy

    geniex::testing::stubSetNextToken(-1);
    _putenv_s("GENIEX_DECODE_WORKERS", "");
}

// decodeBatchTree runs the tree-mask decode path. With a single frontier node
// (the decode width here is 1) it behaves like decodeBatch but exercises
// buildTreeAttentionMask's n_keep / kv_ancestors handling.
TEST(SpeculativeLLMModel, DecodeBatchTreeSingleNode) {
    SpecFixture sf;
    sf.prefillAndDecodeStride({1, 2, 3});

    const size_t                      base         = sf.model.nPast();
    std::vector<int32_t>              tok          = {9};
    std::vector<int32_t>              pos          = {static_cast<int32_t>(base)};
    std::vector<std::vector<int32_t>> kv_ancestors = {{}};  // root: attends only committed prefix
    auto                              res          = sf.model.decodeBatchTree(tok,
        pos,
        /*attention_map=*/{},
        kv_ancestors,
        /*n_keep=*/base,
        base,
        1000000.0f,
        nullptr,
        0,
        "");
    EXPECT_EQ(res.num_tokens, 1u);
    EXPECT_EQ(sf.model.nPast(), base);  // tree decode never commits on its own

    sf.model.commitDecodeRows(std::vector<bool>{true}, 1);
    EXPECT_EQ(sf.model.nPast(), base + 1);
}

// switchToDecodeStride / switchToPrefillStride round-trip the KV layout without
// disturbing n_past.
TEST(SpeculativeLLMModel, StrideToggleRoundTripsPreservingNPast) {
    NoDecodePoolEnv             no_pool;
    ExternalLeadingShardFixture fx;
    TestableSpeculativeLLMModel model{ExternalLeadingShardFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx));
    geniex::testing::stubSetVocabSize(ExternalLeadingShardFixture::kVocab);

    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    model.prefill(prompt, 1000000.0f, nullptr, 0, "");
    const size_t n = model.nPast();

    model.switchToDecodeStride();
    EXPECT_EQ(model.nPast(), n);
    model.switchToPrefillStride();
    EXPECT_EQ(model.nPast(), n);

    geniex::testing::stubSetNextToken(-1);
}

// A batched decode fixture (verify width 4) built over EagleTargetFixture, ready
// to run multi-row decode batches with real intra-batch ancestor chains.
struct BatchedSpecFixture {
    NoDecodePoolEnv             no_pool;
    EagleTargetFixture          fx;
    TestableSpeculativeLLMModel model{EagleTargetFixture::makeSpec()};

    BatchedSpecFixture() {
        EXPECT_TRUE(model.initFromFixture(fx));
        geniex::testing::stubSetVocabSize(EagleTargetFixture::kVocab);
    }
    ~BatchedSpecFixture() { geniex::testing::stubSetNextToken(-1); }

    void prefillAndDecodeStride(const std::vector<int32_t>& prompt) {
        model.prefill(prompt, /*rope_theta=*/1000000.0f, nullptr, 0, "");
        model.switchToDecodeStride();
    }
};

// A multi-row decodeBatch with a non-empty attention_map exercises the causal
// chain: row i attends its ancestor chain (attention_map[i] -> ... -> root).
// Verifies the batch runs and commits advance n_past by the accepted count.
TEST(SpeculativeLLMModel, DecodeBatchChainAttentionMap) {
    BatchedSpecFixture sf;
    sf.prefillAndDecodeStride({1, 2, 3, 4});

    const size_t         base = sf.model.nPast();
    std::vector<int32_t> tok  = {5, 6, 7};  // linear chain of 3 rows
    std::vector<int32_t> pos  = {
        static_cast<int32_t>(base), static_cast<int32_t>(base + 1), static_cast<int32_t>(base + 2)};
    std::vector<int32_t> attn = {-1, 0, 1};  // row 1's parent is row 0, row 2's is row 1
    auto                 res  = sf.model.decodeBatch(tok, pos, attn, base, 1000000.0f, nullptr, 0, "");
    EXPECT_EQ(res.num_tokens, 3u);
    EXPECT_EQ(sf.model.nPast(), base);  // decode never commits on its own

    sf.model.commitDecodeRows(std::vector<bool>{true, true, true}, /*n_accepted=*/3);
    EXPECT_EQ(sf.model.nPast(), base + 3);

    sf.model.switchToPrefillStride();
}

// decodeBatch accepts a per-batch feature override written straight into the
// body's feature input, bypassing the embedding provider -- the path EAGLE uses
// to seed the draft with target hidden states.
TEST(SpeculativeLLMModel, DecodeBatchFeatureOverride) {
    BatchedSpecFixture sf;
    sf.prefillAndDecodeStride({1, 2, 3, 4});

    const size_t base = sf.model.nPast();
    // input_embeds is [ar, kHidden]; provide one row's worth of override bytes.
    std::vector<float>   feat = std::vector<float>(EagleTargetFixture::kHidden, 0.25f);
    std::vector<int32_t> tok  = {5};
    std::vector<int32_t> pos  = {static_cast<int32_t>(base)};
    auto                 res  = sf.model.decodeBatch(tok,
        pos,
        /*attention_map=*/{},
        base,
        1000000.0f,
        feat.data(),
        feat.size() * sizeof(float),
        "input_embeds");
    EXPECT_EQ(res.num_tokens, 1u);

    sf.model.commitDecodeRows(std::vector<bool>{true}, 1);
    EXPECT_EQ(sf.model.nPast(), base + 1);
    sf.model.switchToPrefillStride();
}

// decodeBatchTree over the batched fixture with sibling kv_ancestors and an
// intra-batch attention_map chain exercises buildTreeAttentionMask's per-row
// ancestor unwinding (both the committed-KV ancestor list and the causal chain).
TEST(SpeculativeLLMModel, DecodeBatchTreeChain) {
    BatchedSpecFixture sf;
    sf.prefillAndDecodeStride({1, 2, 3, 4});

    const size_t         base = sf.model.nPast();
    std::vector<int32_t> tok  = {5, 6, 7};
    std::vector<int32_t> pos  = {
        static_cast<int32_t>(base), static_cast<int32_t>(base + 1), static_cast<int32_t>(base + 2)};
    std::vector<int32_t>              attn         = {-1, 0, 1};
    std::vector<std::vector<int32_t>> kv_ancestors = {{}, {0}, {0, 1}};  // sibling KV rows to attend
    auto                              res =
        sf.model.decodeBatchTree(tok, pos, attn, kv_ancestors, /*n_keep=*/base, base, 1000000.0f, nullptr, 0, "");
    EXPECT_EQ(res.num_tokens, 3u);
    EXPECT_EQ(sf.model.nPast(), base);

    sf.model.commitDecodeRows(std::vector<bool>{true, false, true}, /*n_accepted=*/2);
    EXPECT_EQ(sf.model.nPast(), base + 2);
    sf.model.switchToPrefillStride();
}

}  // namespace

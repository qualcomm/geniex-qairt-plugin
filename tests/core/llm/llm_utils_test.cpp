// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/llm_utils.cpp: tensor-name classification, the
// RoPE embedding variants, causal attention masking, and tensor helpers. All
// pure CPU math/logic; no QNN runtime.

#include "llm/llm_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace geniex;

namespace {

// cos^2 + sin^2 == 1 for every element of a RoPE table (within tolerance).
template <typename Rope>
void expectUnitCircle(const Rope& rope, const std::vector<int32_t>& pos, double scale = 1.0) {
    const auto [cos, sin] = rope.forward(pos);
    ASSERT_EQ(cos.size(), sin.size());
    for (size_t k = 0; k < cos.size(); ++k) {
        EXPECT_NEAR(cos[k] * cos[k] + sin[k] * sin[k], scale * scale, 1e-9);
    }
}

}  // namespace

// ─── tensor classification ───────────────────────────────────────────────────

TEST(IsKVTensor, MatchesKeyValueWithInOutSuffix) {
    EXPECT_TRUE(isKVTensor("past_key_in"));
    EXPECT_TRUE(isKVTensor("past_value_out"));
    EXPECT_TRUE(isKVTensor("layer0_key_in"));
    EXPECT_FALSE(isKVTensor("past_key"));           // no _in/_out suffix
    EXPECT_FALSE(isKVTensor("attention_mask_in"));  // no key/value
    EXPECT_FALSE(isKVTensor(""));
}

TEST(IsSpecialTensor, NamedSetAndKV) {
    EXPECT_TRUE(isSpecialTensor("attention_mask"));
    EXPECT_TRUE(isSpecialTensor("position_ids_cos"));
    EXPECT_TRUE(isSpecialTensor("past_key_in"));  // via isKVTensor
    EXPECT_FALSE(isSpecialTensor("hidden_states"));
    EXPECT_FALSE(isSpecialTensor("logits"));
}

// ─── RoPE variants ───────────────────────────────────────────────────────────

TEST(RotaryEmbedding, ShapeAndPositionZero) {
    RotaryEmbedding rope(/*head_dim=*/8, /*theta=*/10000.0f);
    EXPECT_EQ(rope.halfDim(), 4u);

    const auto [cos, sin] = rope.forward({0, 1, 2});
    ASSERT_EQ(cos.size(), 3u * 4u);
    // Position 0 -> cos 1, sin 0.
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(cos[i], 1.0, 1e-12);
        EXPECT_NEAR(sin[i], 0.0, 1e-12);
    }
    expectUnitCircle(rope, {0, 1, 2, 7});
}

TEST(LongRoPEEmbedding, ScalingAboveOriginalContext) {
    // max > original -> scaling_factor > 1, so cos^2+sin^2 == scaling_factor^2.
    LongRoPEEmbedding rope(/*head_dim=*/8,
        /*theta=*/10000.0f,
        /*ext_factors=*/{1.0f, 1.0f, 1.0f, 1.0f},
        /*max=*/8192,
        /*original=*/2048);
    EXPECT_EQ(rope.halfDim(), 4u);
    const double scale_ratio = 8192.0 / 2048.0;
    const double scaling     = std::sqrt(1.0 + std::log(scale_ratio) / std::log(2048.0));
    expectUnitCircle(rope, {0, 5}, scaling);
}

TEST(LongRoPEEmbedding, NoScalingAtOrBelowOriginal) {
    LongRoPEEmbedding rope(8, 10000.0f, {1, 1, 1, 1}, /*max=*/2048, /*original=*/2048);
    expectUnitCircle(rope, {0, 3}, /*scale=*/1.0);  // scale_ratio <= 1 -> factor 1
}

TEST(Llama3RoPEEmbedding, ShapeAndUnitCircle) {
    Llama3RoPEEmbedding rope(/*head_dim=*/16,
        /*theta=*/500000.0f,
        /*factor=*/8.0f,
        /*low_freq_factor=*/1.0f,
        /*high_freq_factor=*/4.0f,
        /*original=*/8192);
    EXPECT_EQ(rope.halfDim(), 8u);
    expectUnitCircle(rope, {0, 1, 100});  // attention_factor is 1.0
}

TEST(PartialRoPEEmbedding, FractionAndScale) {
    // head_dim 16, fraction 0.5 -> rope_dim 8 -> rope_half_dim 4.
    PartialRoPEEmbedding rope(/*head_dim=*/16, /*theta=*/10000.0f, /*rope_fraction=*/0.5f, /*scale=*/2.0f);
    EXPECT_EQ(rope.halfDim(), 4u);
    const auto [cos, sin] = rope.forward({0});
    ASSERT_EQ(cos.size(), 4u);
    // Position 0 with scale 2 -> cos = 2, sin = 0.
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(cos[i], 2.0, 1e-12);
        EXPECT_NEAR(sin[i], 0.0, 1e-12);
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

TEST(GetPositionIds, SequentialFromNPast) {
    EXPECT_EQ(get_position_ids(10, 3), (std::vector<int32_t>{10, 11, 12}));
    EXPECT_TRUE(get_position_ids(5, 0).empty());
}

TEST(GetCosSin, MatchesRotaryEmbedding) {
    const std::vector<int32_t> pos = {0, 1, 2};
    const auto [cos, sin]          = get_cos_sin(pos, /*head_dim=*/8, /*theta=*/10000.0f);
    const auto [rcos, rsin]        = RotaryEmbedding(8, 10000.0f).forward(pos);
    EXPECT_EQ(cos, rcos);
    EXPECT_EQ(sin, rsin);
}

// Causal mask: 0 for visible past + causal triangle, -1e9 elsewhere.
TEST(GetAttentionMask, CausalWithPast) {
    const size_t n_past = 2, curr_len = 2, seq_len = 2, kv_len = 4;
    const auto   mask  = get_attention_mask(n_past, curr_len, seq_len, kv_len);
    const size_t total = kv_len + seq_len;  // 6
    ASSERT_EQ(mask.size(), seq_len * total);

    auto at = [&](size_t r, size_t c) { return mask[r * total + c]; };
    // Visible past columns [0, n_past) are 0.
    EXPECT_FLOAT_EQ(at(0, 0), 0.f);
    EXPECT_FLOAT_EQ(at(0, 1), 0.f);
    // Past beyond n_past is masked.
    EXPECT_FLOAT_EQ(at(0, 2), -1e9f);
    EXPECT_FLOAT_EQ(at(0, 3), -1e9f);
    // Causal triangle in the current block (offset kv_len): row 0 sees col 0 only.
    EXPECT_FLOAT_EQ(at(0, kv_len + 0), 0.f);
    EXPECT_FLOAT_EQ(at(0, kv_len + 1), -1e9f);
    // Row 1 sees cols 0 and 1.
    EXPECT_FLOAT_EQ(at(1, kv_len + 0), 0.f);
    EXPECT_FLOAT_EQ(at(1, kv_len + 1), 0.f);
}

// Sliding-window mask: like the causal mask but additionally band-limited so a
// query at absolute position q only attends keys k with q - window < k <= q.
TEST(GetSlidingWindowMask, BandLimitsPastAndCurrent) {
    // window=2, n_past=3 cached keys, a 2-token current chunk over kv_len=4.
    const size_t n_past = 3, curr_len = 2, seq_len = 2, kv_len = 4, window = 2;
    const auto   mask  = get_sliding_window_mask(n_past, curr_len, seq_len, kv_len, window);
    const size_t total = kv_len + seq_len;  // 6
    ASSERT_EQ(mask.size(), seq_len * total);

    auto visible = [&](size_t r, size_t c) { return mask[r * total + c] == 0.f; };
    // Cached keys occupy cols [0, min(n_past,kv_len)) = [0,3) at abs positions
    // 0,1,2. Row 0 is query abs pos 3, window 2 -> attends k with q-window < k,
    // i.e. k_pos > 1: only cached k_pos 2 survives; k_pos 0,1 are band-masked.
    EXPECT_FALSE(visible(0, 0));  // k_pos 0: 0+2 > 3 false -> masked
    EXPECT_FALSE(visible(0, 1));  // k_pos 1: 1+2 > 3 false -> masked
    EXPECT_TRUE(visible(0, 2));   // k_pos 2: 2+2 > 3 true  -> visible
    // Current-chunk key col kv_len+0 is k_pos 3 (== q) -> visible; the causal
    // upper triangle keeps kv_len+1 (future) masked for row 0.
    EXPECT_TRUE(visible(0, kv_len + 0));
    EXPECT_FALSE(visible(0, kv_len + 1));
    // Row 1 is query abs pos 4 -> attends k_pos > 2. Cached k_pos 2 (2+2>4 is
    // false) drops out; the current chunk (k_pos 3,4) stays visible.
    EXPECT_FALSE(visible(1, 2));
    EXPECT_TRUE(visible(1, kv_len + 0));
    EXPECT_TRUE(visible(1, kv_len + 1));
}

TEST(TokensToEmbedding, LooksUpRows) {
    const size_t             hidden = 2;
    const std::vector<float> table  = {0, 0, 1, 1, 2, 2, 3, 3};  // 4 rows
    const auto               out    = tokensToEmbedding({3, 1}, table.data(), hidden);
    EXPECT_EQ(out, (std::vector<float>{3, 3, 1, 1}));
}

TEST(GetKVCache, ZeroFilledSize) {
    const auto kv = get_kv_cache(/*num_kv_heads=*/2, /*head_dim=*/3, /*kv_len=*/4);
    EXPECT_EQ(kv.size(), 2u * 3u * 4u);
    for (float v : kv) EXPECT_EQ(v, 0.f);
}

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// KeyDiff (https://arxiv.org/abs/2504.15364) token scoring/selection: pure functions, no
// QNN/graph dependency.

#include "llm/kv_eviction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace geniex::kv;

namespace {

// Flattens a list of head_dim-sized vectors into the row-major [n, head_dim] layout
// keyDiffScores expects.
std::vector<float> flatten(const std::vector<std::vector<float>>& rows) {
    std::vector<float> out;
    for (const auto& r : rows) out.insert(out.end(), r.begin(), r.end());
    return out;
}

}  // namespace

TEST(KeyDiffScores, OutlierScoresHigherThanNearDuplicates) {
    // Four near-identical vectors plus one clear outlier. The outlier should score
    // meaningfully higher (more distinctive) than any of the near-duplicates.
    std::vector<std::vector<float>> keys = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.01f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.01f, 0.0f},
        {1.0f, -0.01f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},  // orthogonal outlier
    };
    const auto flat   = flatten(keys);
    const auto scores = keyDiffScores(flat.data(), keys.size(), 4);

    ASSERT_EQ(scores.size(), keys.size());
    const float outlier_score = scores[4];
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GT(outlier_score, scores[i]) << "near-duplicate index " << i;
    }
}

TEST(KeyDiffScores, IdenticalKeysScoreZero) {
    std::vector<std::vector<float>> keys(5, std::vector<float>{1.0f, 2.0f, 3.0f});
    const auto                      flat   = flatten(keys);
    const auto                      scores = keyDiffScores(flat.data(), keys.size(), 3);

    for (float s : scores) EXPECT_NEAR(s, 0.0f, 1e-5f);
}

TEST(KeyDiffScores, EmptyInputReturnsEmpty) { EXPECT_TRUE(keyDiffScores(nullptr, 0, 4).empty()); }

TEST(AggregateScores, AveragesAcrossHeads) {
    std::vector<std::vector<float>> per_head = {
        {1.0f, 2.0f, 3.0f},
        {3.0f, 2.0f, 1.0f},
    };
    const auto out = aggregateScores(per_head, 3);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], 2.0f, 1e-6f);
    EXPECT_NEAR(out[1], 2.0f, 1e-6f);
    EXPECT_NEAR(out[2], 2.0f, 1e-6f);
}

TEST(AggregateScores, EmptyPerHeadReturnsZeros) {
    const auto out = aggregateScores({}, 4);
    EXPECT_EQ(out, std::vector<float>(4, 0.0f));
}

TEST(SelectKeyDiffSurvivors, ReturnsExactTargetCountWhenUnconstrained) {
    // 10 tokens, scores favor evicting the low-scoring middle ones. No sink/recency
    // protection (n_keep=0, recent_window=0), target 6 survivors.
    std::vector<float> scores = {9, 1, 2, 8, 3, 7, 4, 6, 0, 5};
    const auto         keep   = selectKeyDiffSurvivors(scores, 10, /*n_keep=*/0, /*recent_window=*/0, /*target=*/6);

    ASSERT_EQ(keep.size(), 6u);
    // Ascending (chronological) order.
    for (size_t i = 1; i < keep.size(); ++i) EXPECT_LT(keep[i - 1], keep[i]);
    // The 6 highest-scoring indices are {0,3,5,7,9,6} -> sorted ascending {0,3,5,6,7,9}.
    EXPECT_EQ(keep, (std::vector<size_t>{0, 3, 5, 6, 7, 9}));
}

TEST(SelectKeyDiffSurvivors, ProtectsSinkPrefixAndRecentWindowEvenIfLowScoring) {
    // Index 0 (sink) and index 9 (most recent) score lowest, but must survive anyway.
    std::vector<float> scores = {-100, 5, 5, 5, 5, 5, 5, 5, 5, -100};
    const auto         keep   = selectKeyDiffSurvivors(scores, 10, /*n_keep=*/1, /*recent_window=*/1, /*target=*/4);

    ASSERT_TRUE(std::find(keep.begin(), keep.end(), 0u) != keep.end());
    ASSERT_TRUE(std::find(keep.begin(), keep.end(), 9u) != keep.end());
    EXPECT_EQ(keep.size(), 4u);
}

TEST(SelectKeyDiffSurvivors, ProtectionCanExceedTargetCount) {
    // n_keep + recent_window alone cover more than target_count; protection wins.
    std::vector<float> scores(10, 0.0f);
    const auto         keep = selectKeyDiffSurvivors(scores, 10, /*n_keep=*/4, /*recent_window=*/4, /*target=*/2);
    EXPECT_GE(keep.size(), 8u);
}

TEST(SelectKeyDiffSurvivors, TargetCountClampedToNValid) {
    std::vector<float> scores(5, 1.0f);
    const auto         keep = selectKeyDiffSurvivors(scores, 5, 0, 0, /*target=*/100);
    EXPECT_EQ(keep.size(), 5u);
}

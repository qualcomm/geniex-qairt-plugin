// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for qnn-api/src/ContextLengthLadder.hpp — the back-off policy used
// when no context-length set was pinned, its driver, and the retryable-error
// classification. Pure CPU; no QNN runtime or NPU involved.
//
// This is the only mechanized coverage of the policy: QnnApi.cpp is compiled into
// no test target, so the state reset a real retry performs is verified on device
// instead.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "ContextLengthLadder.hpp"

namespace {

using qnn_api::contextLengthLadder;
using qnn_api::driveContextLengthLadder;
using qnn_api::isRetryableContextCreateError;
using qnn_api::parseGraphContextLength;

using Rungs = std::vector<std::vector<size_t>>;

// The reference bundle: AI Hub qwen3_4b_instruct_2507 w4a16 ships five variants.
const std::vector<size_t> kReferenceSet{512, 1024, 2048, 3072, 4096};

// ── parseGraphContextLength ──────────────────────────────────────────────────

TEST(ParseGraphContextLength, ReadsTheFieldFromRealGraphNames) {
    EXPECT_EQ(parseGraphContextLength("prompt_ar128_cl4096_1_of_4"), 4096u);
    EXPECT_EQ(parseGraphContextLength("token_ar1_cl512_3_of_4"), 512u);
    EXPECT_EQ(parseGraphContextLength("prefill_ar4_cl16_1_of_1"), 16u);
}

TEST(ParseGraphContextLength, NamesWithoutTheFieldAreUnclassified) {
    // nullopt, not zero: the caller keeps such graphs rather than dropping them.
    EXPECT_FALSE(parseGraphContextLength("some_graph").has_value());
    EXPECT_FALSE(parseGraphContextLength("").has_value());
    EXPECT_FALSE(parseGraphContextLength(nullptr).has_value());
    // "_cl" present but not followed by digits.
    EXPECT_FALSE(parseGraphContextLength("model_cluster_ar1").has_value());
}

TEST(ParseGraphContextLength, AnUnrelatedClSubstringDoesNotMaskTheRealField) {
    // Regression: scanning only the first "_cl" occurrence found "_cluster", saw a
    // non-digit, and reported the graph as unclassified — silently hiding a variant
    // from the back-off ladder.
    EXPECT_EQ(parseGraphContextLength("foo_cluster_ar1_cl512_1_of_1"), 512u);
    EXPECT_EQ(parseGraphContextLength("_cl_cl_cl2048_1_of_1"), 2048u);
}

TEST(ParseGraphContextLength, TrailingFieldWithNoSeparatorStillParses) {
    EXPECT_EQ(parseGraphContextLength("prompt_ar128_cl4096"), 4096u);
}

// ── contextLengthLadder: exact expected output ───────────────────────────────

TEST(ContextLengthLadder, EmptyInputYieldsNoRungs) { EXPECT_EQ(contextLengthLadder({}), Rungs{}); }

TEST(ContextLengthLadder, SingleVariantYieldsOneRung) {
    // Nothing to drop, so there is no second attempt to make.
    EXPECT_EQ(contextLengthLadder({4096}), Rungs({{4096}}));
}

TEST(ContextLengthLadder, TwoVariantsDropTheSmallest) {
    // No middles exist, so the {min,max} rung would duplicate rung 0 and is skipped.
    EXPECT_EQ(contextLengthLadder({512, 4096}), Rungs({{512, 4096}, {4096}}));
}

TEST(ContextLengthLadder, ThreeVariantsDropMiddleThenSmallest) {
    EXPECT_EQ(contextLengthLadder({512, 2048, 4096}), Rungs({{512, 2048, 4096}, {512, 4096}, {4096}}));
}

TEST(ContextLengthLadder, ReferenceSetCollapsesStraightToTheExtremes) {
    // The measured outcome on the reference bundle: rung 1 is {512, 4096}, which
    // is both the fastest and the only set comfortably inside the I/O budget.
    EXPECT_EQ(contextLengthLadder(kReferenceSet), Rungs({{512, 1024, 2048, 3072, 4096}, {512, 4096}, {4096}}));
}

TEST(ContextLengthLadder, NonContiguousValuesAreNotAssumedToBePowersOfTwo) {
    // The policy indexes; it never does arithmetic on the values.
    EXPECT_EQ(contextLengthLadder({333, 777, 5000}), Rungs({{333, 777, 5000}, {333, 5000}, {5000}}));
}

TEST(ContextLengthLadder, InputIsSortedAndDeduplicated) {
    // Callers pass a raw scan result, whose order is whatever the graph metadata
    // happened to be in.
    EXPECT_EQ(contextLengthLadder({4096, 512, 2048, 512}), contextLengthLadder({512, 2048, 4096}));
}

// ── contextLengthLadder: invariants that must hold for every input ───────────

TEST(ContextLengthLadder, InvariantsHoldAcrossAllShapes) {
    const std::vector<std::vector<size_t>> inputs{
        {4096},
        {512, 4096},
        {512, 2048, 4096},
        kReferenceSet,
        {333, 777, 5000},
        {4096, 512, 2048, 512},
        {1, 2},
        {128, 256, 512, 1024},
    };

    for (const auto& input : inputs) {
        std::vector<size_t> normalized(input);
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        const size_t largest = normalized.back();

        const Rungs rungs = contextLengthLadder(input);
        ASSERT_FALSE(rungs.empty());

        // Rung 0 is "everything", so the first attempt reproduces today's behaviour.
        EXPECT_EQ(rungs.front(), normalized);
        // The last rung is minimal: only the capacity-defining variant is left.
        EXPECT_EQ(rungs.back(), std::vector<size_t>{largest});

        for (size_t i = 0; i < rungs.size(); i++) {
            // Ascending: LLMModel derives spec_.context_lengths through a std::set
            // and graphIndex() assumes ascending order.
            EXPECT_TRUE(std::is_sorted(rungs[i].begin(), rungs[i].end()));
            // Never invent a context length the bundle does not provide.
            for (size_t cl : rungs[i]) {
                EXPECT_NE(std::find(normalized.begin(), normalized.end(), cl), normalized.end());
            }
            // The largest is never dropped: it is the model's hard capacity.
            EXPECT_NE(std::find(rungs[i].begin(), rungs[i].end(), largest), rungs[i].end());
            // Strictly shrinking, so the ladder always terminates and never repeats
            // an attempt that already failed.
            if (i > 0) EXPECT_LT(rungs[i].size(), rungs[i - 1].size());
        }
    }
}

// ── driveContextLengthLadder ─────────────────────────────────────────────────

TEST(DriveContextLengthLadder, StopsAtTheFirstRungThatSucceeds) {
    const Rungs rungs = contextLengthLadder(kReferenceSet);
    Rungs       attempted;

    const auto landed = driveContextLengthLadder(rungs, [&](const std::vector<size_t>& rung, size_t) {
        attempted.push_back(rung);
        return rung.size() <= 2;  // the {512,4096} rung is the first to fit
    });

    ASSERT_TRUE(landed.has_value());
    EXPECT_EQ(*landed, 1u);
    // Exactly two attempts: it must not keep going after one succeeded.
    EXPECT_EQ(attempted, Rungs({{512, 1024, 2048, 3072, 4096}, {512, 4096}}));
}

TEST(DriveContextLengthLadder, SucceedingImmediatelyMakesOneAttempt) {
    // The common case: the bundle fits as shipped and the ladder is inert.
    size_t     calls = 0;
    const auto landed =
        driveContextLengthLadder(contextLengthLadder(kReferenceSet), [&](const std::vector<size_t>&, size_t) {
            calls++;
            return true;
        });

    ASSERT_TRUE(landed.has_value());
    EXPECT_EQ(*landed, 0u);
    EXPECT_EQ(calls, 1u);
}

TEST(DriveContextLengthLadder, ReportsFailureAfterExhaustingEveryRung) {
    const Rungs rungs = contextLengthLadder(kReferenceSet);
    size_t      calls = 0;

    const auto landed = driveContextLengthLadder(rungs, [&](const std::vector<size_t>&, size_t) {
        calls++;
        return false;
    });

    EXPECT_FALSE(landed.has_value());
    EXPECT_EQ(calls, rungs.size());
}

TEST(DriveContextLengthLadder, SingleRungLadderMakesExactlyOneAttempt) {
    // A single-CL bundle has nothing to back off to, so a failure must not be
    // retried — that is what keeps a genuine error from costing extra load time.
    size_t     calls  = 0;
    const auto landed = driveContextLengthLadder(contextLengthLadder({4096}), [&](const std::vector<size_t>&, size_t) {
        calls++;
        return false;
    });

    EXPECT_FALSE(landed.has_value());
    EXPECT_EQ(calls, 1u);
}

TEST(DriveContextLengthLadder, PassesTheRungIndexThrough) {
    std::vector<size_t> indices;
    driveContextLengthLadder(contextLengthLadder(kReferenceSet), [&](const std::vector<size_t>&, size_t i) {
        indices.push_back(i);
        return false;
    });
    EXPECT_EQ(indices, std::vector<size_t>({0, 1, 2}));
}

TEST(DriveContextLengthLadder, EmptyLadderMakesNoAttempt) {
    size_t     calls  = 0;
    const auto landed = driveContextLengthLadder({}, [&](const std::vector<size_t>&, size_t) {
        calls++;
        return true;
    });
    EXPECT_FALSE(landed.has_value());
    EXPECT_EQ(calls, 0u);
}

// ── isRetryableContextCreateError ────────────────────────────────────────────

TEST(IsRetryableContextCreateError, TerminalErrorsAreNotRetried) {
    // Wrong SDK, wrong SoC, or a rejected argument does not depend on how many
    // graphs are being deserialized, so a smaller set cannot help.
    EXPECT_FALSE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE));
    EXPECT_FALSE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_INVALID_ARGUMENT));
    EXPECT_FALSE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_INVALID_HANDLE));
    EXPECT_FALSE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_BINARY_VERSION));
    EXPECT_FALSE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_BINARY_CONFIGURATION));
    EXPECT_FALSE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_INVALID_CONFIG));
}

TEST(IsRetryableContextCreateError, ResourceAndUnknownErrorsAreRetried) {
    // The protection-domain overrun is expected to surface as one of these; the
    // exact code is undocumented, which is why the classification is a deny-list.
    EXPECT_TRUE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_CREATE_FROM_BINARY));
    EXPECT_TRUE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_MEM_ALLOC));
    EXPECT_TRUE(isRetryableContextCreateError(QNN_COMMON_ERROR_SYSTEM_COMMUNICATION));
    EXPECT_TRUE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_ABORTED));
    EXPECT_TRUE(isRetryableContextCreateError(QNN_CONTEXT_ERROR_TIMED_OUT));
    // An undocumented backend-specific code must not be treated as terminal.
    EXPECT_TRUE(isRetryableContextCreateError(0x7FFF0001u));
}

}  // namespace

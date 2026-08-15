// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/include/pipeline/stop_matcher.h: byte-level stop-sequence
// matching used by the LLM pipeline's generateTokens(). Pure string logic; no
// QNN runtime, no tokenizer.

#include "pipeline/stop_matcher.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

using namespace geniex;

namespace {

// Drives a StopMatcher the same way generateTokens() does: feed one decoded
// token at a time, emit the safe bytes, stop on a match, and flush the held
// tail when generation ends without one. Returns the emitted text, whether a
// stop matched, and — on a match — the offset at which it began.
struct RunResult {
    std::string emitted;
    bool        matched      = false;
    std::size_t match_offset = 0;
};

RunResult run(const std::vector<std::string>& stops, const std::vector<std::string>& tokens) {
    RunResult   r;
    StopMatcher m(stops);
    for (const auto& tok : tokens) {
        if (m.feed(tok)) {
            r.matched      = true;
            r.match_offset = m.matchOffset();
            break;
        }
        while (true) {
            const std::string safe = m.takeReady();
            if (safe.empty()) break;
            r.emitted += safe;
        }
    }
    while (true) {
        const std::string safe = m.takeReady();
        if (safe.empty()) break;
        r.emitted += safe;
    }
    if (!r.matched) {
        r.emitted += m.flush();
    }
    return r;
}

}  // namespace

TEST(StopMatcher, NoMatchEmitsEverything) {
    const auto r = run({"<|endoftext|>"}, {"hello", " world"});
    EXPECT_EQ(r.emitted, "hello world");
    EXPECT_FALSE(r.matched);
}

TEST(StopMatcher, MatchInsideOneToken) {
    const auto r = run({"<fim_middle>"}, {"a)<fim_middle>tail"});
    EXPECT_EQ(r.emitted, "a)");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 2);
}

TEST(StopMatcher, MatchSpanningTokens) {
    const auto r = run({"<fim_middle>"}, {"done<fim_", "middle>rest"});
    EXPECT_EQ(r.emitted, "done");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 4);
}

TEST(StopMatcher, MatchSpanningManyTokens) {
    const auto r = run({"abcdef"}, {"a", "b", "c", "d", "e", "f", "tail"});
    EXPECT_EQ(r.emitted, "");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 0);
}

TEST(StopMatcher, EarliestStopWins) {
    const auto r = run({"</code>", "<fim_"}, {"x<fim_</code>"});
    EXPECT_EQ(r.emitted, "x");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 1);
}

TEST(StopMatcher, StopAtStart) {
    const auto r = run({"\n\n"}, {"\n\n", "more"});
    EXPECT_EQ(r.emitted, "");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 0);
}

TEST(StopMatcher, EmptyStopsPassThrough) {
    EXPECT_FALSE(StopMatcher({""}).active());
    EXPECT_FALSE(StopMatcher({}).active());
    const auto r = run({""}, {"abc"});
    EXPECT_EQ(r.emitted, "abc");
    EXPECT_FALSE(r.matched);
}

TEST(StopMatcher, HeldTailFlushedAtEnd) {
    const auto r = run({"<|endoftext|>"}, {"end<|endo"});
    EXPECT_EQ(r.emitted, "end<|endo");
    EXPECT_FALSE(r.matched);
}

TEST(StopMatcher, LongTailBeforeStopHeldUntilSafe) {
    // "oops</stop" must not leak "<st" before the match completes.
    const auto r = run({"<stop>"}, {"oops<sto", "p>"});
    EXPECT_EQ(r.emitted, "oops");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 4);
}

TEST(StopMatcher, Utf8StopSplitAcrossTokenBoundary) {
    // "💩" is 4 bytes; the first token ends mid-sequence.
    const std::string stop = "\xF0\x9F\x92\xA9";
    const auto        r    = run({stop}, {"prefix\xF0\x9F", "\x92\xA9suffix"});
    EXPECT_EQ(r.emitted, "prefix");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 6);
}

TEST(StopMatcher, EmptyTokensDoNotAdvanceState) {
    const auto r = run({"xy"}, {"ab", "", "x", "y"});
    EXPECT_EQ(r.emitted, "ab");
    EXPECT_TRUE(r.matched);
    EXPECT_EQ(r.match_offset, 2);
}

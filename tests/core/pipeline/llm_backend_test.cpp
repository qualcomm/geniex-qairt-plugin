// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for pipeline/llm_backend.h: the optional-capability defaults every
// LLMBackend inherits (what a speculative backend like EAGLE relies on), and
// LLMModelBackend's pass-through to an LLMModel. No QNN device or graph bring-up.

#include "pipeline/llm_backend.h"

#include <gtest/gtest.h>

#include <memory>

namespace geniex {
namespace {

// Implements only the pure virtuals, so every optional capability is a default.
class MinimalBackend final : public LLMBackend {
   public:
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt, const GenerationConfig&, std::function<bool(int32_t)> on_token) override {
        for (int32_t t : prompt)
            if (on_token && !on_token(t)) break;
        return prompt;
    }
    void   reset() override { ++resets; }
    size_t nPast() const override { return 7; }
    size_t vocabSize() const override { return 32; }

    int resets = 0;
};

TEST(LLMBackendDefaults, NoInputEmbeddings) {
    MinimalBackend b;
    EXPECT_FALSE(b.supportsInputEmbeddings());
    EXPECT_EQ(b.hiddenSize(), 0u);
    EXPECT_THROW(b.generateFromEmbeddings({1.0f}, GenerationConfig{}, nullptr), UnsupportedOperationError);
}

TEST(LLMBackendDefaults, NoForwardLogitsOrKvCache) {
    MinimalBackend b;
    EXPECT_THROW(b.forwardLogits({1, 2}, false), UnsupportedOperationError);
    EXPECT_THROW(b.saveKVCache("x"), UnsupportedOperationError);
    EXPECT_THROW(b.loadKVCache("x"), UnsupportedOperationError);
}

TEST(LLMBackendDefaults, UnsupportedIsALogicError) {
    // Callers that catch std::logic_error (or std::exception) must still see it.
    MinimalBackend b;
    EXPECT_THROW(b.saveKVCache("x"), std::logic_error);
}

TEST(LLMBackendDefaults, PureVirtualsAreForwarded) {
    MinimalBackend             b;
    LLMBackend&                base = b;
    std::vector<int32_t>       seen;
    const std::vector<int32_t> prompt{4, 5, 6};
    const auto                 out = base.generate(prompt, GenerationConfig{}, [&](int32_t t) {
        seen.push_back(t);
        return seen.size() < 2;  // stop after the second token
    });
    EXPECT_EQ(out, prompt);
    EXPECT_EQ(seen.size(), 2u);
    base.reset();
    EXPECT_EQ(b.resets, 1);
    EXPECT_EQ(base.nPast(), 7u);
    EXPECT_EQ(base.vocabSize(), 32u);
}

TEST(LLMModelBackendTest, ForwardsToUninitializedModel) {
    // An LLMModel that was never initialize()d reports empty state; the adapter
    // must forward to it rather than answer on its own.
    LLMModelBackend b(std::make_unique<LLMModel>(LLMSpec{}));
    EXPECT_EQ(b.nPast(), 0u);
    EXPECT_EQ(b.vocabSize(), 0u);
    EXPECT_EQ(b.hiddenSize(), 0u);
    EXPECT_NO_THROW(b.reset());
    EXPECT_EQ(&b.model(), &b.model());
}

}  // namespace
}  // namespace geniex

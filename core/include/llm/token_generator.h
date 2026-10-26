// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"

namespace geniex {

// Thrown by a TokenGenerator for an operation its decoding strategy cannot offer
// (e.g. KV save/load on a two-engine speculative decoder), as opposed to an I/O or
// runtime failure of a supported one.
class UnsupportedOperationError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

// Interface for autoregressive generation over token ids.
class TokenGenerator {
   public:
    virtual ~TokenGenerator() = default;

    // Returns the generated token ids, excluding the prompt. token_callback is
    // called with each emitted token; return false to stop early.
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
        const GenerationConfig& gen_cfg = {}, std::function<bool(int32_t)> token_callback = nullptr) = 0;

    virtual void resetKVCache() = 0;

    // Writes the KV cache to `path` / restores it from there. Both throw
    // std::runtime_error on I/O failure.
    virtual void saveKVCacheToFile(const std::string& path) const = 0;
    virtual void loadKVCacheFromFile(const std::string& path)     = 0;

    // Logits from a single non-autoregressive forward pass over `tokens`: the
    // last position's row (vocabSize() floats), or every position row-major
    // [tokens.size(), vocabSize()] when all_positions is set. Resets the KV
    // cache before and after, so any retained conversation is discarded.
    virtual std::vector<float> forwardLogits(const std::vector<int32_t>& tokens, bool all_positions = false) = 0;

    // Number of positions held in the KV cache.
    virtual size_t nPast() const = 0;

    // Size of the output vocabulary; 0 before the generator is initialized.
    virtual size_t vocabSize() const = 0;
};

}  // namespace geniex

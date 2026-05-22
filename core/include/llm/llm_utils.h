// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "geniex_export.h"

namespace geniex {

// Standard RoPE: constructed once with (head_dim, theta), reused across forward calls.
class GENIEX_API RotaryEmbedding {
   public:
    RotaryEmbedding() = default;
    RotaryEmbedding(size_t head_dim, float theta = 10000.f);

    // Returns {cos, sin}, each flat [n * half_dim] where n = position_ids.size().
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<float> inv_freq_;  // [half_dim]
    size_t             half_dim_ = 0;
};

// LongRoPE with dynamic scaling and per-dimension extension factors.
// ext_factors shorter than half_dim are padded with 1.0.
class GENIEX_API LongRoPEEmbedding {
   public:
    LongRoPEEmbedding() = default;
    LongRoPEEmbedding(size_t head_dim, float theta, std::vector<float> ext_factors,
        int max_position_embeddings = 131072, int original_max_position_embeddings = 4096);

    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<float> ext_factors_;  // [half_dim], per-dimension extension factors
    float              base_                             = 10000.f;
    int                dim_                              = 0;  // full head_dim (half_dim_ * 2)
    int                max_position_embeddings_          = 131072;
    int                original_max_position_embeddings_ = 4096;
    size_t             half_dim_                         = 0;
};

// Partial RoPE: rotates only (rope_fraction * head_dim) dimensions, with a post-scale factor.
class GENIEX_API PartialRoPEEmbedding {
   public:
    PartialRoPEEmbedding() = default;
    PartialRoPEEmbedding(size_t head_dim, float theta = 10000.f, float rope_fraction = 1.0f, float scale = 1.0f);

    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<float> inv_freq_;  // [rope_half_dim]
    float              scale_         = 1.f;
    size_t             rope_half_dim_ = 0;
};

// Returns position IDs [n_past, n_past + count) as a flat int32 vector.
GENIEX_API std::vector<int32_t> get_position_ids(size_t n_past, size_t count);

// Returns {cos, sin}, each flat [n * half_dim] for the given position IDs.
GENIEX_API std::pair<std::vector<float>, std::vector<float>> get_cos_sin(
    const std::vector<int32_t>& position_ids, size_t head_dim, float rope_theta = 10000.f);

// Returns a causal attention mask, flat [seq_len * (kv_len + seq_len)].
// Columns [0, n_past) in all current-chunk rows are unmasked (0.0); everything
// else is -1e9 except the causal triangle in the current chunk.
GENIEX_API std::vector<float> get_attention_mask(size_t n_past, size_t curr_len, size_t seq_len, size_t kv_len);

// embedded_tokens: flat row-major [vocab_size * hidden_size] float32 table.
GENIEX_API std::vector<float> tokensToEmbedding(
    const std::vector<int32_t>& token_ids, const float* embedded_tokens, size_t hidden_size);

std::vector<float> get_kv_cache(size_t num_kv_heads, size_t head_dim, size_t kv_len);

// Tensor-name classification (shared by the spec loader and LLMModel).
//
// "Special" = a tensor that is NOT part of the inter-shard hidden-state
// stream — e.g. attention_mask, position_ids_*, KV-cache tensors. The two
// callers use this when picking the first non-special input/output of a
// graph as the shard's hidden-state wiring point.
GENIEX_API bool isKVTensor(const std::string& name);
GENIEX_API bool isSpecialTensor(const std::string& name);

}  // namespace geniex

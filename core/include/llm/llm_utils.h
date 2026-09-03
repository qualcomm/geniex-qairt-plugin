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
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<double> inv_freq_;  // [half_dim]
    size_t              half_dim_ = 0;
};

// LongRoPE with dynamic scaling and per-dimension extension factors.
// ext_factors shorter than half_dim are padded with 1.0.
class GENIEX_API LongRoPEEmbedding {
   public:
    LongRoPEEmbedding() = default;
    LongRoPEEmbedding(size_t head_dim, float theta, std::vector<float> ext_factors,
        int max_position_embeddings = 131072, int original_max_position_embeddings = 4096);

    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<double> ext_factors_;  // [half_dim], per-dimension extension factors
    double              base_                             = 10000.0;
    int                 dim_                              = 0;  // full head_dim (half_dim_ * 2)
    int                 max_position_embeddings_          = 131072;
    int                 original_max_position_embeddings_ = 4096;
    size_t              half_dim_                         = 0;
};

// Llama3 RoPE: frequency-dependent scaling of the base RoPE inv_freq.
// Low-frequency dimensions (long wavelength) are divided by `factor`,
// high-frequency dimensions are left unchanged, and a middle band is smoothly
// interpolated between the two. Ports the HuggingFace / Genie llama3 formula.
class GENIEX_API Llama3RoPEEmbedding {
   public:
    Llama3RoPEEmbedding() = default;
    Llama3RoPEEmbedding(size_t head_dim, float theta, float factor, float low_freq_factor, float high_freq_factor,
        int original_max_position_embeddings = 8192);

    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<double> inv_freq_;  // [half_dim], llama3-scaled
    size_t              half_dim_ = 0;
};

// Partial RoPE: rotates only (rope_fraction * head_dim) dimensions, with a post-scale factor.
//
// Two on-disk layouts exist for the same math:
//   - compact (full_width=false): the cos/sin tensor is exactly rope_dim/2 wide
//     (rope_dim = rope_fraction * head_dim), inv_freq[i] = theta^(-2i/rope_dim).
//   - full-width (full_width=true): the tensor is head_dim/2 wide (real Gemma3/4
//     rotate_half layout). The first rope_dim/2 entries are the real frequencies
//     computed against the FULL head_dim (inv_freq[i] = theta^(-2i/head_dim)) and
//     the remaining head_dim/2 - rope_dim/2 entries are zero (cos=1, sin=0 →
//     identity), so full-head rotate_half equals partial-rotary. Gemma4's
//     position_ids_global_cos/sin uses this layout (e.g. 256-wide = 64 real
//     freqs over head_dim 512, then 192 zeros).
class GENIEX_API PartialRoPEEmbedding {
   public:
    PartialRoPEEmbedding() = default;
    PartialRoPEEmbedding(size_t head_dim, float theta = 10000.f, float rope_fraction = 1.0f, float scale = 1.0f,
        bool full_width = false);

    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

   private:
    std::vector<double> inv_freq_;  // [out_half_dim_], zero-padded tail when full_width
    double              scale_        = 1.0;
    size_t              out_half_dim_ = 0;  // emitted cos/sin width per row
};

// Returns position IDs [n_past, n_past + count) as a flat int32 vector.
GENIEX_API std::vector<int32_t> get_position_ids(size_t n_past, size_t count);

// Returns {cos, sin}, each flat [n * half_dim] for the given position IDs.
GENIEX_API std::pair<std::vector<double>, std::vector<double>> get_cos_sin(
    const std::vector<int32_t>& position_ids, size_t head_dim, float rope_theta = 10000.f);

// Returns a causal attention mask, flat [seq_len * (kv_len + seq_len)].
// Columns [0, n_past) in all current-chunk rows are unmasked (0.0); everything
// else is -1e9 except the causal triangle in the current chunk.
// Sentinel for get_attention_mask's `new_base`: place this pass's fresh keys
// immediately after the cached ones (a concat cache).
inline constexpr size_t kNewBaseAfterCache = static_cast<size_t>(-1);

// Causal mask over a key axis `row_len` wide.
//
// `kv_len` is how many cached slots are visible; `new_base` is the column where
// this pass's fresh keys live. The defaults describe a CONCAT cache: axis
// kv_len + seq_len wide with the fresh block at kv_len. A SCATTER cache (the graph
// reads one CL-wide cache and drops the fresh keys into it at `cache_index`)
// passes row_len = CL and new_base = cache_index.
GENIEX_API std::vector<float> get_attention_mask(size_t n_past, size_t curr_len, size_t seq_len, size_t kv_len,
    size_t row_len = 0, size_t new_base = kNewBaseAfterCache);

// Sliding-window variant of get_attention_mask (Gemma3/4 local-attention
// layers). Same causal structure, but a query at absolute position p may only
// attend to keys with absolute position in (p - window, p]; older keys are
// masked (-1e9). Layout is identical: flat [seq_len * (kv_len + seq_len)].
GENIEX_API std::vector<float> get_sliding_window_mask(
    size_t n_past, size_t curr_len, size_t seq_len, size_t kv_len, size_t window);

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

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <vector>

#include "geniex_export.h"

// KeyDiff (https://arxiv.org/abs/2504.15364, NeurIPS 2025): training-free KV cache eviction
// that scores tokens by key-vector geometric distinctiveness instead of attention scores, so
// it stays compatible with fused attention kernels. These are the pure scoring/selection
// primitives; LLMModel::keyDiffEvict owns reading/dequantizing the KV buffers and re-prefilling
// the chosen survivors (QAIRT caches post-RoPE K/V, so an eviction that drops interior tokens
// must re-run the survivors to re-rotate their keys -- see LLMModel::reprefillKeep).
namespace geniex::kv {

// Per-token distinctiveness score for one attention head: 1 - cosine_similarity(k_i, mean(K)),
// using the paper's efficient O(n) variant (an unnormalized mean as the anchor vector -- "the
// normalized anchor can be swapped for an unnormalized mean without hurting accuracy"). `keys`
// is that head's key matrix, row-major [n_valid, head_dim]. Higher score == more geometrically
// distinctive == more worth keeping.
GENIEX_API std::vector<float> keyDiffScores(const float* keys, size_t n_valid, size_t head_dim);

// Averages one score vector per (layer, head) into a single per-token score. The paper scores
// and evicts independently per head/layer; this runtime shares one attention mask and position
// cursor across all of them, so a single eviction decision has to serve the whole cache.
GENIEX_API std::vector<float> aggregateScores(const std::vector<std::vector<float>>& per_head_scores, size_t n_valid);

// Chooses which of [0, n_valid) survive eviction, returned in ascending (chronological) order.
// [0, n_keep) and [n_valid - recent_window, n_valid) are always kept (clamped to n_valid, and
// may together exceed target_count -- protection takes priority over the budget). The remaining
// budget (up to target_count total survivors) is filled with the highest-scoring tokens outside
// those two ranges. target_count is clamped to n_valid.
GENIEX_API std::vector<size_t> selectKeyDiffSurvivors(
    const std::vector<float>& scores, size_t n_valid, size_t n_keep, size_t recent_window, size_t target_count);

}  // namespace geniex::kv

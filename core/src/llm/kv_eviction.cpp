// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/kv_eviction.h"

#include <algorithm>
#include <cmath>

namespace geniex::kv {

std::vector<float> keyDiffScores(const float* keys, size_t n_valid, size_t head_dim) {
    std::vector<float> scores(n_valid, 0.0f);
    if (n_valid == 0 || head_dim == 0) return scores;

    // Unnormalized mean anchor (the paper's efficient variant).
    std::vector<double> mean(head_dim, 0.0);
    for (size_t i = 0; i < n_valid; ++i) {
        const float* k = keys + i * head_dim;
        for (size_t d = 0; d < head_dim; ++d) mean[d] += k[d];
    }
    const double inv_n = 1.0 / static_cast<double>(n_valid);
    for (double& m : mean) m *= inv_n;

    double mean_norm_sq = 0.0;
    for (double m : mean) mean_norm_sq += m * m;
    const double mean_norm = std::sqrt(mean_norm_sq);

    for (size_t i = 0; i < n_valid; ++i) {
        const float* k   = keys + i * head_dim;
        double       dot = 0.0, k_norm_sq = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(k[d]) * mean[d];
            k_norm_sq += static_cast<double>(k[d]) * k[d];
        }
        const double k_norm  = std::sqrt(k_norm_sq);
        const double cos_sim = (mean_norm > 0.0 && k_norm > 0.0) ? dot / (mean_norm * k_norm) : 0.0;
        // Distinctiveness: a key far from the mean direction scores high and is kept; a
        // near-mean (redundant) key scores low and is the first evicted.
        scores[i] = static_cast<float>(1.0 - cos_sim);
    }
    return scores;
}

std::vector<float> aggregateScores(const std::vector<std::vector<float>>& per_head_scores, size_t n_valid) {
    std::vector<float> out(n_valid, 0.0f);
    if (per_head_scores.empty()) return out;

    for (const auto& s : per_head_scores) {
        for (size_t i = 0; i < n_valid; ++i) out[i] += s[i];
    }
    const float inv = 1.0f / static_cast<float>(per_head_scores.size());
    for (float& v : out) v *= inv;
    return out;
}

std::vector<size_t> selectKeyDiffSurvivors(
    const std::vector<float>& scores, size_t n_valid, size_t n_keep, size_t recent_window, size_t target_count) {
    target_count              = std::min(target_count, n_valid);
    n_keep                    = std::min(n_keep, n_valid);
    const size_t recent_begin = n_valid - std::min(recent_window, n_valid);

    std::vector<bool> kept(n_valid, false);
    size_t            kept_count = 0;
    for (size_t i = 0; i < n_keep; ++i) {
        kept[i] = true;
        ++kept_count;
    }
    for (size_t i = std::max(recent_begin, n_keep); i < n_valid; ++i) {
        kept[i] = true;
        ++kept_count;
    }

    // Fill the remaining budget with the highest-scoring unprotected tokens.
    std::vector<size_t> candidates;
    for (size_t i = n_keep; i < recent_begin; ++i) candidates.push_back(i);
    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });

    for (size_t idx : candidates) {
        if (kept_count >= target_count) break;
        kept[idx] = true;
        ++kept_count;
    }

    std::vector<size_t> survivors;
    survivors.reserve(kept_count);
    for (size_t i = 0; i < n_valid; ++i) {
        if (kept[i]) survivors.push_back(i);
    }
    return survivors;
}

}  // namespace geniex::kv

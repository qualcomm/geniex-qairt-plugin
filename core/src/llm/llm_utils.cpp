// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/llm_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace geniex {

bool isKVTensor(const std::string& s) {
    // Any tensor whose name contains "key" or "value" and ends with _in/_out.
    const bool has_inout_suffix = (s.size() >= 3 && s.compare(s.size() - 3, 3, "_in") == 0) ||
                                  (s.size() >= 4 && s.compare(s.size() - 4, 4, "_out") == 0);
    return has_inout_suffix && (s.find("key") != std::string::npos || s.find("value") != std::string::npos);
}

bool isSpecialTensor(const std::string& name) {
    // Named non-hidden-state tensors that currently appear in shipped bundles.
    // Add more as new bundles introduce them.
    static const std::unordered_set<std::string> kNamed = {
        "attention_mask",
        "position_ids",
        "position_ids_cos",
        "position_ids_sin",
    };
    return kNamed.count(name) > 0 || isKVTensor(name);
}

RotaryEmbedding::RotaryEmbedding(size_t head_dim, float theta) : half_dim_(head_dim / 2) {
    inv_freq_.resize(half_dim_);
    const double theta_d  = static_cast<double>(theta);
    const double exponent = 1.0 / static_cast<double>(head_dim);
    for (size_t i = 0; i < half_dim_; ++i) {
        inv_freq_[i] = 1.0 / std::pow(theta_d, static_cast<double>(i * 2) * exponent);
    }
}

std::pair<std::vector<double>, std::vector<double>> RotaryEmbedding::forward(
    const std::vector<int32_t>& position_ids) const {
    const size_t        n = position_ids.size();
    std::vector<double> cos_out(n * half_dim_);
    std::vector<double> sin_out(n * half_dim_);

    for (size_t t = 0; t < n; ++t) {
        const double pos = static_cast<double>(position_ids[t]);
        for (size_t i = 0; i < half_dim_; ++i) {
            const double freq          = pos * inv_freq_[i];
            cos_out[t * half_dim_ + i] = std::cos(freq);
            sin_out[t * half_dim_ + i] = std::sin(freq);
        }
    }
    return {cos_out, sin_out};
}

size_t RotaryEmbedding::halfDim() const { return half_dim_; }

LongRoPEEmbedding::LongRoPEEmbedding(size_t head_dim, float theta, std::vector<float> ext_factors,
    int max_position_embeddings, int original_max_position_embeddings)
    : base_(static_cast<double>(theta)),
      dim_(static_cast<int>(head_dim)),
      max_position_embeddings_(max_position_embeddings),
      original_max_position_embeddings_(original_max_position_embeddings),
      half_dim_(head_dim / 2) {
    // Promote per-dimension ext_factors to double; pad to half_dim_ with 1.0.
    ext_factors_.assign(ext_factors.begin(), ext_factors.end());
    ext_factors_.resize(half_dim_, 1.0);
}

std::pair<std::vector<double>, std::vector<double>> LongRoPEEmbedding::forward(
    const std::vector<int32_t>& position_ids) const {
    const size_t n = position_ids.size();

    // inv_freq[i] = 1.0 / (ext_factors[i] * base^(arange(0, dim, 2)[i] / dim))
    std::vector<double> inv_freq(half_dim_);
    const double        dim_d = static_cast<double>(dim_);
    for (size_t i = 0; i < half_dim_; ++i) {
        const double exponent   = static_cast<double>(i * 2) / dim_d;
        const double base_power = std::pow(base_, exponent);
        inv_freq[i]             = 1.0 / (ext_factors_[i] * base_power);
    }

    const double scale_ratio =
        static_cast<double>(max_position_embeddings_) / static_cast<double>(original_max_position_embeddings_);
    double scaling_factor;
    if (scale_ratio <= 1.0) {
        scaling_factor = 1.0;
    } else {
        scaling_factor =
            std::sqrt(1.0 + std::log(scale_ratio) / std::log(static_cast<double>(original_max_position_embeddings_)));
    }

    std::vector<double> cos_out(n * half_dim_);
    std::vector<double> sin_out(n * half_dim_);

    for (size_t t = 0; t < n; ++t) {
        const double pos = static_cast<double>(position_ids[t]);
        for (size_t i = 0; i < half_dim_; ++i) {
            const double freq          = pos * inv_freq[i];
            cos_out[t * half_dim_ + i] = std::cos(freq) * scaling_factor;
            sin_out[t * half_dim_ + i] = std::sin(freq) * scaling_factor;
        }
    }
    return {cos_out, sin_out};
}

size_t LongRoPEEmbedding::halfDim() const { return half_dim_; }

Llama3RoPEEmbedding::Llama3RoPEEmbedding(size_t head_dim, float theta, float factor, float low_freq_factor,
    float high_freq_factor, int original_max_position_embeddings)
    : half_dim_(head_dim / 2) {
    // Base inv_freq, then the HuggingFace / Genie llama3 frequency scaling.
    // half_dim_ here equals Genie's m_pos_dim (the cos/sin tensor's last dim),
    // and exponent = 1/half_dim matches Genie's 1/m_pos_dim; see
    // nsp-model.cpp:calculate_rope_embeddings.
    inv_freq_.resize(half_dim_);
    const double theta_d  = static_cast<double>(theta);
    const double exponent = 1.0 / static_cast<double>(half_dim_);
    for (size_t j = 0; j < half_dim_; ++j) {
        inv_freq_[j] = 1.0 / std::pow(theta_d, static_cast<double>(j) * exponent);
    }

    const double factor_d           = static_cast<double>(factor);
    const double low_freq_factor_d  = static_cast<double>(low_freq_factor);
    const double high_freq_factor_d = static_cast<double>(high_freq_factor);
    const double old_ctx            = static_cast<double>(original_max_position_embeddings);
    const double low_freq_wavelen   = old_ctx / low_freq_factor_d;
    const double high_freq_wavelen  = old_ctx / high_freq_factor_d;

    // Genie uses 2*M_PI; M_PI is the standard double π. Use the literal so we
    // don't depend on _USE_MATH_DEFINES being set before <cmath> (MSVC).
    constexpr double kPi = 3.14159265358979323846;
    for (size_t j = 0; j < half_dim_; ++j) {
        const double wavelen = 2.0 * kPi / inv_freq_[j];
        if (wavelen < high_freq_wavelen) {
            // high-frequency: leave unchanged
            continue;
        } else if (wavelen > low_freq_wavelen) {
            // low-frequency: divide by factor (recomputed as Genie does it)
            inv_freq_[j] = 1.0 / (factor_d * std::pow(theta_d, static_cast<double>(j) * exponent));
        } else {
            // mid band: smooth interpolation between the two
            const double smooth = (old_ctx / wavelen - low_freq_factor_d) / (high_freq_factor_d - low_freq_factor_d);
            inv_freq_[j]        = (1.0 - smooth) * inv_freq_[j] / factor_d + smooth * inv_freq_[j];
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> Llama3RoPEEmbedding::forward(
    const std::vector<int32_t>& position_ids) const {
    const size_t        n = position_ids.size();
    std::vector<double> cos_out(n * half_dim_);
    std::vector<double> sin_out(n * half_dim_);

    // attention_factor is 1.0 for the llama3 path (only ≠1 for longrope / yarn).
    for (size_t t = 0; t < n; ++t) {
        const double pos = static_cast<double>(position_ids[t]);
        for (size_t i = 0; i < half_dim_; ++i) {
            const double freq          = pos * inv_freq_[i];
            cos_out[t * half_dim_ + i] = std::cos(freq);
            sin_out[t * half_dim_ + i] = std::sin(freq);
        }
    }
    return {cos_out, sin_out};
}

size_t Llama3RoPEEmbedding::halfDim() const { return half_dim_; }

PartialRoPEEmbedding::PartialRoPEEmbedding(size_t head_dim, float theta, float rope_fraction, float scale)
    : scale_(static_cast<double>(scale)) {
    size_t rope_dim = static_cast<size_t>(head_dim * rope_fraction);
    rope_half_dim_  = rope_dim / 2;
    inv_freq_.resize(rope_half_dim_);
    const double theta_d  = static_cast<double>(theta);
    const double exponent = 1.0 / static_cast<double>(rope_dim);
    for (size_t i = 0; i < rope_half_dim_; ++i) {
        inv_freq_[i] = 1.0 / std::pow(theta_d, static_cast<double>(i * 2) * exponent);
    }
}

std::pair<std::vector<double>, std::vector<double>> PartialRoPEEmbedding::forward(
    const std::vector<int32_t>& position_ids) const {
    const size_t        n = position_ids.size();
    std::vector<double> cos_out(n * rope_half_dim_);
    std::vector<double> sin_out(n * rope_half_dim_);

    for (size_t t = 0; t < n; ++t) {
        const double pos = static_cast<double>(position_ids[t]);
        for (size_t i = 0; i < rope_half_dim_; ++i) {
            const double freq               = pos * inv_freq_[i];
            cos_out[t * rope_half_dim_ + i] = std::cos(freq) * scale_;
            sin_out[t * rope_half_dim_ + i] = std::sin(freq) * scale_;
        }
    }
    return {cos_out, sin_out};
}

size_t PartialRoPEEmbedding::halfDim() const { return rope_half_dim_; }

std::vector<int32_t> get_position_ids(size_t n_past, size_t count) {
    std::vector<int32_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = static_cast<int32_t>(n_past + i);
    return ids;
}

std::pair<std::vector<double>, std::vector<double>> get_cos_sin(
    const std::vector<int32_t>& position_ids, size_t head_dim, float rope_theta) {
    RotaryEmbedding rope(head_dim, rope_theta);
    return rope.forward(position_ids);
}

std::vector<float> get_attention_mask(size_t n_past, size_t curr_len, size_t seq_len, size_t kv_len) {
    const size_t       total_len = kv_len + seq_len;
    std::vector<float> mask(seq_len * total_len, -1e9f);

    for (size_t row = 0; row < curr_len; ++row) {
        float* row_ptr = mask.data() + row * total_len;

        const size_t visible_past = std::min(n_past, kv_len);
        for (size_t col = 0; col < visible_past; ++col) row_ptr[col] = 0.f;

        for (size_t col = 0; col <= row; ++col) row_ptr[kv_len + col] = 0.f;
    }

    return mask;
}

std::vector<float> tokensToEmbedding(
    const std::vector<int32_t>& token_ids, const float* embedded_tokens, size_t hidden_size) {
    const size_t       n = token_ids.size();
    std::vector<float> out(n * hidden_size);
    for (size_t i = 0; i < n; ++i) {
        const float* src = embedded_tokens + static_cast<size_t>(token_ids[i]) * hidden_size;
        float*       dst = out.data() + i * hidden_size;
        std::copy(src, src + hidden_size, dst);
    }
    return out;
}

std::vector<float> get_kv_cache(size_t num_kv_heads, size_t head_dim, size_t kv_len) {
    return std::vector<float>(num_kv_heads * head_dim * kv_len, 0.f);
}

}  // namespace geniex

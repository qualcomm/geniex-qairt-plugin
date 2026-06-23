// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm/vlm_input_provider.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>

#include "llm/llm_utils.h"
#include "xtensor/io/xnpy.hpp"

namespace geniex {

namespace {

bool endsWithICase(const std::string& path, const std::string& suffix) {
    if (path.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), path.rbegin(), [](char a, char b) {
        return std::tolower(a) == std::tolower(b);
    });
}

}  // namespace

PrecomputedEmbeddingProvider::PrecomputedEmbeddingProvider(std::string tensor_name)
    : tensor_name_(std::move(tensor_name)) {}

void PrecomputedEmbeddingProvider::loadTable(const std::string& path, size_t vocab_size, size_t hidden_size) {
    if (!table_.empty()) return;  // idempotent

    if (endsWithICase(path, ".npy")) {
        auto arr = xt::load_npy<float>(path);
        if (arr.dimension() != 2) {
            throw std::runtime_error("PrecomputedEmbeddingProvider: expected 2-D npy, got " +
                                     std::to_string(arr.dimension()) + "-D in " + path);
        }
        const size_t npy_vocab  = arr.shape(0);
        const size_t npy_hidden = arr.shape(1);
        if ((vocab_size && vocab_size != npy_vocab) || (hidden_size && hidden_size != npy_hidden)) {
            throw std::runtime_error("PrecomputedEmbeddingProvider: shape mismatch in " + path + " (npy is [" +
                                     std::to_string(npy_vocab) + "," + std::to_string(npy_hidden) + "], expected [" +
                                     std::to_string(vocab_size) + "," + std::to_string(hidden_size) + "])");
        }
        hidden_size_ = npy_hidden;
        table_.assign(arr.begin(), arr.end());
    } else {
        // Headerless raw float32: both dims must be provided by the caller.
        if (vocab_size == 0 || hidden_size == 0) {
            throw std::runtime_error("PrecomputedEmbeddingProvider: raw embedding file " + path +
                                     " requires non-zero (vocab_size, hidden_size)");
        }
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            throw std::runtime_error("PrecomputedEmbeddingProvider: cannot open " + path);
        }
        const std::streamsize expected = static_cast<std::streamsize>(vocab_size * hidden_size * sizeof(float));
        if (f.tellg() != expected) {
            throw std::runtime_error("PrecomputedEmbeddingProvider: size mismatch for " + path + " (expected " +
                                     std::to_string(expected) + " bytes, got " + std::to_string(f.tellg()) + ")");
        }
        f.seekg(0, std::ios::beg);
        table_.assign(vocab_size * hidden_size, 0.0f);
        f.read(reinterpret_cast<char*>(table_.data()), expected);
        if (!f) {
            throw std::runtime_error("PrecomputedEmbeddingProvider: failed to read " + path);
        }
        hidden_size_ = hidden_size;
    }
}

void PrecomputedEmbeddingProvider::onInitialized(const ModelConfig& model_cfg, const LLMSpec& spec) {
    if (table_.empty() && model_cfg.embedding_path) {
        loadTable(*model_cfg.embedding_path, spec.vocab_size, spec.hidden_size);
    }

    // Cache the EOS embedding to pad partial prefill chunks (matches Genie's
    // setupInputEmbeddings). Falls back to vocab[0] when eos isn't configured.
    if (pad_embed_.empty() && !table_.empty() && hidden_size_ > 0) {
        const int32_t pad_id = !spec.eos_token_ids.empty() ? spec.eos_token_ids.front() : 0;
        const size_t  vocab  = table_.size() / hidden_size_;
        if (pad_id >= 0 && static_cast<size_t>(pad_id) < vocab) {
            const float* src = table_.data() + static_cast<size_t>(pad_id) * hidden_size_;
            pad_embed_.assign(src, src + hidden_size_);
        }
    }
}

std::vector<float> PrecomputedEmbeddingProvider::lookupBatch(const std::vector<int32_t>& token_ids) const {
    return tokensToEmbedding(token_ids, table_.data(), hidden_size_);
}

void PrecomputedEmbeddingProvider::setBuffer(std::vector<float> embeds, size_t n_past_offset) {
    buffer_        = std::move(embeds);
    buffer_offset_ = n_past_offset;
}

void PrecomputedEmbeddingProvider::clearBuffer() {
    buffer_.clear();
    buffer_offset_ = 0;
}

void PrecomputedEmbeddingProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(tensor_name_)) return;

    const auto& spec     = g.inputSpec(tensor_name_);
    size_t      capacity = 1;
    for (auto d : spec.shape) capacity *= d;

    if (!buffer_.empty() && ctx.phase == 0) {
        const size_t local_offset = ctx.n_past - buffer_offset_;
        const size_t valid_count  = ctx.curr_len * hidden_size_;

        // Short prefill chunks must still fill the graph buffer; otherwise
        // the trailing rows hold stale bytes from a prior run.
        if (valid_count < capacity && !pad_embed_.empty()) {
            std::vector<float> buf(capacity);
            std::copy_n(buffer_.data() + local_offset * hidden_size_, valid_count, buf.data());
            for (size_t row = ctx.curr_len; row * hidden_size_ < capacity; ++row) {
                std::copy_n(pad_embed_.data(), hidden_size_, buf.data() + row * hidden_size_);
            }
            g.write(tensor_name_, buf.data(), capacity);
        } else {
            const float* src = buffer_.data() + local_offset * hidden_size_;
            g.write(tensor_name_, src, valid_count);
        }
    } else {
        if (table_.empty()) return;
        auto embeds = tokensToEmbedding(ctx.token_ids, table_.data(), hidden_size_);
        g.write(tensor_name_, embeds.data(), embeds.size());
    }
}

MRoPEInputProvider::MRoPEInputProvider(
    std::vector<int> mrope_section, float theta, MRoPEInterleaving style, std::string cos_name, std::string sin_name)
    : mrope_section_(std::move(mrope_section)),
      theta_(theta),
      style_(style),
      cos_name_(std::move(cos_name)),
      sin_name_(std::move(sin_name)),
      mrope_deltas_(3, 0) {
    half_dim_ = static_cast<size_t>(std::accumulate(mrope_section_.begin(), mrope_section_.end(), 0));

    inv_freq_.resize(half_dim_);
    const double theta_d = static_cast<double>(theta_);
    for (size_t i = 0; i < half_dim_; ++i) {
        inv_freq_[i] = 1.0 / std::pow(theta_d, static_cast<double>(i) / static_cast<double>(half_dim_));
    }
}

void MRoPEInputProvider::setPositionIds(
    const std::vector<int32_t>& position_ids, size_t seq_len, size_t n_past_offset) {
    fillCosSin(position_ids, seq_len, cos_table_, sin_table_);
    seq_len_               = seq_len;
    position_offset_       = n_past_offset;
    has_prefill_positions_ = true;
}

void MRoPEInputProvider::clearPositionIds() {
    has_prefill_positions_ = false;
    seq_len_               = 0;
    position_offset_       = 0;
    cos_table_.clear();
    sin_table_.clear();
}

const std::vector<int32_t>& MRoPEInputProvider::mropeDeltas() const { return mrope_deltas_; }

void MRoPEInputProvider::setMropeDeltas(std::vector<int32_t> deltas) { mrope_deltas_ = std::move(deltas); }

void MRoPEInputProvider::resetMropeDeltas() { mrope_deltas_.assign(3, 0); }

void MRoPEInputProvider::fillCosSin(const std::vector<int32_t>& position_ids, size_t seq_len,
    std::vector<double>& out_cos, std::vector<double>& out_sin) const {
    // freqs: flat [3 * seq_len * half_dim_], freqs[dim][t][i] = pos * inv_freq_[i]
    std::vector<double> freqs(3 * seq_len * half_dim_);
    for (int dim = 0; dim < 3; ++dim) {
        for (size_t t = 0; t < seq_len; ++t) {
            const double pos = static_cast<double>(position_ids[dim * seq_len + t]);
            double*      row = freqs.data() + (dim * seq_len + t) * half_dim_;
            for (size_t i = 0; i < half_dim_; ++i) {
                row[i] = pos * inv_freq_[i];
            }
        }
    }

    out_cos.resize(seq_len * half_dim_);
    out_sin.resize(seq_len * half_dim_);

    if (style_ == MRoPEInterleaving::BLOCK) {
        size_t off = 0;
        for (size_t dim = 0; dim < mrope_section_.size() && off < half_dim_; ++dim) {
            const size_t len = std::min<size_t>(mrope_section_[dim], half_dim_ - off);
            for (size_t t = 0; t < seq_len; ++t) {
                const double* src     = freqs.data() + (dim * seq_len + t) * half_dim_;
                double*       cos_row = out_cos.data() + t * half_dim_;
                double*       sin_row = out_sin.data() + t * half_dim_;
                for (size_t i = off; i < off + len; ++i) {
                    cos_row[i] = std::cos(src[i]);
                    sin_row[i] = std::sin(src[i]);
                }
            }
            off += len;
        }
    } else {
        // STRIDE: temporal fills all, then height/width override at stride-3 offsets.
        for (size_t t = 0; t < seq_len; ++t) {
            const double* src     = freqs.data() + t * half_dim_;  // dim=0
            double*       cos_row = out_cos.data() + t * half_dim_;
            double*       sin_row = out_sin.data() + t * half_dim_;
            for (size_t i = 0; i < half_dim_; ++i) {
                cos_row[i] = std::cos(src[i]);
                sin_row[i] = std::sin(src[i]);
            }
        }

        for (int dim = 1; dim < 3 && dim < static_cast<int>(mrope_section_.size()); ++dim) {
            const int offset = dim;
            const int length = mrope_section_[dim] * 3;
            for (int i = offset; i < length && i < static_cast<int>(half_dim_); i += 3) {
                for (size_t t = 0; t < seq_len; ++t) {
                    const double f             = freqs[(dim * seq_len + t) * half_dim_ + i];
                    out_cos[t * half_dim_ + i] = std::cos(f);
                    out_sin[t * half_dim_ + i] = std::sin(f);
                }
            }
        }
    }
}

void MRoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    // Pad cos/sin to graph capacity (cos=1, sin=0 = identity rotation) so
    // short prefill chunks don't leave stale RoPE values in trailing rows.
    auto write_padded = [&](Graph& gg, const std::string& name, const double* src, size_t valid_count) {
        const auto& spec     = gg.inputSpec(name);
        size_t      capacity = 1;
        for (auto d : spec.shape) capacity *= d;
        if (valid_count >= capacity) {
            gg.write(name, src, valid_count);
            return;
        }
        const bool          is_cos = (name == cos_name_);
        std::vector<double> buf(capacity, is_cos ? 1.0 : 0.0);
        std::copy_n(src, valid_count, buf.data());
        gg.write(name, buf.data(), capacity);
    };

    if (has_prefill_positions_ && ctx.phase == 0) {
        const size_t local_offset = (ctx.n_past - position_offset_) * half_dim_;
        const size_t count        = ctx.curr_len * half_dim_;
        if (has_cos) write_padded(g, cos_name_, cos_table_.data() + local_offset, count);
        if (has_sin) write_padded(g, sin_name_, sin_table_.data() + local_offset, count);
    } else {
        // Decode: sequential positions offset by accumulated mrope_deltas_.
        // mrope_deltas_ keeps decode tokens aligned with image/audio-expanded positions.
        const int32_t        base = static_cast<int32_t>(ctx.n_past);
        std::vector<int32_t> pos3d(3 * ctx.curr_len);
        for (size_t t = 0; t < ctx.curr_len; ++t) {
            for (int d = 0; d < 3; ++d) pos3d[d * ctx.curr_len + t] = base + static_cast<int32_t>(t) + mrope_deltas_[d];
        }

        std::vector<double> cos_buf, sin_buf;
        fillCosSin(pos3d, ctx.curr_len, cos_buf, sin_buf);
        const size_t count = ctx.curr_len * half_dim_;
        if (has_cos) write_padded(g, cos_name_, cos_buf.data(), count);
        if (has_sin) write_padded(g, sin_name_, sin_buf.data(), count);
    }
}

// ── DeepstackInputProvider ───────────────────────────────────────────────────

DeepstackInputProvider::DeepstackInputProvider(std::string embeds_name_prefix, std::string mask_name)
    : embeds_prefix_(std::move(embeds_name_prefix)), mask_name_(std::move(mask_name)) {}

void DeepstackInputProvider::setEmbeds(std::vector<std::vector<float>> levels, size_t hidden_size) {
    embeds_      = std::move(levels);
    hidden_size_ = hidden_size;
}

void DeepstackInputProvider::setVisualMask(std::vector<uint8_t> mask, size_t n_past_offset) {
    mask_        = std::move(mask);
    mask_offset_ = n_past_offset;
}

void DeepstackInputProvider::clear() {
    embeds_.clear();
    mask_.clear();
    hidden_size_ = 0;
    mask_offset_ = 0;
}

void DeepstackInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(mask_name_)) return;

    // Only prefill chunks of a round with a visual mask carry visual tokens;
    // decode steps, no-image turns, and post-image chunks write all-zeros.
    // Guard against ctx.n_past < mask_offset_ (stale offset / out-of-order call):
    // size_t subtraction would wrap and the mask_ indexing below would read OOB.
    const bool   is_visual_prefill = (ctx.phase == 0 && !mask_.empty() && ctx.n_past >= mask_offset_);
    const size_t start             = is_visual_prefill ? ctx.n_past - mask_offset_ : 0;

    // Visual-position mask for this chunk, sliced from the full mask and padded
    // to graph capacity. Always written (even all-zero): graph RPC buffers
    // persist across executes, so stale bytes would inject deepstack onto
    // ordinary tokens and corrupt generation (repetition/looping).
    const auto&          mask_spec = g.inputSpec(mask_name_);
    const size_t         mask_cap  = mask_spec.elementCount();
    std::vector<uint8_t> chunk(mask_cap, 0);
    if (is_visual_prefill) {
        for (size_t t = 0; t < ctx.curr_len && t < mask_cap; ++t) {
            const size_t idx = start + t;
            chunk[t]         = (idx < mask_.size() && mask_[idx]) ? 1 : 0;
        }
    }
    g.write(mask_name_, static_cast<const void*>(chunk.data()), mask_cap * mask_spec.elementSize());

    // Visual tokens before this chunk → first deepstack row it consumes. The
    // graph scatters consecutive rows onto the chunk's masked positions, so a
    // chunk must feed its aligned slice when an image straddles multiple chunks.
    size_t visual_offset = 0;
    for (size_t i = 0; i < start && i < mask_.size(); ++i) visual_offset += (mask_[i] ? 1 : 0);

    // Always overwrite every deepstack slot: zero-fill, then copy the aligned
    // source slice when one applies, so stale rows never leak into the scatter.
    for (size_t k = 0; k < embeds_.size(); ++k) {
        const std::string name = embeds_prefix_ + std::to_string(k);
        if (!g.hasInput(name)) continue;
        const size_t       capacity = g.inputSpec(name).elementCount();  // = rows * hidden_size_
        std::vector<float> buf(capacity, 0.0f);

        if (is_visual_prefill && hidden_size_ > 0) {
            const auto&  src       = embeds_[k];
            const size_t src_start = visual_offset * hidden_size_;
            if (src_start < src.size()) {
                std::copy_n(src.data() + src_start, std::min(capacity, src.size() - src_start), buf.data());
            }
        }
        g.write(name, buf.data(), capacity);
    }
}

}  // namespace geniex

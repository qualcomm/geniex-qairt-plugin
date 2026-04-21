#include "vlm/vlm_input_provider.h"
#include "llm/llm_utils.h"

#include "xtensor/io/xnpy.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace geniex {

// ── PrecomputedEmbeddingProvider ──────────────────────────────────────────────

PrecomputedEmbeddingProvider::PrecomputedEmbeddingProvider(std::string tensor_name)
    : tensor_name_(std::move(tensor_name)) {}

bool PrecomputedEmbeddingProvider::loadTable(const std::string& path,
                                              size_t             vocab_size,
                                              size_t             hidden_size) {
    auto arr = xt::load_npy<float>(path);
    if (arr.size() != vocab_size * hidden_size) {
        throw std::runtime_error(
            "PrecomputedEmbeddingProvider: table size mismatch in " + path);
    }
    table_.assign(arr.begin(), arr.end());
    hidden_size_ = hidden_size;
    return true;
}

void PrecomputedEmbeddingProvider::onInitialized(const ModelConfig& model_cfg) {
    if (!table_.empty()) return;
    if (model_cfg.embedding_path.empty()) return;

    auto arr = xt::load_npy<float>(model_cfg.embedding_path);
    if (arr.dimension() != 2) {
        throw std::runtime_error(
            "PrecomputedEmbeddingProvider: expected 2-D npy, got " +
            std::to_string(arr.dimension()) + "-D");
    }
    hidden_size_ = arr.shape(1);
    table_.assign(arr.begin(), arr.end());
}

std::vector<float> PrecomputedEmbeddingProvider::lookupBatch(
    const std::vector<int32_t>& token_ids) const {
    return tokensToEmbedding(token_ids, table_.data(), hidden_size_);
}

void PrecomputedEmbeddingProvider::setBuffer(std::vector<float> embeds, size_t n_past_offset) {
    buffer_ = std::move(embeds);
    buffer_offset_ = n_past_offset;
}

void PrecomputedEmbeddingProvider::clearBuffer() {
    buffer_.clear();
    buffer_offset_ = 0;
}

void PrecomputedEmbeddingProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(tensor_name_)) return;

    if (!buffer_.empty() && ctx.phase == 0) {
        // Prefill: slice from pre-computed buffer by token offset relative to buffer start.
        const size_t local_offset = ctx.n_past - buffer_offset_;
        const float* src = buffer_.data() + local_offset * hidden_size_;
        g.write(tensor_name_, src, ctx.curr_len * hidden_size_);
    } else {
        // Decode: per-token table lookup.
        if (table_.empty()) return;
        auto embeds = tokensToEmbedding(ctx.token_ids, table_.data(), hidden_size_);
        g.write(tensor_name_, embeds.data(), embeds.size());
    }
}

// ── MRoPEInputProvider ────────────────────────────────────────────────────────

MRoPEInputProvider::MRoPEInputProvider(std::vector<int>  mrope_section,
                                       float             theta,
                                       MRoPEInterleaving style,
                                       std::string       cos_name,
                                       std::string       sin_name)
    : mrope_section_(std::move(mrope_section))
    , theta_(theta)
    , style_(style)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name))
    , mrope_deltas_(3, 0)
{
    half_dim_ = static_cast<size_t>(
        std::accumulate(mrope_section_.begin(), mrope_section_.end(), 0));

    inv_freq_.resize(half_dim_);
    for (size_t i = 0; i < half_dim_; ++i) {
        inv_freq_[i] = 1.0f / std::pow(theta_, float(i) / float(half_dim_));
    }
}

void MRoPEInputProvider::setPositionIds(const std::vector<int32_t>& position_ids,
                                         size_t seq_len, size_t n_past_offset) {
    fillCosSin(position_ids, seq_len, cos_table_, sin_table_);
    seq_len_       = seq_len;
    position_offset_ = n_past_offset;
    has_prefill_positions_ = true;
}

void MRoPEInputProvider::clearPositionIds() {
    has_prefill_positions_ = false;
    seq_len_       = 0;
    position_offset_ = 0;
    cos_table_.clear();
    sin_table_.clear();
}

const std::vector<int32_t>& MRoPEInputProvider::mropeDeltas() const {
    return mrope_deltas_;
}

void MRoPEInputProvider::setMropeDeltas(std::vector<int32_t> deltas) {
    mrope_deltas_ = std::move(deltas);
}

void MRoPEInputProvider::resetMropeDeltas() {
    mrope_deltas_.assign(3, 0);
}

void MRoPEInputProvider::fillCosSin(const std::vector<int32_t>& position_ids,
                                     size_t                       seq_len,
                                     std::vector<float>&          out_cos,
                                     std::vector<float>&          out_sin) const {
    // freqs[dim][t][i] = position_ids[dim * seq_len + t] * inv_freq_[i]
    // Laid out as flat [3 * seq_len * half_dim_]
    std::vector<float> freqs(3 * seq_len * half_dim_);
    for (int dim = 0; dim < 3; ++dim) {
        for (size_t t = 0; t < seq_len; ++t) {
            const float pos = float(position_ids[dim * seq_len + t]);
            float* row = freqs.data() + (dim * seq_len + t) * half_dim_;
            for (size_t i = 0; i < half_dim_; ++i) {
                row[i] = pos * inv_freq_[i];
            }
        }
    }

    out_cos.resize(seq_len * half_dim_);
    out_sin.resize(seq_len * half_dim_);

    if (style_ == MRoPEInterleaving::BLOCK) {
        // Contiguous segments: [0:S0] from temporal, [S0:S0+S1] from height, etc.
        size_t off = 0;
        for (size_t dim = 0; dim < mrope_section_.size() && off < half_dim_; ++dim) {
            const size_t len = std::min<size_t>(mrope_section_[dim], half_dim_ - off);
            for (size_t t = 0; t < seq_len; ++t) {
                const float* src = freqs.data() + (dim * seq_len + t) * half_dim_;
                float* cos_row   = out_cos.data() + t * half_dim_;
                float* sin_row   = out_sin.data() + t * half_dim_;
                for (size_t i = off; i < off + len; ++i) {
                    cos_row[i] = std::cos(src[i]);
                    sin_row[i] = std::sin(src[i]);
                }
            }
            off += len;
        }
    } else {
        // STRIDE: temporal fills all, then height/width override at stride-3 offsets.

        // Step 1: fill all positions from temporal (dim=0)
        for (size_t t = 0; t < seq_len; ++t) {
            const float* src = freqs.data() + t * half_dim_;  // dim=0
            float* cos_row   = out_cos.data() + t * half_dim_;
            float* sin_row   = out_sin.data() + t * half_dim_;
            for (size_t i = 0; i < half_dim_; ++i) {
                cos_row[i] = std::cos(src[i]);
                sin_row[i] = std::sin(src[i]);
            }
        }

        // Step 2: override height (dim=1) and width (dim=2) at stride-3 offsets
        for (int dim = 1; dim < 3 && dim < static_cast<int>(mrope_section_.size()); ++dim) {
            const int offset = dim;
            const int length = mrope_section_[dim] * 3;
            for (int i = offset; i < length && i < static_cast<int>(half_dim_); i += 3) {
                for (size_t t = 0; t < seq_len; ++t) {
                    const float f = freqs[(dim * seq_len + t) * half_dim_ + i];
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

    if (has_prefill_positions_ && ctx.phase == 0) {
        // Prefill: slice from pre-computed tables for [n_past - offset, n_past - offset + curr_len)
        const size_t local_offset = (ctx.n_past - position_offset_) * half_dim_;
        const size_t count  = ctx.curr_len * half_dim_;
        if (has_cos) g.write(cos_name_, cos_table_.data() + local_offset, count);
        if (has_sin) g.write(sin_name_, sin_table_.data() + local_offset, count);
    } else {
        // Decode fallback: sequential positions offset by accumulated mrope_deltas.
        // mrope_deltas_ is synced from the model after each prefill, so decode tokens
        // continue from the correct position even after multimodal (image/audio) prefill.
        const int32_t base = static_cast<int32_t>(ctx.n_past);
        std::vector<int32_t> pos3d(3 * ctx.curr_len);
        for (size_t t = 0; t < ctx.curr_len; ++t) {
            for (int d = 0; d < 3; ++d)
                pos3d[d * ctx.curr_len + t] = base + static_cast<int32_t>(t) + mrope_deltas_[d];
        }

        std::vector<float> cos_buf, sin_buf;
        fillCosSin(pos3d, ctx.curr_len, cos_buf, sin_buf);
        const size_t count = ctx.curr_len * half_dim_;
        if (has_cos) g.write(cos_name_, cos_buf.data(), count);
        if (has_sin) g.write(sin_name_, sin_buf.data(), count);
    }
}

} // namespace geniex

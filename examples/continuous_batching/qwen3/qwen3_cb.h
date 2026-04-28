// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Continuous-batching adapter for Qwen3. Reuses the single-session spec
// from models/qwen3/qwen3.h — the only Qwen3-specific code here is the
// token-id writer and the RoPE cos/sin writer.

#include "cb/cb.h"
#include "llm/llm_utils.h"
#include "pipeline/chat_template.h"
#include "qwen3/qwen3.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace geniex {
namespace qwen3_cb {

static constexpr size_t kHeadDim   = 128;
static constexpr float  kRopeTheta = 1000000.0f;

// Writes the concatenated, zero-padded input_ids buffer into shard 0.
// No-ops on shards that don't take `input_ids`.
class Qwen3CBTokenIdProvider : public cb::CBInputProvider {
public:
    explicit Qwen3CBTokenIdProvider(std::string tensor_name = "input_ids",
                                    int32_t pad_token_id = 0)
        : tensor_name_(std::move(tensor_name)), pad_token_id_(pad_token_id) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        if (!g.hasInput(tensor_name_)) return;

        const auto& spec = g.inputSpec(tensor_name_);
        size_t capacity = 1;
        for (auto d : spec.shape) capacity *= d;

        std::vector<int32_t> buf(capacity, pad_token_id_);
        const size_t n = std::min(ctx.concat_tokens.size(), capacity);
        std::copy_n(ctx.concat_tokens.begin(), n, buf.begin());

        g.write(tensor_name_, buf.data(), buf.size());
    }

private:
    std::string tensor_name_;
    int32_t     pad_token_id_;
};

// Builds per-session position IDs and writes the RoPE cos/sin tables.
// Each session's positions start at its own kv_length rather than a single
// global n_past — the only semantic difference vs. RoPEInputProvider.
class Qwen3CBRoPEProvider : public cb::CBInputProvider {
public:
    Qwen3CBRoPEProvider(size_t head_dim, float theta,
                        std::string cos_name = "position_ids_cos",
                        std::string sin_name = "position_ids_sin")
        : rope_(head_dim, theta),
          cos_name_(std::move(cos_name)),
          sin_name_(std::move(sin_name)) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        const bool has_cos = g.hasInput(cos_name_);
        const bool has_sin = g.hasInput(sin_name_);
        if (!has_cos && !has_sin) return;

        std::vector<int32_t> pos_ids;
        cb::KVCacheManager::getPositionIds(ctx.kv_segs, ctx.in_segs, ctx.seq_len, pos_ids);

        auto [cos_vec, sin_vec] = rope_.forward(pos_ids);
        if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
        if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
    }

private:
    RotaryEmbedding rope_;
    std::string     cos_name_;
    std::string     sin_name_;
};

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

// Per-variant CB namespaces mirror the single-session namespaces in
// models/qwen3/qwen3.h 1:1. To add a new variant, add an inner namespace
// with a `makeModel()` that reuses the matching spec (and the right
// head_dim / theta, if they differ).

namespace qwen3_4b_instruct_2507 {
inline cb::CBLLMModel makeModel() {
    cb::CBLLMModel m(geniex::qwen3_4b_instruct_2507::makeSpec());
    m.addCBProvider(std::make_unique<Qwen3CBTokenIdProvider>("input_ids", 151645));
    m.addCBProvider(std::make_unique<Qwen3CBRoPEProvider>(kHeadDim, kRopeTheta));
    return m;
}
}  // namespace qwen3_4b_instruct_2507

}  // namespace qwen3_cb
}  // namespace geniex

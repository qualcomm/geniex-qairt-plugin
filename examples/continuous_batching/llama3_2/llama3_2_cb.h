// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Continuous-batching adapter for the Llama-3.2 family (1B / 3B / …).
// Reuses the single-session specs from models/llama3_2/llama3_2.h — the
// only Llama-3.2-specific code here is the token-id writer and the RoPE
// cos/sin writer. Llama 3.2 uses standard RoPE with no frequency scaling,
// so providers are structurally identical to Qwen3's; only head_dim and
// theta differ per size.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cb/cb.h"
#include "llama3/llama3.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_utils.h"

namespace geniex {
namespace llama3_2_cb {

// Pad with Llama-3 EOS ("end_of_text"); unused positions are masked out.
static constexpr int32_t kPadTokenId = 128001;

// Writes the concatenated, zero-padded input_ids buffer into shard 0.
// No-ops on shards that don't take `input_ids`.
class Llama32CBTokenIdProvider : public cb::CBInputProvider {
   public:
    explicit Llama32CBTokenIdProvider(std::string tensor_name = "input_ids", int32_t pad_token_id = kPadTokenId)
        : tensor_name_(std::move(tensor_name)), pad_token_id_(pad_token_id) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        if (!g.hasInput(tensor_name_)) return;

        const auto& spec     = g.inputSpec(tensor_name_);
        size_t      capacity = 1;
        for (auto d : spec.shape) capacity *= d;

        std::vector<int32_t> buf(capacity, pad_token_id_);
        const size_t         n = std::min(ctx.concat_tokens.size(), capacity);
        std::copy_n(ctx.concat_tokens.begin(), n, buf.begin());

        g.write(tensor_name_, buf.data(), buf.size());
    }

   private:
    std::string tensor_name_;
    int32_t     pad_token_id_;
};

// Builds per-session position IDs and writes the RoPE cos/sin tables.
class Llama32CBRoPEProvider : public cb::CBInputProvider {
   public:
    Llama32CBRoPEProvider(size_t head_dim, float theta, std::string cos_name = "position_ids_cos",
        std::string sin_name = "position_ids_sin")
        : rope_(head_dim, theta), cos_name_(std::move(cos_name)), sin_name_(std::move(sin_name)) {}

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

// Per-size CB namespaces mirror the single-session namespaces in
// models/llama3_2/llama3_2.h 1:1. Add a new size by adding an inner
// namespace with its own `makeModel()` that reuses the matching spec.

// Builds a CB-capable model from a bundle directory. Same shape across the
// 1B / 3B variants. head_dim for the CB RoPE provider is read from
// metadata.json (the CB provider is constructed up front); all other shapes
// are inferred from the graph tensors at initialize() time.
inline cb::CBLLMModel makeModel(const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       meta   = parseQAIRTMetadata(bundle);
    auto       gc     = parseGenieConfig(bundle);

    cb::CBLLMModel m(buildSpecSkeleton(gc), gc);
    m.addCBProvider(std::make_unique<Llama32CBTokenIdProvider>("input_ids", kPadTokenId));
    m.addCBProvider(std::make_unique<Llama32CBRoPEProvider>(meta.head_dim, gc.rope_theta));
    return m;
}

}  // namespace llama3_2_cb
}  // namespace geniex

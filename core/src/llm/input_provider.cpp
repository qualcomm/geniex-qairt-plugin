// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/input_provider.h"
#include "types.h"

#include "xtensor/io/xnpy.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace geniex {

namespace {

bool endsWithICase(const std::string& path, const std::string& suffix) {
    if (path.size() < suffix.size()) return false;
    return std::equal(
        suffix.rbegin(), suffix.rend(), path.rbegin(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

} // namespace

EmbeddingInputProvider::EmbeddingInputProvider(std::string tensor_name)
    : tensor_name_(std::move(tensor_name)) {}

void EmbeddingInputProvider::loadTable(const std::string& path,
                                       size_t             vocab_size,
                                       size_t             hidden_size) {
    if (!table_.empty()) return;          // idempotent

    if (endsWithICase(path, ".npy")) {
        // .npy carries its own shape — trust the header, validate hints if given.
        auto arr = xt::load_npy<float>(path);
        if (arr.dimension() != 2) {
            throw std::runtime_error(
                "EmbeddingInputProvider: expected 2-D npy, got " +
                std::to_string(arr.dimension()) + "-D in " + path);
        }
        const size_t npy_vocab  = arr.shape(0);
        const size_t npy_hidden = arr.shape(1);
        if ((vocab_size  && vocab_size  != npy_vocab) ||
            (hidden_size && hidden_size != npy_hidden)) {
            throw std::runtime_error(
                "EmbeddingInputProvider: shape mismatch in " + path +
                " (npy is [" + std::to_string(npy_vocab) + "," +
                std::to_string(npy_hidden) + "], expected [" +
                std::to_string(vocab_size) + "," +
                std::to_string(hidden_size) + "])");
        }
        hidden_size_ = npy_hidden;
        table_.assign(arr.begin(), arr.end());
    } else {
        // Headerless raw float32 — both dims must come from the caller.
        if (vocab_size == 0 || hidden_size == 0) {
            throw std::runtime_error(
                "EmbeddingInputProvider: raw embedding file " + path +
                " requires non-zero (vocab_size, hidden_size)");
        }
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            throw std::runtime_error(
                "EmbeddingInputProvider: cannot open " + path);
        }
        const std::streamsize expected = static_cast<std::streamsize>(
            vocab_size * hidden_size * sizeof(float));
        if (f.tellg() != expected) {
            throw std::runtime_error(
                "EmbeddingInputProvider: size mismatch for " + path +
                " (expected " + std::to_string(expected) + " bytes, got " +
                std::to_string(f.tellg()) + ")");
        }
        f.seekg(0, std::ios::beg);
        table_.assign(vocab_size * hidden_size, 0.0f);
        f.read(reinterpret_cast<char*>(table_.data()), expected);
        if (!f) {
            throw std::runtime_error(
                "EmbeddingInputProvider: failed to read " + path);
        }
        hidden_size_ = hidden_size;
    }
}

void EmbeddingInputProvider::onInitialized(const ModelConfig& model_cfg,
                                           const LLMSpec&     spec) {
    if (!table_.empty()) return;          // idempotent
    if (!model_cfg.embedding_path) return;

    loadTable(*model_cfg.embedding_path, spec.vocab_size, spec.hidden_size);
}

void EmbeddingInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(tensor_name_)) return;
    if (table_.empty()) return;
    auto embeds = tokensToEmbedding(ctx.token_ids, table_.data(), hidden_size_);
    g.write(tensor_name_, embeds.data(), embeds.size());
}

TokenIdInputProvider::TokenIdInputProvider(std::string tensor_name,
                                           int32_t     pad_token_id)
    : tensor_name_(std::move(tensor_name))
    , pad_token_id_(pad_token_id) {}

void TokenIdInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(tensor_name_)) return;

    const auto& spec = g.inputSpec(tensor_name_);
    size_t capacity = 1;
    for (auto d : spec.shape) capacity *= d;

    std::vector<int32_t> buf(capacity, pad_token_id_);
    const size_t n = std::min(ctx.token_ids.size(), capacity);
    std::copy_n(ctx.token_ids.begin(), n, buf.begin());

    g.write(tensor_name_, buf.data(), buf.size());
}

RoPEInputProvider::RoPEInputProvider(size_t head_dim, float theta,
                                     std::string cos_name,
                                     std::string sin_name)
    : rope_(head_dim, theta)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name)) {}

void RoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(ctx.n_past, ctx.curr_len));
    if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
    if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
}

LongRoPEInputProvider::LongRoPEInputProvider(size_t head_dim, float theta,
                                             std::vector<float> ext_factors,
                                             int max_position_embeddings,
                                             int original_max_position_embeddings,
                                             std::string cos_name,
                                             std::string sin_name)
    : rope_(head_dim, theta, std::move(ext_factors), max_position_embeddings, original_max_position_embeddings)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name)) {}

void LongRoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(ctx.n_past, ctx.curr_len));
    if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
    if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
}

PartialRoPEInputProvider::PartialRoPEInputProvider(size_t head_dim, float theta,
                                                   float rope_fraction, float scale,
                                                   std::string cos_name,
                                                   std::string sin_name)
    : rope_(head_dim, theta, rope_fraction, scale)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name)) {}

void PartialRoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(ctx.n_past, ctx.curr_len));
    if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
    if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
}

} // namespace geniex

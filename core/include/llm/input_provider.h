// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "geniex_export.h"
#include "graph.h"
#include "llm/llm_types.h"
#include "llm/llm_utils.h"
#include "types.h"

namespace geniex {

// Abstract interface for CPU-side tensor writes at inference time.
// Implementations write one or more named inputs into a Graph.
class GENIEX_API InputProvider {
   public:
    virtual ~InputProvider() = default;

    // Called once after all Graph objects are ready.
    // Implementations that need runtime configuration perform their one-time
    // setup here. They receive both the ModelConfig (paths, per-run config)
    // and the architectural LLMSpec (shapes, vocab size, etc.), so they do
    // not need to have these baked in at construction time.
    virtual void onInitialized(const ModelConfig&, const LLMSpec&) {}

    // Write input tensor(s) into g for the given context.
    // Implementations must silently do nothing if the target tensor is absent
    // on this shard (use Graph::hasInput() to check).
    virtual void write(Graph& g, const LLMRunContext& ctx) = 0;
};

// Owns a token embedding table and writes the [curr_len * hidden_size] embeds
// tensor for each forward pass.
//
// The embedding table is loaded automatically by onInitialized() from
// `model_cfg.embedding_path`, or explicitly by the caller via loadTable().
// Two on-disk formats are supported, auto-detected by file extension:
//   * `.npy` — numpy array [vocab_size, hidden_size] (float32); shape read
//              from the header. vocab_size / hidden_size arguments (if
//              non-zero) are validated against the header.
//   * other  — flat row-major float32 binary with no header. vocab_size and
//              hidden_size must be known either from the caller (loadTable)
//              or from the owning LLMSpec (onInitialized).
class GENIEX_API EmbeddingInputProvider : public InputProvider {
   public:
    // tensor_name: name of the graph input to write (default "input_embeds").
    explicit EmbeddingInputProvider(std::string tensor_name = "input_embeds");

    // Loads the embedding table from `path`.
    //  * `.npy`  — shape is read from the header; vocab_size/hidden_size are
    //              optional and, if non-zero, validated against it.
    //  * other   — flat row-major float32 binary; both vocab_size and
    //              hidden_size must be non-zero and match the file size.
    // Idempotent: does nothing if a table is already loaded.
    void loadTable(const std::string& path, size_t vocab_size = 0, size_t hidden_size = 0);

    // Loads the table from `model_cfg.embedding_path`. For headerless raw
    // files, shape is taken from `spec.vocab_size` / `spec.hidden_size`.
    // No-op if embedding_path is empty or the table is already loaded.
    void onInitialized(const ModelConfig& model_cfg, const LLMSpec& spec) override;

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    std::string        tensor_name_;
    std::vector<float> table_;      // flat row-major [vocab_size * hidden_size]
    std::vector<float> pad_embed_;  // flat [hidden_size]; pads short prefill chunks
    size_t             hidden_size_ = 0;
};

// For models where embedding lookup runs on-device (e.g. AI Hub exports).
// Pads to the graph's tensor capacity with a configurable pad token.
class GENIEX_API TokenIdInputProvider : public InputProvider {
   public:
    explicit TokenIdInputProvider(std::string tensor_name = "input_ids", int32_t pad_token_id = 0);

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    std::string tensor_name_;
    int32_t     pad_token_id_;
};

// Computes RoPE cos/sin tables and writes them into the two named graph inputs.
class GENIEX_API RoPEInputProvider : public InputProvider {
   public:
    // head_dim: per-head dimension (cos/sin vectors have size curr_len * head_dim/2).
    // theta:    RoPE base frequency.
    // cos_name / sin_name: names of the two graph inputs to write.
    RoPEInputProvider(size_t head_dim, float theta, std::string cos_name = "position_ids_cos",
        std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    RotaryEmbedding rope_;
    std::string     cos_name_;
    std::string     sin_name_;
};

// LongRoPE with dynamic scaling and per-dimension extension factors.
class GENIEX_API LongRoPEInputProvider : public InputProvider {
   public:
    LongRoPEInputProvider(size_t head_dim, float theta, std::vector<float> ext_factors,
        int max_position_embeddings = 131072, int original_max_position_embeddings = 4096,
        std::string cos_name = "position_ids_cos", std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    LongRoPEEmbedding rope_;
    std::string       cos_name_;
    std::string       sin_name_;
};

// Partial RoPE (rope_fraction of head_dim) with post-scale factor.
class GENIEX_API PartialRoPEInputProvider : public InputProvider {
   public:
    PartialRoPEInputProvider(size_t head_dim, float theta = 10000.f, float rope_fraction = 1.0f, float scale = 1.0f,
        std::string cos_name = "position_ids_cos", std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    PartialRoPEEmbedding rope_;
    std::string          cos_name_;
    std::string          sin_name_;
};

}  // namespace geniex

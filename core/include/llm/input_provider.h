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
#include "llm/quantized_lut.h"
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

    // Explicit-config variant for a table that is NOT model_cfg.embedding_path
    // and whose width differs from spec.hidden_size — e.g. Gemma3/4's per-layer
    // embedding stream (`per_layer_inputs`, width = num_layers * per_layer_dim).
    // onInitialized() loads `table_path` with this row width instead of the
    // main embedding path. `pad_token_override` (>=0) picks the pad-embedding
    // row; <0 falls back to the spec's first EOS as usual.
    EmbeddingInputProvider(
        std::string tensor_name, std::string table_path, size_t row_hidden_size, int32_t pad_token_override = -1);

    // Declares the on-disk table quantized. The table is then memory-mapped and
    // rows are converted per token instead of being dequantized into RAM (see
    // QuantizedLut). Must be called before onInitialized().
    //
    // When the target graph input carries the same dtype and (scale, offset) as
    // the table, write() skips conversion entirely and copies stored bytes
    // straight into the tensor -- both faster and bit-exact with Genie, which
    // applies the same short-circuit.
    void setQuantization(QuantizedLutSpec spec) { quant_ = std::move(spec); }

    // Substitutes externally computed embeddings for the table lookup at absolute
    // prompt positions [start, start + rows.size()/row_width).
    //
    // This is how a vision encoder's output reaches the decoder: the prompt
    // carries N image-token ids, and the encoder's N rows replace what the table
    // would have produced for them. Rows are plain float32 in model space.
    // Chunks that overlap the override take the dequantize/requantize path, so
    // the byte-copy fast path still applies to every purely-textual chunk.
    void setEmbeddingOverride(size_t start_position, std::vector<float> rows);
    void clearEmbeddingOverride();

    // Substitutes `token_id` for whatever ids sit at absolute positions
    // [start, start + count) before the table lookup.
    //
    // The companion of setEmbeddingOverride() for a model's *auxiliary* per-token
    // embedding stream (Gemma3/4's `per_layer_inputs`). Such a stream is a plain
    // LUT lookup on the input ids, but HF rewrites multimodal positions to the
    // pad id first:
    //     llm_input_ids = where(multimodal_mask, pad_token_id, llm_input_ids)
    // so the aux stream must NOT see the image token's own row. Feeding it the
    // image token's row corrupts all layers at every image position, and the
    // model then answers as though no image were present at all.
    //
    // Row selection only, so the byte-copy fast path still applies.
    void setTokenSubstitution(size_t start_position, size_t count, int32_t token_id);
    void clearTokenSubstitution();

    // Graph input this provider writes.
    const std::string& tensorName() const { return tensor_name_; }

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
    // Decides, once, whether `g`'s target tensor can take stored bytes verbatim.
    bool canByteCopy(const Graph& g) const;

    // Override-buffer helpers; see setEmbeddingOverride().
    bool applyOverrideRow(size_t pos, float* dst) const;
    bool overrideOverlaps(const LLMRunContext& ctx, size_t rows) const;

    // Token id to look up for absolute position `pos`; see setTokenSubstitution().
    int32_t tokenAt(size_t pos, int32_t raw_id) const;

    std::string        tensor_name_;
    std::vector<float> table_;      // flat row-major [vocab_size * hidden_size]; empty when quantized
    std::vector<float> pad_embed_;  // flat [hidden_size]; pads short prefill chunks
    size_t             hidden_size_ = 0;

    // Explicit-config path (Gemma per-layer stream). Empty = use the default
    // model_cfg.embedding_path + spec.hidden_size behaviour.
    std::string explicit_table_path_;
    size_t      explicit_row_hidden_ = 0;
    int32_t     pad_token_override_  = -1;

    // Quantized (memory-mapped) table. Used instead of table_ when set.
    QuantizedLutSpec quant_;
    QuantizedLut     qlut_;
    int32_t          pad_token_id_ = 0;  // resolved in onInitialized, used by the mmap path
    std::vector<float> scratch_;         // reused per write() to avoid per-token allocation

    // Externally supplied rows (vision embeddings) covering absolute positions
    // [override_start_, override_start_ + override_rows_/…). Empty = inactive.
    std::vector<float> override_rows_;
    size_t             override_start_ = 0;

    // Token-id substitution span; see setTokenSubstitution(). count 0 = inactive.
    size_t  sub_start_ = 0;
    size_t  sub_count_ = 0;
    int32_t sub_token_ = 0;
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

// Llama3 RoPE with frequency-dependent inv_freq scaling.
class GENIEX_API Llama3RoPEInputProvider : public InputProvider {
   public:
    Llama3RoPEInputProvider(size_t head_dim, float theta, float factor, float low_freq_factor, float high_freq_factor,
        int original_max_position_embeddings = 8192, std::string cos_name = "position_ids_cos",
        std::string sin_name = "position_ids_sin");

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    Llama3RoPEEmbedding rope_;
    std::string         cos_name_;
    std::string         sin_name_;
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

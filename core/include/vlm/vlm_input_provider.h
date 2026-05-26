// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "geniex_export.h"
#include "graph.h"
#include "llm/input_provider.h"
#include "llm/llm_types.h"
#include "types.h"

namespace geniex {

// Embedding provider for VLMs. Owns the embedding table and operates in two modes:
//   Prefill: write() slices from a pre-computed buffer (set via setBuffer()).
//   Decode:  write() falls back to per-token table lookup.
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
class GENIEX_VLM_API PrecomputedEmbeddingProvider : public InputProvider {
   public:
    explicit PrecomputedEmbeddingProvider(std::string tensor_name = "input_embeds");

    // Loads the embedding table from `path`.
    //  * `.npy`  — shape is read from the header; vocab_size/hidden_size are
    //              optional and, if non-zero, validated against it.
    //  * other   — flat row-major float32 binary; both vocab_size and
    //              hidden_size must be non-zero and match the file size.
    // Idempotent: does nothing if a table is already loaded.
    void loadTable(const std::string& path, size_t vocab_size = 0, size_t hidden_size = 0);

    void onInitialized(const ModelConfig& model_cfg, const LLMSpec& spec) override;

    // Returns flat [token_ids.size() * hidden_size] embeddings for the given tokens.
    std::vector<float> lookupBatch(const std::vector<int32_t>& token_ids) const;

    // Switch to prefill mode: write() slices from this buffer by [n_past - offset, n_past - offset + curr_len).
    // Buffer contains embeddings for the current round's tokens only (not full history),
    // so n_past_offset anchors it to the correct absolute KV position.
    void setBuffer(std::vector<float> embeds, size_t n_past_offset = 0);

    // Switch back to decode mode (per-token table lookup).
    void clearBuffer();

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    std::string        tensor_name_;
    std::vector<float> table_;      // flat [vocab_size * hidden_size]
    std::vector<float> buffer_;     // flat [current_round_tokens * hidden_size]; empty = decode mode
    std::vector<float> pad_embed_;  // flat [hidden_size]; pads short prefill chunks
    size_t             hidden_size_   = 0;
    size_t             buffer_offset_ = 0;  // absolute KV position where buffer[0] starts (= n_past at setBuffer time)
};

// How the three position dimensions (temporal, height, width) map onto the head dimension:
//   BLOCK  — contiguous segments: [0:S0] from temporal, [S0:S0+S1] from height,
//            [S0+S1:] from width.
//   STRIDE — stride-3 interleaving: temporal fills all positions, then height and
//            width override at offsets {1, 4, 7, ...} and {2, 5, 8, ...} respectively,
//            each up to mrope_section[dim] * 3 positions.
enum class MRoPEInterleaving { BLOCK, STRIDE };

// Position provider for models using Multi-dimensional RoPE (MRoPE).
// Each token has a 3D position (temporal, height, width). The head dimension is split
// into three segments (mrope_section), each driven by one position dimension.
// Stateful: tracks mrope_deltas_ across conversation turns.
class GENIEX_VLM_API MRoPEInputProvider : public InputProvider {
   public:
    // mrope_section: per-dimension sizes summing to head_dim/2, e.g. {24,20,20}.
    MRoPEInputProvider(std::vector<int> mrope_section, float theta, MRoPEInterleaving style,
        std::string cos_name = "position_ids_cos", std::string sin_name = "position_ids_sin");

    // Pre-computes cos/sin tables from full-sequence 3D position IDs.
    // position_ids: flat [3 * seq_len], layout [temporal..., height..., width...].
    // n_past_offset anchors the table to the correct absolute KV position.
    void setPositionIds(const std::vector<int32_t>& position_ids, size_t seq_len, size_t n_past_offset = 0);

    // Clear tables; write() falls back to sequential 1D positions (decode phase).
    void clearPositionIds();

    // Accumulated position offset across conversation turns, flat [3].
    const std::vector<int32_t>& mropeDeltas() const;
    void                        setMropeDeltas(std::vector<int32_t> deltas);
    void                        resetMropeDeltas();

    void write(Graph& g, const LLMRunContext& ctx) override;

   private:
    // Fills out_cos / out_sin [seq_len * half_dim_] from flat [3 * seq_len] pos IDs.
    void fillCosSin(const std::vector<int32_t>& position_ids, size_t seq_len, std::vector<float>& out_cos,
        std::vector<float>& out_sin) const;

    std::vector<int>   mrope_section_;
    float              theta_;
    MRoPEInterleaving  style_;
    std::string        cos_name_;
    std::string        sin_name_;
    size_t             half_dim_ = 0;  // sum of mrope_section_
    std::vector<float> inv_freq_;      // cached [half_dim_], computed once at construction

    std::vector<float> cos_table_;  // flat [current_round_seq_len * half_dim_]; prefill only
    std::vector<float> sin_table_;  // flat [current_round_seq_len * half_dim_]; prefill only
    size_t             seq_len_   = 0;
    size_t position_offset_       = 0;  // absolute KV position where table[0] starts (= n_past at setPositionIds time)
    bool   has_prefill_positions_ = false;  // set by setPositionIds(), cleared by clearPositionIds()

    std::vector<int32_t> mrope_deltas_;  // flat [3], initialised to {0, 0, 0}
};

}  // namespace geniex

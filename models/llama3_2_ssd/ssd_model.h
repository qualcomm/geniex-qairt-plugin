// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "geniex_export.h"
#include "llm/llm_model.h"
#include "llm/llm_utils.h"
#include "ssd_types.h"

namespace geniex {

// Decode loop override that performs tree-based self-speculative decoding.
// Drafts a candidate tree in a single AR-32 pass, then verifies and accepts greedily;
// typically yields 2-3 tokens per forward pass without a separate draft model.
class GENIEX_API SSDModel : public LLMModel {
   public:
    SSDModel(LLMSpec spec, SSDConfig ssd_cfg);

    // Return false from the callback to stop early.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg = {},
        std::function<bool(int32_t)> token_callback = nullptr) override;

    void resetKVCache() override;

   protected:
    bool onInitialized() override;

   private:
    // Returns flat parent-index array encoding the draft tree + forecast links.
    // Also populates num_draft_nodes_, samples_per_draft_level_, nodes_per_draft_level_.
    std::vector<int32_t> genAttentionMap();

    // Forecast token IDs are [vocab_size, vocab_size + draft_levels - 1].
    std::vector<int32_t> genForecastTokens(size_t repeat) const;

    std::vector<int32_t> buildSampleTree(int32_t last_token, size_t phase, size_t start_offset) const;

    std::pair<std::vector<int32_t>, std::vector<int32_t>> verifyDraftTree(
        const std::vector<int32_t>& draft_tree, size_t phase) const;

    // Dequantizes only the requested logit row (vocab_size elements).
    void readLogitsAt(size_t phase, size_t position, float* dst) const;

    int32_t argmaxLogits(const float* logits_row) const;

    std::vector<int32_t> topKLogits(const float* logits_row, size_t k) const;

    // Mask shape [num_tokens, kv_len + num_tokens].
    // Positions < kv_prefix_offset skip forecast prefix KV [0, forecast_prefix).
    std::vector<float> buildTreeAttentionMask(
        size_t n_past, size_t num_tokens, size_t seq_len, size_t kv_len, size_t kv_prefix_offset) const;

    void selectiveKVUpdate(const std::vector<bool>& selected, size_t n_accepted);

    bool loadForecastPrefix();

    // kv_prefix_offset: positions < offset skip forecast prefix; >= offset attend to it.
    void runShardsWithTreeMask(
        const std::vector<int32_t>& tokens, size_t phase, size_t n_past, size_t kv_prefix_offset);

    // Root position = n_past - forecast_prefix; each child = parent + 1.
    std::vector<int32_t> computeTreePositionIds(size_t n_past, size_t num_tokens) const;

    // Returns the RoPE table, which onInitialized() builds once head_dim is
    // known. Throws instead of a bad_optional_access if a decode/prefill path
    // is somehow reached before initialization completed.
    RotaryEmbedding& requireRope();

    SSDConfig ssd_cfg_;
    // Constructed in onInitialized() once head_dim is inferred from the graphs.
    // Overrides standard RoPE with tree-based position IDs during SSD decode.
    std::optional<RotaryEmbedding> rope_;

    size_t               draft_levels_    = 0;      // = branches.size()
    size_t               num_draft_nodes_ = 0;      // total nodes in draft tree (including root)
    std::vector<int32_t> attention_map_;            // static tree structure
    std::vector<size_t>  samples_per_draft_level_;  // top-k count per level
    std::vector<size_t>  nodes_per_draft_level_;    // node count per level

    // Pre-cached KV tensor pointers, populated in onInitialized() to avoid per-iteration lookups.
    struct KVTensorInfo {
        size_t shard;
        // Key: [H, 1, hd, kv_len] input / [H, 1, hd, seq_len] output
        void*       key_in_ptr      = nullptr;
        const void* key_out_ptr     = nullptr;
        size_t      key_in_kv_len   = 0;  // stride in token dimension
        size_t      key_out_seq_len = 0;
        size_t      key_elem_size   = 0;
        size_t      key_n_rows      = 0;  // H * hd
        // Value: [H, 1, kv_len, hd] input / [H, 1, seq_len, hd] output
        void*       val_in_ptr      = nullptr;
        const void* val_out_ptr     = nullptr;
        size_t      val_in_kv_len   = 0;
        size_t      val_out_seq_len = 0;
        size_t      val_token_size  = 0;  // hd * elem_size
        size_t      val_n_heads     = 0;  // H
    };
    std::vector<KVTensorInfo> kv_tensor_cache_;
};

}  // namespace geniex

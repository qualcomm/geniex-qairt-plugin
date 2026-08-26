// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "geniex_export.h"
#include "llm/input_provider.h"  // QuantizedLutSpec
#include "llm/llm_model.h"

namespace geniex {

// An LLMModel that additionally supports batched and tree-shaped decode passes,
// the primitives a speculative driver (EAGLE) needs to verify many candidate
// tokens in one forward and selectively commit the accepted rows' KV. Plain LLM
// and VLM decoding never touch this surface, so it lives in a subclass rather
// than on the general-purpose base.
class GENIEX_API SpeculativeLLMModel : public LLMModel {
   public:
    using LLMModel::LLMModel;

    // Requests that this engine build a quantized embedding provider bound to the
    // embedding tensor discovered from the graphs (shard-0's state input), rather
    // than the base class's default input_ids/input_embeds provider. Set before
    // initialize(); the provider is created on the standard createInputProviders()
    // seam once inferSpecFromGraphs() has resolved the tensor name, so the name is
    // never hard-coded. `table_path` empty ⇒ fall back to ModelConfig::embedding_path.
    void setEmbeddingBinding(QuantizedLutSpec quant, std::string table_path) {
        embedding_binding_ = EmbeddingBinding{std::move(quant), std::move(table_path)};
    }

    // Result of one batched decode forward. Logits and features reference the
    // engine's own graph buffers via the returned graph indices so a driver can
    // read individual rows without copying whole tensors.
    struct DecodeBatchResult {
        size_t lm_head_graph;  // graphs_ index whose out_state_name holds logits
        size_t body_graph;     // graphs_ index whose out holds the feature tensor
        size_t num_tokens;     // rows produced (= tokens.size())
    };

    // Runs one decode-phase forward over a batch of `tokens`.
    //
    // pos_ids        : per-token RoPE position (tree positions, not sequential).
    // attention_map  : parent index per token (-1 = root); builds the tree mask.
    //                  Empty ⇒ plain left-to-right causal over the chunk.
    // n_past         : KV positions already committed.
    // feature_override: optional byte buffer copied verbatim into `feature_name`
    //                  before execution (EAGLE feeds the target's hidden states
    //                  into the draft). nullptr ⇒ no override.
    DecodeBatchResult decodeBatch(const std::vector<int32_t>& tokens, const std::vector<int32_t>& pos_ids,
        const std::vector<int32_t>& attention_map, size_t n_past, float rope_theta, const void* feature_override,
        size_t feature_override_bytes, const std::string& feature_name);

    // decodeBatch over a speculative KV region: rows [0, n_keep) are the real
    // committed sequence and are always attended, while rows [n_keep, n_past)
    // hold sibling tree branches -- token i attends to a speculative row only if
    // it appears in kv_ancestors[i]. Each level of tree expansion commits the
    // frontier's KV, and the next level's nodes attend to their own committed
    // ancestors, not their siblings'. Self (row n_past + i) is always attended.
    // feature/override semantics match decodeBatch. attention_map still adds
    // intra-batch ancestors on top.
    DecodeBatchResult decodeBatchTree(const std::vector<int32_t>& tokens, const std::vector<int32_t>& pos_ids,
        const std::vector<int32_t>& attention_map, const std::vector<std::vector<int32_t>>& kv_ancestors, size_t n_keep,
        size_t n_past, float rope_theta, const void* feature_override, size_t feature_override_bytes,
        const std::string& feature_name);

    // Commits accepted KV rows (selected[i]==true) from the last decode outputs
    // into the KV input buffers at [nPast, nPast+n_accepted), then advances
    // n_past_. Mirrors selective acceptance in tree speculation.
    void commitDecodeRows(const std::vector<bool>& selected, size_t n_accepted);

    // Async variant: advances n_past_ synchronously (so the caller's position
    // math is immediately correct) but offloads the KV buffer copy to the decode
    // worker pool. The caller MUST drainDecodePool() before the next op that
    // reads this engine's KV inputs (i.e. the next decodeBatch). Falls back to
    // the synchronous path when no pool is configured. Lets a speculative driver
    // overlap the KV write with the CPU tail that follows a verify (feature read,
    // token emit, replay-seed build) — the write touches only the KV INPUT rows
    // [nPast, nPast+n_accepted), which that tail never reads.
    void commitDecodeRowsAsync(const std::vector<bool>& selected, size_t n_accepted);

    // Re-strides every KV cache between the prefill (CL - seq_len_prefill) and
    // decode (CL - seq_len_decode) layouts around a speculation loop. A driver
    // that drives decodeBatch()/prefill() directly must bracket its decode
    // passes with these, mirroring what generate() does internally.
    void switchToDecodeStride();
    void switchToPrefillStride();

    // Promotes to the smallest context length whose decode-strided KV buffer
    // (CL - seq_len_decode) can hold `n_past_ + extra_rows`, restriding every KV
    // layer decode→decode across the upgrade. `extra_rows` is the width of the
    // batch about to be written (a verify batch on the target, the peak tree
    // rows on the draft). Returns true if the active CL advanced. The KV buffer
    // must already be at the decode stride (i.e. inside a speculation loop).
    // Mirrors the per-step promoteCL() the vanilla decode loop runs so a
    // speculative driver can grow into larger CLs mid-generation instead of
    // failing once the initial CL is exhausted.
    bool promoteDecodeCL(size_t extra_rows);

   protected:
    // Builds the quantized embedding provider from the graph-inferred embedding
    // tensor name when setEmbeddingBinding() was called; otherwise defers to the
    // base. Runs after inferSpecFromGraphs(), so shard-0's state input is known.
    void createInputProviders() override;

    // Shared execution path for decodeBatch / decodeBatchTree: writes the given
    // attention mask, RoPE tables and feature override into every shard, runs the
    // decode graphs, and returns the output graph indices. Does not commit KV.
    //
    // Writes only spec_.attention_mask_name -- the speculative decode path assumes
    // a single global KV cache. Unlike prefill, it does not populate a swa_* mask,
    // so a sliding-window bundle (a second local-attention cache) is unsupported
    // here; only the global-attention family (e.g. Qwen3) is speculated.
    DecodeBatchResult runDecodeForward(const std::vector<int32_t>& tokens, const std::vector<int32_t>& pos_ids,
        const std::vector<float>& mask, size_t n_past, float rope_theta, const void* feature_override,
        size_t feature_override_bytes, const std::string& feature_name);

    // Copies the selected decode-output KV rows into the KV input buffers
    // starting at input row `dst_base`. Shared body of the sync/async commit; the
    // caller owns advancing n_past_. Runs on a worker thread in the async path.
    void copyAcceptedKVRows(const std::vector<bool>& selected, size_t dst_base);

   private:
    // Set by setEmbeddingBinding(); consumed once by createInputProviders().
    struct EmbeddingBinding {
        QuantizedLutSpec quant;
        std::string      table_path;  // empty ⇒ ModelConfig::embedding_path
    };
    std::optional<EmbeddingBinding> embedding_binding_;
};

namespace detail {

// Additive attention mask for a speculative tree KV cache. Rows [0, n_keep) are
// the real committed sequence (always attended); rows [n_keep, n_past) are
// sibling tree branches, attended only when listed in kv_ancestors[i]. Self and
// intra-batch ancestors (attention_map) are attended too. Pure and free of
// engine state; declared here so the tree-mask invariants can be tested directly.
std::vector<float> buildTreeAttentionMask(const std::vector<int32_t>& attention_map,
    const std::vector<std::vector<int32_t>>& kv_ancestors, size_t n_keep, size_t n_past, size_t num_tokens,
    size_t seq_len, size_t kv_len);

}  // namespace detail

}  // namespace geniex

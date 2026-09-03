// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "geniex-proc/tokenizer.h"  // Tokenizer, ApplyChatTemplateOptions
#include "geniex-proc/types.h"      // ChatMessage
#include "geniex_export.h"
#include "llm/llm_model.h"
#include "types.h"

namespace geniex {

struct GenerateResult {
    std::string full_text;
    double      ttft_ms           = 0.0;  // time-to-first-token
    double      media_ms          = 0.0;  // vision/audio encoder time (VLM only; 0 for text)
    double      decode_ms         = 0.0;  // decode phase wall time
    int64_t     prompt_tokens     = 0;    // text + media tokens
    int64_t     generated_tokens  = 0;
    double      tokens_per_second = 0.0;
    std::string stop_reason;  // "eos" | "length" | "user" | "stop_sequence" | "context_length"
};

// High-level API: tokenizer + chat template + streaming generation over an LLMModel.
class GENIEX_API LLMPipeline {
   public:
    LLMPipeline();
    ~LLMPipeline();

    LLMPipeline(LLMPipeline&&) noexcept;
    LLMPipeline& operator=(LLMPipeline&&) noexcept;
    LLMPipeline(const LLMPipeline&)            = delete;
    LLMPipeline& operator=(const LLMPipeline&) = delete;

    // Takes ownership of `model`, initializes it,. Returns false on failure.
    bool create(LLMModel model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg);

    template <typename ModelT,
        std::enable_if_t<std::is_base_of_v<LLMModel, ModelT> && !std::is_same_v<LLMModel, ModelT>, int> = 0>
    bool create(ModelT model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
        return createImpl(std::make_unique<ModelT>(std::move(model)), runtime_cfg, model_cfg);
    }

    bool isReady() const;

    // Clears KV state and resets to the start of a new conversation.
    void reset();

    // Prepends `token_id` once on the first turn.
    // Pass -1 (default) to disable BOS prepending.
    void setBosTokenId(int32_t token_id);

    // Renders `messages` through the bundled Jinja chat template.
    std::string applyChatTemplate(
        const std::vector<ChatMessage>& messages, const ApplyChatTemplateOptions& opts = {}) const;

    // on_token is called with each decoded text piece; return false to stop early.
    GenerateResult generate(const std::string& prompt_utf8, const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr);

    // Generates from a fully rendered chat prompt while safely reusing any
    // exact token prefix already resident in KV. If the rendered history edits,
    // removes, or normalizes earlier tokens, the model rewinds to the exact
    // common prefix and recomputes only the divergent suffix. Sampler state is
    // rebuilt from the complete canonical prompt so branch tokens cannot leak
    // through repetition/frequency penalties.
    GenerateResult generateFullPrompt(const std::string& prompt_utf8, const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr);

    // Pre-tokenized variant. Bypasses encode(); the caller is responsible for any
    // special tokens (BOS/EOS) — no BOS is prepended here.
    GenerateResult generate(const std::vector<int32_t>& input_ids, const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr);

    // Raw logits from a single (non-autoregressive) forward pass over `input_ids`.
    // Thin pass-through to LLMModel::forwardLogits — for on-target metrics
    // (perplexity, MMLU, MMMU) that examine logits instead of generating text.
    // No BOS is prepended; the caller supplies any special tokens.
    //
    // all_positions == false (default): the last token's logits row (vocabSize() floats).
    // all_positions == true: every position, row-major [input_ids.size(), vocabSize()].
    //
    // Runs against a fresh KV cache and leaves it clean. Throws if the pipeline is
    // not ready, if `input_ids` is empty, or if it exceeds the max context length.
    std::vector<float> forwardLogits(const std::vector<int32_t>& input_ids, bool all_positions = false);

    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

    // Static model metadata for callers that need vocab_size / BOS without a
    // tokenizer (e.g. random-id benchmark prefill). 0 / -1 when unavailable.
    size_t  vocabSize() const;
    int32_t bosTokenId() const;

    bool createImpl(std::unique_ptr<LLMModel> model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg);

   private:
    GenerateResult generateTokens(std::vector<int32_t> input_ids, const GenerationConfig& gen_cfg,
        const std::function<bool(const char*)>& on_token,
        const std::vector<int32_t>*             canonical_prompt_tokens = nullptr);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geniex

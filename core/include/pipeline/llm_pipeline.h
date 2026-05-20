// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "geniex_export.h"
#include "llm/llm_model.h"
#include "pipeline/chat_template.h"
#include "types.h"

namespace geniex {

struct GenerateResult {
    std::string full_text;
    int64_t     prompt_tokens     = 0;
    int64_t     generated_tokens  = 0;
    double      ttft_ms           = 0.0;  // time-to-first-token
    double      decode_ms         = 0.0;  // decode phase wall time
    double      tokens_per_second = 0.0;
    std::string stop_reason;  // "eos" | "length" | "user" | "context_length"
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

    // Takes ownership of `model` and initializes it. Returns false on failure.
    bool create(ChatTemplateFunc chat_template, LLMModel model, const QnnRuntimeConfig& runtime_cfg,
        const ModelConfig& model_cfg);

    // Polymorphic overload: accepts any LLMModel subclass without slicing.
    template <typename ModelT,
        std::enable_if_t<std::is_base_of_v<LLMModel, ModelT> && !std::is_same_v<LLMModel, ModelT>, int> = 0>
    bool create(ChatTemplateFunc chat_template, ModelT model, const QnnRuntimeConfig& runtime_cfg,
        const ModelConfig& model_cfg) {
        return createImpl(chat_template, std::make_unique<ModelT>(std::move(model)), runtime_cfg, model_cfg);
    }

    bool isReady() const;

    // Clears KV state and resets to the start of a new conversation.
    void reset();

    // Sets a system prompt to be injected on the next applyChatTemplate() call.
    // The prompt is consumed (cleared) once applyChatTemplate() is called, so
    // it must be set again if needed for a subsequent turn. Typical usage is to
    // call setSystemPrompt() once before the first turn of a conversation.
    void setSystemPrompt(const std::string& prompt);

    // Sets the tools (already parsed from JSON at the SDK plugin layer) to be
    // injected on the next applyChatTemplate() call. Tools are consumed
    // (cleared) once applyChatTemplate() is called and must be re-staged each
    // turn that should expose tools to the model. Pass an empty vector to
    // clear without re-staging. Whether tools are actually rendered depends on
    // the configured chat-template formatter; formatters without tool-call
    // support silently ignore them.
    void setTools(ChatTools tools);

    // Formats a user message with the chat template, injecting any pending
    // system prompt set via setSystemPrompt() and any pending tools set via
    // setTools(). Both the system prompt and tools state are cleared after
    // this call; call setSystemPrompt() / setTools() again to re-inject.
    std::string applyChatTemplate(const std::string& user_message, bool enable_thinking = true);

    // on_token is called with each decoded text piece; return false to stop early.
    GenerateResult generate(const std::string& prompt_utf8, const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr);

    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

    bool createImpl(ChatTemplateFunc chat_template, std::unique_ptr<LLMModel> model,
        const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geniex
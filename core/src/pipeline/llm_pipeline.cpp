// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/llm_pipeline.h"

#include <chrono>
#include <optional>
#include <sstream>

#include "llm/llm_model.h"
#include "types.h"
#include "geniex-proc/tokenizer.h"

namespace geniex {


struct LLMPipeline::Impl {
    std::unique_ptr<LLMModel>              model;
    std::unique_ptr<geniex::Tokenizer>     tokenizer;
    ChatTemplateFunc                       chat_template = chatMLTemplate;
    std::string                            system_prompt;
    ChatTools                              tools;
    bool ready = false;
};

LLMPipeline::LLMPipeline()  : impl_(std::make_unique<Impl>()) {}
LLMPipeline::~LLMPipeline() = default;
LLMPipeline::LLMPipeline(LLMPipeline&&) noexcept            = default;
LLMPipeline& LLMPipeline::operator=(LLMPipeline&&) noexcept = default;

bool LLMPipeline::create(ChatTemplateFunc chat_template,
                          LLMModel model,
                          const QnnRuntimeConfig& runtime_cfg,
                          const ModelConfig& model_cfg) {
    return createImpl(chat_template,
                      std::make_unique<LLMModel>(std::move(model)),
                      runtime_cfg, model_cfg);
}

bool LLMPipeline::createImpl(ChatTemplateFunc chat_template,
                              std::unique_ptr<LLMModel> model,
                              const QnnRuntimeConfig& runtime_cfg,
                              const ModelConfig& model_cfg) {
    impl_->chat_template = chat_template;
    impl_->model = std::move(model);

    try {
        if (!impl_->model->initialize(runtime_cfg, model_cfg)) {
            impl_->model.reset();
            return false;
        }
    } catch (...) {
        impl_->model.reset();
        return false;
    }

    impl_->tokenizer = geniex::Tokenizer::from_file(model_cfg.tokenizer_path);
    if (!impl_->tokenizer) {
        impl_->model.reset();
        return false;
    }

    impl_->ready = true;
    return true;
}

bool LLMPipeline::isReady() const {
    return impl_ && impl_->ready;
}

void LLMPipeline::reset() {
    if (impl_->model)
        impl_->model->resetKVCache();
}

void LLMPipeline::setSystemPrompt(const std::string& prompt) {
    impl_->system_prompt = prompt;
}

void LLMPipeline::setTools(ChatTools tools) {
    impl_->tools = std::move(tools);
}

std::string LLMPipeline::applyChatTemplate(
    const std::string& user_message,
    bool enable_thinking)
{
    std::string result = impl_->chat_template(user_message,
                                              impl_->system_prompt,
                                              impl_->tools,
                                              enable_thinking);
    impl_->system_prompt.clear();
    impl_->tools.clear();
    return result;
}

GenerateResult LLMPipeline::generate(
    const std::string& prompt_utf8,
    const GenerationConfig& gen_cfg,
    std::function<bool(const char*)> on_token)
{
    GenerateResult result;
    if (!impl_->ready || !impl_->model) {
        result.stop_reason = "error";
        return result;
    }

    auto encoded = impl_->tokenizer->encode(prompt_utf8);
    std::vector<int32_t> input_ids(encoded.begin(), encoded.end());
    result.prompt_tokens = static_cast<int64_t>(input_ids.size());

    // Inject the tokenizer; LLMModel needs it for grammar/EOG resolution.
    GenerationConfig effective_cfg = gen_cfg;
    effective_cfg.tokenizer        = impl_->tokenizer.get();

    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();
    Clock::time_point t_first_token;
    bool got_first = false;
    bool user_stopped = false;

    std::ostringstream full_text;

    auto output_tokens = impl_->model->generate(
        input_ids,
        effective_cfg,
        [&](int32_t tok) -> bool {
            if (!got_first) {
                t_first_token = Clock::now();
                got_first     = true;
            }

            std::string piece = impl_->tokenizer->decode({tok});
            full_text << piece;

            if (on_token && !piece.empty()) {
                if (!on_token(piece.c_str())) {
                    user_stopped = true;
                    return false;
                }
            }
            return !user_stopped;
        });

    auto t_end = Clock::now();

    result.full_text        = full_text.str();
    result.generated_tokens = static_cast<int64_t>(output_tokens.size());

    if (got_first) {
        result.ttft_ms   = std::chrono::duration<double, std::milli>(
                               t_first_token - t_start).count();
        result.decode_ms = std::chrono::duration<double, std::milli>(
                               t_end - t_first_token).count();

        size_t decode_tok = output_tokens.size() > 1
                                ? output_tokens.size() - 1 : 0;
        result.tokens_per_second =
            result.decode_ms > 0.0
                ? decode_tok / (result.decode_ms / 1000.0)
                : 0.0;
    }

    if (user_stopped)
        result.stop_reason = "user";
    else if (result.generated_tokens >= gen_cfg.max_tokens)
        result.stop_reason = "length";
    else
        result.stop_reason = "eos";

    return result;
}

void LLMPipeline::saveKVCache(const std::string& path) const {
    if (impl_->model) impl_->model->saveKVCacheToFile(path);
}

void LLMPipeline::loadKVCache(const std::string& path) {
    if (impl_->model) impl_->model->loadKVCacheFromFile(path);
}

size_t LLMPipeline::nPast() const {
    return impl_->model ? impl_->model->nPast() : 0;
}

} // namespace geniex
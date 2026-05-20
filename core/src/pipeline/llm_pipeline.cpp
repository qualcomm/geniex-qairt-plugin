// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/llm_pipeline.h"

#include <chrono>
#include <optional>
#include <sstream>

#include "geniex-proc/tokenizer.h"
#include "llm/llm_model.h"
#include "types.h"

namespace geniex {

namespace {

using Clock = std::chrono::high_resolution_clock;

// Populate `result` from in-flight generation state. Used on both the success
// path and the context-length-exceeded catch path so partial output is surfaced
// uniformly.
void finalize_generate_result(GenerateResult& result, std::ostringstream& full_text, int64_t generated_tokens,
    Clock::time_point t_start, Clock::time_point t_first_token, Clock::time_point t_end, bool got_first,
    const char* stop_reason) {
    result.full_text        = full_text.str();
    result.generated_tokens = generated_tokens;
    if (got_first) {
        result.ttft_ms   = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
        result.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();

        const int64_t decode_tok = generated_tokens > 1 ? generated_tokens - 1 : 0;
        result.tokens_per_second = result.decode_ms > 0.0 ? decode_tok / (result.decode_ms / 1000.0) : 0.0;
    }
    result.stop_reason = stop_reason;
}

}  // namespace

struct LLMPipeline::Impl {
    std::unique_ptr<LLMModel>          model;
    std::unique_ptr<geniex::Tokenizer> tokenizer;
    ChatTemplateFunc                   chat_template = chatMLTemplate;
    std::string                        system_prompt;
    ChatTools                          tools;
    bool                               ready = false;
};

LLMPipeline::LLMPipeline() : impl_(std::make_unique<Impl>()) {}
LLMPipeline::~LLMPipeline()                                 = default;
LLMPipeline::LLMPipeline(LLMPipeline&&) noexcept            = default;
LLMPipeline& LLMPipeline::operator=(LLMPipeline&&) noexcept = default;

bool LLMPipeline::create(
    ChatTemplateFunc chat_template, LLMModel model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    return createImpl(chat_template, std::make_unique<LLMModel>(std::move(model)), runtime_cfg, model_cfg);
}

bool LLMPipeline::createImpl(ChatTemplateFunc chat_template, std::unique_ptr<LLMModel> model,
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    impl_->chat_template = chat_template;
    impl_->model         = std::move(model);

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

bool LLMPipeline::isReady() const { return impl_ && impl_->ready; }

void LLMPipeline::reset() {
    if (impl_->model) impl_->model->resetKVCache();
}

void LLMPipeline::setSystemPrompt(const std::string& prompt) { impl_->system_prompt = prompt; }

void LLMPipeline::setTools(ChatTools tools) { impl_->tools = std::move(tools); }

std::string LLMPipeline::applyChatTemplate(const std::string& user_message, bool enable_thinking) {
    std::string result = impl_->chat_template(user_message, impl_->system_prompt, impl_->tools, enable_thinking);
    impl_->system_prompt.clear();
    impl_->tools.clear();
    return result;
}

GenerateResult LLMPipeline::generate(
    const std::string& prompt_utf8, const GenerationConfig& gen_cfg, std::function<bool(const char*)> on_token) {
    GenerateResult result;
    if (!impl_->ready || !impl_->model) {
        result.stop_reason = "error";
        return result;
    }

    auto                 encoded = impl_->tokenizer->encode(prompt_utf8);
    std::vector<int32_t> input_ids(encoded.begin(), encoded.end());
    result.prompt_tokens = static_cast<int64_t>(input_ids.size());

    // Inject the tokenizer; LLMModel needs it for grammar/EOG resolution.
    GenerationConfig effective_cfg = gen_cfg;
    effective_cfg.tokenizer        = impl_->tokenizer.get();

    auto              t_start = Clock::now();
    Clock::time_point t_first_token;
    bool              got_first    = false;
    bool              user_stopped = false;

    std::ostringstream full_text;
    // Counted inside the callback so the partial total is correct even if the
    // model throws mid-decode (the returned vector is destroyed during unwind).
    int64_t streamed_tokens = 0;

    auto on_each_token = [&](int32_t tok) -> bool {
        if (!got_first) {
            t_first_token = Clock::now();
            got_first     = true;
        }

        std::string piece = impl_->tokenizer->decode_token(tok);
        full_text << piece;
        ++streamed_tokens;

        if (on_token && !piece.empty()) {
            if (!on_token(piece.c_str())) {
                user_stopped = true;
                return false;
            }
        }
        return !user_stopped;
    };

    try {
        auto output_tokens = impl_->model->generate(input_ids, effective_cfg, on_each_token);
        auto t_end         = Clock::now();

        const int64_t total  = static_cast<int64_t>(output_tokens.size());
        const char*   reason = user_stopped ? "user" : (total >= gen_cfg.max_tokens ? "length" : "eos");
        finalize_generate_result(result, full_text, total, t_start, t_first_token, t_end, got_first, reason);
        return result;
    } catch (const ContextLengthExceededError&) {
        const auto t_end = Clock::now();
        finalize_generate_result(
            result, full_text, streamed_tokens, t_start, t_first_token, t_end, got_first, "context_length");
        return result;
    }
}

void LLMPipeline::saveKVCache(const std::string& path) const {
    if (impl_->model) impl_->model->saveKVCacheToFile(path);
}

void LLMPipeline::loadKVCache(const std::string& path) {
    if (impl_->model) impl_->model->loadKVCacheFromFile(path);
}

size_t LLMPipeline::nPast() const { return impl_->model ? impl_->model->nPast() : 0; }

}  // namespace geniex
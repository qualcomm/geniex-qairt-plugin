// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/llm_pipeline.h"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "geniex-proc/stop_matcher.h"
#include "geniex-proc/tokenizer.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"  // bundleDirOf
#include "logging.h"
#include "types.h"

namespace geniex {

namespace {

using Clock = std::chrono::high_resolution_clock;

// model_cfg.tokenizer_config_path when set, else tokenizer_config.json beside the bundle.
std::string resolveTokenizerConfigPath(const ModelConfig& model_cfg) {
    if (model_cfg.tokenizer_config_path && !model_cfg.tokenizer_config_path->empty()) {
        return *model_cfg.tokenizer_config_path;
    }
    return (bundleDirOf(model_cfg) / "tokenizer_config.json").string();
}

// Populate `result` from in-flight generation state. Used on both the success
// path and the context-length-exceeded catch path so partial output is surfaced
// uniformly.
void finalize_generate_result(GenerateResult& result, const std::string& full_text, int64_t generated_tokens,
    Clock::time_point t_start, Clock::time_point t_first_token, Clock::time_point t_end, bool got_first,
    const char* stop_reason) {
    result.full_text = full_text;

    // Token-count convention: align with Genie's `num-generated-tokens`, which
    // counts the terminating EOS sample as a generated token. geniex's decode
    // loop stops before emitting EOS into the text, so `generated_tokens` here
    // excludes it; add it back to the reported count (but not to full_text) when
    // generation actually ended on EOS. Length/user stops have no EOS to count.
    const bool ended_on_eos = stop_reason != nullptr && std::strcmp(stop_reason, "eos") == 0;
    result.generated_tokens = generated_tokens + (ended_on_eos ? 1 : 0);

    if (got_first) {
        result.ttft_ms   = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
        result.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();

        // Genie's token-generation-rate divides the EOS-inclusive token count by
        // the decode window, so use the same numerator for a comparable tok/s.
        result.tokens_per_second = result.decode_ms > 0.0 ? result.generated_tokens / (result.decode_ms / 1000.0) : 0.0;
    }
    result.stop_reason = stop_reason;
}

}  // namespace

struct LLMPipeline::Impl {
    std::unique_ptr<LLMModel>          model;
    std::unique_ptr<geniex::Tokenizer> tokenizer;
    bool                               ready        = false;
    int32_t                            bos_token_id = -1;
};

LLMPipeline::LLMPipeline() : impl_(std::make_unique<Impl>()) {}
LLMPipeline::~LLMPipeline()                                 = default;
LLMPipeline::LLMPipeline(LLMPipeline&&) noexcept            = default;
LLMPipeline& LLMPipeline::operator=(LLMPipeline&&) noexcept = default;

bool LLMPipeline::create(LLMModel model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    return createImpl(std::make_unique<LLMModel>(std::move(model)), runtime_cfg, model_cfg);
}

bool LLMPipeline::createImpl(
    std::unique_ptr<LLMModel> model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    impl_->model = std::move(model);

    try {
        if (!impl_->model->initialize(runtime_cfg, model_cfg)) {
            GENIEX_LOG_ERROR("LLMPipeline: model initialize() failed");
            impl_->model.reset();
            return false;
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("LLMPipeline: model initialize() threw: {}", e.what());
        impl_->model.reset();
        return false;
    } catch (...) {
        GENIEX_LOG_ERROR("LLMPipeline: model initialize() threw a non-std exception");
        impl_->model.reset();
        return false;
    }

    const std::string tcp = resolveTokenizerConfigPath(model_cfg);
    impl_->tokenizer      = geniex::Tokenizer::from_file(model_cfg.tokenizer_path, tcp);
    if (!impl_->tokenizer) {
        impl_->model.reset();
        return false;
    }
    if (!impl_->tokenizer->has_chat_template()) {
        GENIEX_LOG_ERROR("LLMPipeline: tokenizer_config.json at '{}' has no chat_template", tcp);
        impl_->tokenizer.reset();
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

void LLMPipeline::setBosTokenId(int32_t token_id) { impl_->bos_token_id = token_id; }

std::string LLMPipeline::applyChatTemplate(
    const std::vector<ChatMessage>& messages, const ApplyChatTemplateOptions& opts) const {
    if (!impl_->tokenizer) {
        throw std::runtime_error("LLMPipeline::applyChatTemplate: pipeline is not initialized");
    }
    return impl_->tokenizer->apply_chat_template(messages, opts);
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

    // Prepend BOS only on the first turn. Gemma's tokenizer has add_bos_token=false,
    // but the model REQUIRES a leading <bos> or generation degenerates into repeated
    // garbage; the chat template already emits one, so the front()-check avoids a
    // double-BOS when a template (or the caller) supplied it.
    if (impl_->bos_token_id >= 0 && impl_->model->nPast() == 0 &&
        (input_ids.empty() || input_ids.front() != impl_->bos_token_id)) {
        input_ids.insert(input_ids.begin(), impl_->bos_token_id);
    }

    // Mirror test_inference.py's "ids start with [...]" diagnostic so BOS/chat-template
    // application is verifiable at runtime. Only the leading ids matter for the check.
    if (!input_ids.empty()) {
        const size_t       n = std::min<size_t>(input_ids.size(), 5);
        std::ostringstream head;
        for (size_t i = 0; i < n; ++i) head << (i ? ", " : "") << input_ids[i];
        GENIEX_LOG_INFO("prompt: {} tokens, first-turn BOS={}, ids start with [{}]",
            input_ids.size(),
            (impl_->model->nPast() == 0 && !input_ids.empty() && input_ids.front() == impl_->bos_token_id),
            head.str());
    }

    return generateTokens(std::move(input_ids), gen_cfg, on_token);
}

GenerateResult LLMPipeline::generateFullPrompt(
    const std::string& prompt_utf8, const GenerationConfig& gen_cfg, std::function<bool(const char*)> on_token) {
    GenerateResult result;
    if (!impl_->ready || !impl_->model) {
        result.stop_reason = "error";
        return result;
    }

    auto                 encoded = impl_->tokenizer->encode(prompt_utf8);
    std::vector<int32_t> full_prompt_ids(encoded.begin(), encoded.end());

    // This is a canonical full-history tokenization, so BOS belongs at the
    // beginning regardless of how many retained KV rows currently exist.
    if (impl_->bos_token_id >= 0 && (full_prompt_ids.empty() || full_prompt_ids.front() != impl_->bos_token_id)) {
        full_prompt_ids.insert(full_prompt_ids.begin(), impl_->bos_token_id);
    }
    if (full_prompt_ids.empty()) {
        result.stop_reason = "error";
        return result;
    }

    const size_t matched = impl_->model->reconcilePromptTokens(full_prompt_ids);
    if (matched > full_prompt_ids.size()) {
        throw std::runtime_error("LLMPipeline::generateFullPrompt: invalid prefix match");
    }
    std::vector<int32_t> suffix(full_prompt_ids.begin() + static_cast<std::ptrdiff_t>(matched), full_prompt_ids.end());
    if (suffix.empty()) {
        // Sampling again without a newly evaluated prompt token would reuse
        // stale logits from an unspecified phase. Clear both KV and the
        // one-shot sampler override before failing so a direct SDK caller that
        // catches the exception cannot accidentally continue from provisional
        // reconciliation state.
        impl_->model->resetKVCache();
        throw std::runtime_error("LLMPipeline::generateFullPrompt: full prompt adds no tokens");
    }

    GENIEX_LOG_INFO("full prompt: {} tokens, matched KV prefix={}, evaluating suffix={}",
        full_prompt_ids.size(),
        matched,
        suffix.size());
    return generateTokens(std::move(suffix), gen_cfg, on_token);
}

GenerateResult LLMPipeline::generate(
    const std::vector<int32_t>& input_ids, const GenerationConfig& gen_cfg, std::function<bool(const char*)> on_token) {
    GenerateResult result;
    if (!impl_->ready || !impl_->model) {
        result.stop_reason = "error";
        return result;
    }

    // Pre-tokenized path: no BOS prepending, no chat template —
    // the caller owns the token ids verbatim.
    return generateTokens(input_ids, gen_cfg, on_token);
}

GenerateResult LLMPipeline::generateTokens(
    std::vector<int32_t> input_ids, const GenerationConfig& gen_cfg, const std::function<bool(const char*)>& on_token) {
    GenerateResult result;
    result.prompt_tokens = static_cast<int64_t>(input_ids.size());

    // Inject the tokenizer; LLMModel needs it for grammar/EOG resolution.
    GenerationConfig effective_cfg = gen_cfg;
    effective_cfg.tokenizer        = impl_->tokenizer.get();

    auto              t_start = Clock::now();
    Clock::time_point t_first_token;
    bool              got_first    = false;
    bool              user_stopped = false;

    std::string full_text;
    // Counted inside the callback so the partial total is correct even if the
    // model throws mid-decode (the returned vector is destroyed during unwind).
    int64_t streamed_tokens = 0;

    // Native stop-sequence support (mirrors llama_cpp): stop strings are matched
    // byte-wise against the streamed output, so a stop that spans token
    // boundaries never leaks through the token callback.
    StopMatcher stop_matcher(gen_cfg.stop_sequences);
    const bool  has_stops    = stop_matcher.active();
    bool        stop_matched = false;

    auto on_each_token = [&](int32_t tok) -> bool {
        if (!got_first) {
            t_first_token = Clock::now();
            got_first     = true;
        }

        std::string piece = impl_->tokenizer->decode_token(tok);
        full_text += piece;
        ++streamed_tokens;

        if (!has_stops) {
            if (on_token && !piece.empty()) {
                if (!on_token(piece.c_str())) {
                    user_stopped = true;
                    return false;
                }
            }
            return !user_stopped;
        }

        if (stop_matcher.feed(piece)) {
            // Stop sequence matched: drop the match (and anything after it)
            // from the output and cancel generation.
            stop_matched = true;
            full_text.resize(stop_matcher.matchOffset());
            while (true) {
                const std::string safe = stop_matcher.takeReady();
                if (safe.empty()) {
                    break;
                }
                if (on_token && !on_token(safe.c_str())) {
                    user_stopped = true;
                    return false;
                }
            }
            return false;
        }

        // Emit the bytes that can no longer start a match; hold back the tail.
        while (true) {
            const std::string safe = stop_matcher.takeReady();
            if (safe.empty()) {
                break;
            }
            if (on_token && !on_token(safe.c_str())) {
                user_stopped = true;
                return false;
            }
        }
        return !user_stopped;
    };

    // Generation ended without a stop match: the streamed callback still owes
    // the held-back tail, which is all valid output.
    auto release_held_tail = [&]() {
        if (has_stops && on_token && !user_stopped) {
            const std::string tail = stop_matcher.flush();
            if (!tail.empty()) {
                on_token(tail.c_str());
            }
        }
    };

    try {
        auto output_tokens = impl_->model->generate(input_ids, effective_cfg, on_each_token);
        auto t_end         = Clock::now();

        release_held_tail();

        const int64_t total  = static_cast<int64_t>(output_tokens.size());
        const char*   reason = stop_matched   ? "stop_sequence"
                               : user_stopped ? "user"
                                              : (total >= gen_cfg.max_tokens ? "length" : "eos");
        finalize_generate_result(result, full_text, total, t_start, t_first_token, t_end, got_first, reason);
        return result;
    } catch (const PromptTooLongError&) {
        const auto t_end = Clock::now();
        finalize_generate_result(
            result, full_text, streamed_tokens, t_start, t_first_token, t_end, got_first, "prompt_too_long");
        return result;
    } catch (const ContextLengthExceededError&) {
        release_held_tail();
        const auto t_end = Clock::now();
        finalize_generate_result(
            result, full_text, streamed_tokens, t_start, t_first_token, t_end, got_first, "context_length");
        return result;
    }
}

std::vector<float> LLMPipeline::forwardLogits(const std::vector<int32_t>& input_ids, bool all_positions) {
    if (!impl_->model) {
        throw std::runtime_error("LLMPipeline::forwardLogits: pipeline is not initialized");
    }
    return impl_->model->forwardLogits(input_ids, all_positions);
}

void LLMPipeline::saveKVCache(const std::string& path) const {
    if (impl_->model) impl_->model->saveKVCacheToFile(path);
}

void LLMPipeline::loadKVCache(const std::string& path) {
    if (impl_->model) impl_->model->loadKVCacheFromFile(path);
}

size_t LLMPipeline::nPast() const { return impl_->model ? impl_->model->nPast() : 0; }

size_t LLMPipeline::vocabSize() const { return impl_->model ? impl_->model->vocabSize() : 0; }

int32_t LLMPipeline::bosTokenId() const { return impl_->bos_token_id; }

}  // namespace geniex

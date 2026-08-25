// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Generic, family-free LLM factory + pipeline. Intended as a sketch of where
// per-family files (`models/qwen3/qwen3.h`, `models/llama3/llama3.h`, ...)
// can converge once the runtime stops needing model-specific specs.
//
// Mirrors the per-family shape — `makeModel` + `makePipeline` — but:
//   - LLMSpec is built entirely from `metadata.json` + `genie_config.json`
//     via `core/` loaders. No hardcoded constants.
//   - Chat template is loaded from `tokenizer_config.json` via geniex-proc,
//     not bound at create() time as a `ChatTemplateFunc`.
//   - applyChatTemplate takes the FULL message history. HuggingFace Jinja
//     templates index into prior turns (Qwen3 reverse-iterates messages,
//     tool-call/tool-response interleaving needs full context, etc.).

#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "pipeline/llm_pipeline.h"  // GenerateResult
#include "runtime.h"
#include "types.h"

namespace geniex {
namespace auto_llm {

// 1:1 with qwen3::makeModel — uses ONLY core/ loaders + factories. The
// returned LLMModel builds the standard embedding + RoPE providers at
// initialize() time, with architecture shapes inferred from the graph tensors.
inline LLMModel makeModel(const ModelConfig& model_cfg) {
    auto gc = parseGenieConfig(bundleDirOf(model_cfg));
    return LLMModel(buildSpecSkeleton(gc), std::move(gc));
}

// High-level pipeline that owns the model, the geniex-proc Tokenizer, and
// streaming/timing for generation. The geniex-proc Tokenizer carries the
// chat template loaded from tokenizer_config.json.
class Pipeline {
   public:
    Pipeline()  = default;
    ~Pipeline() = default;

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&)                 = default;
    Pipeline& operator=(Pipeline&&)      = default;

    // Initializes the model and loads the tokenizer + chat template. Returns
    // false on failure (logs the cause).
    bool create(LLMModel model, const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg,
        const std::string& tokenizer_config_path) {
        try {
            model_ = std::make_unique<LLMModel>(std::move(model));
            if (!model_->initialize(runtime_cfg, model_cfg)) {
                model_.reset();
                return false;
            }
            tokenizer_ = Tokenizer::from_file(model_cfg.tokenizer_path, tokenizer_config_path);
            if (!tokenizer_) {
                model_.reset();
                return false;
            }
            if (!tokenizer_->has_chat_template()) {
                GENIEX_LOG_ERROR(
                    "auto_llm::Pipeline: tokenizer_config.json at '{}' has no chat_template", tokenizer_config_path);
                tokenizer_.reset();
                model_.reset();
                return false;
            }
            ready_ = true;
            return true;
        } catch (const std::exception& e) {
            GENIEX_LOG_ERROR("auto_llm::Pipeline::create failed: {}", e.what());
            model_.reset();
            tokenizer_.reset();
            ready_ = false;
            return false;
        }
    }

    bool isReady() const { return ready_ && model_ && tokenizer_; }

    void reset() {
        if (model_) model_->resetKVCache();
    }

    size_t nPast() const { return model_ ? model_->nPast() : 0; }

    Tokenizer& tokenizer() { return *tokenizer_; }

    // Renders the full message history through the bundled Jinja template.
    std::string applyChatTemplate(
        const std::vector<ChatMessage>& messages, const Tokenizer::ApplyChatTemplateOptions& opts = {}) const {
        return tokenizer_->apply_chat_template(messages, opts);
    }

    // Generates against a pre-formatted prompt string. Mirrors
    // LLMPipeline::generate — same GenerateResult shape and on_token
    // streaming contract.
    GenerateResult generate(const std::string& prompt_utf8, const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr) {
        GenerateResult result;
        if (!isReady()) {
            result.stop_reason = "error";
            return result;
        }

        auto                 encoded = tokenizer_->encode(prompt_utf8, /*add_special_tokens=*/false);
        std::vector<int32_t> input_ids(encoded.begin(), encoded.end());
        result.prompt_tokens = static_cast<int64_t>(input_ids.size());

        // Inject the tokenizer; LLMModel needs it for grammar/EOG resolution.
        GenerationConfig effective_cfg = gen_cfg;
        effective_cfg.tokenizer        = tokenizer_.get();

        using Clock               = std::chrono::high_resolution_clock;
        auto              t_start = Clock::now();
        Clock::time_point t_first_token;
        bool              got_first    = false;
        bool              user_stopped = false;

        std::ostringstream full_text;
        // Counted inside the callback so the partial total survives a
        // mid-decode throw (the returned vector is destroyed during unwind).
        int64_t streamed_tokens = 0;

        auto on_each_token = [&](int32_t tok) -> bool {
            if (!got_first) {
                t_first_token = Clock::now();
                got_first     = true;
            }
            std::string piece = tokenizer_->decode_token(tok);
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
            auto          output_tokens = model_->generate(input_ids, effective_cfg, on_each_token);
            const auto    t_end         = Clock::now();
            const int64_t total         = static_cast<int64_t>(output_tokens.size());
            const char*   reason        = user_stopped ? "user" : (total >= gen_cfg.max_tokens ? "length" : "eos");
            finalize(result, full_text, total, t_start, t_first_token, t_end, got_first, reason);
            return result;
        } catch (const ContextLengthExceededError&) {
            const auto t_end = Clock::now();
            finalize(result, full_text, streamed_tokens, t_start, t_first_token, t_end, got_first, "context_length");
            return result;
        }
    }

    // Convenience: applyChatTemplate + generate in one call.
    GenerateResult generateChat(const std::vector<ChatMessage>& messages, const GenerationConfig& gen_cfg = {},
        const Tokenizer::ApplyChatTemplateOptions& opts = {}, std::function<bool(const char*)> on_token = nullptr) {
        GenerateResult result;
        if (!isReady()) {
            result.stop_reason = "error";
            return result;
        }
        std::string prompt;
        try {
            prompt = applyChatTemplate(messages, opts);
        } catch (const std::exception& e) {
            GENIEX_LOG_ERROR("auto_llm::Pipeline::generateChat: chat template failed: {}", e.what());
            result.stop_reason = "error";
            return result;
        }
        return generate(prompt, gen_cfg, std::move(on_token));
    }

   private:
    static void finalize(GenerateResult& result, std::ostringstream& full_text, int64_t generated_tokens,
        std::chrono::high_resolution_clock::time_point t_start,
        std::chrono::high_resolution_clock::time_point t_first_token,
        std::chrono::high_resolution_clock::time_point t_end, bool got_first, const char* stop_reason) {
        result.full_text = full_text.str();

        // Align with Genie's `num-generated-tokens`, which counts the terminating
        // EOS sample (geniex stops before emitting EOS into the text).
        const bool ended_on_eos = stop_reason != nullptr && std::strcmp(stop_reason, "eos") == 0;
        result.generated_tokens = generated_tokens + (ended_on_eos ? 1 : 0);
        if (got_first) {
            result.ttft_ms   = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
            result.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
            // Genie divides the EOS-inclusive token count by the decode window.
            result.tokens_per_second =
                result.decode_ms > 0.0 ? result.generated_tokens / (result.decode_ms / 1000.0) : 0.0;
        }
        result.stop_reason = stop_reason;
    }

    std::unique_ptr<LLMModel>  model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    bool                       ready_ = false;
};

// Top-level factory. Defaults `tokenizer_config_path` to
// `<bundle>/tokenizer_config.json` (where the bundle is derived from
// `model_cfg.model_paths`). Returns std::nullopt and logs on failure.
inline std::optional<Pipeline> makePipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg, const std::string& tokenizer_config_path = "") {
    try {
        std::string tcp = tokenizer_config_path;
        if (tcp.empty()) {
            tcp = (bundleDirOf(model_cfg) / "tokenizer_config.json").string();
        }
        Pipeline pipe;
        if (!pipe.create(makeModel(model_cfg), runtime_cfg, model_cfg, tcp)) return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("auto_llm::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace auto_llm
}  // namespace geniex

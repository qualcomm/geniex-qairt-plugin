// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/vlm_pipeline.h"

#include <chrono>
#include <cstring>
#include <sstream>
#include <utility>

#include "geniex-proc/processor.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace geniex {

namespace {

using Clock = std::chrono::high_resolution_clock;

PixelData toPixelData(const BatchFeatures& bf) {
    PixelData pd;

    // Two mutually exclusive geometry descriptions, one per encoder family:
    //   * grid-based (Qwen2.5-VL / Qwen3-VL / InternVL) report image_grid_thw;
    //   * patch-budget (Gemma4, SigLIP2) pad to a fixed patch count and report
    //     per-patch (x, y) ids instead, leaving image_grid_thw empty.
    // Keying "no image" on image_grid_thw alone silently dropped every
    // patch-budget image, so accept either descriptor.
    const bool has_grid = bf.image_grid_thw.dimension() > 0 && bf.image_grid_thw.shape()[0] > 0;
    const bool has_pos  = bf.image_position_ids.dimension() > 0 && bf.image_position_ids.shape()[0] > 0;
    if (!has_grid && !has_pos) {
        return pd;
    }
    pd.pixel_values.assign(bf.pixel_values.cbegin(), bf.pixel_values.cend());

    if (has_grid) {
        const size_t n = bf.image_grid_thw.shape()[0];
        pd.image_grid_thw.resize(n);
        for (size_t i = 0; i < n; ++i) {
            pd.image_grid_thw[i] = {
                static_cast<int32_t>(bf.image_grid_thw(i, 0)),
                static_cast<int32_t>(bf.image_grid_thw(i, 1)),
                static_cast<int32_t>(bf.image_grid_thw(i, 2)),
            };
        }
    }

    if (has_pos) {
        pd.image_position_ids.assign(bf.image_position_ids.cbegin(), bf.image_position_ids.cend());
    }
    pd.num_soft_tokens_per_image = bf.num_soft_tokens_per_image;

    return pd;
}

// Populate `result` from in-flight generation state.
void finalize_generate_result(GenerateResult& result, std::ostringstream& full_text, int64_t generated_tokens,
    Clock::time_point t_start, Clock::time_point t_first_token, Clock::time_point t_end, bool got_first,
    const char* stop_reason) {
    result.full_text = full_text.str();

    // Align with Genie's `num-generated-tokens`, which counts the terminating EOS
    // sample. The decode loop stops before emitting EOS into the text, so add it
    // back to the reported count (not to full_text) when generation ended on EOS.
    const bool ended_on_eos = stop_reason != nullptr && std::strcmp(stop_reason, "eos") == 0;
    result.generated_tokens = generated_tokens + (ended_on_eos ? 1 : 0);

    if (got_first) {
        result.ttft_ms   = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
        result.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();

        // Genie divides the EOS-inclusive token count by the decode window.
        result.tokens_per_second = result.decode_ms > 0.0 ? result.generated_tokens / (result.decode_ms / 1000.0) : 0.0;
    }
    result.stop_reason = stop_reason;
}

}  // namespace

struct VLMPipeline::Impl {
    std::unique_ptr<VLMModel>        model;
    std::unique_ptr<VisionProcessor> processor;
    Tokenizer*                       tokenizer    = nullptr;
    bool                             ready        = false;
    int32_t                          bos_token_id = -1;
};

VLMPipeline::VLMPipeline() : impl_(std::make_unique<Impl>()) {}
VLMPipeline::~VLMPipeline()                                 = default;
VLMPipeline::VLMPipeline(VLMPipeline&&) noexcept            = default;
VLMPipeline& VLMPipeline::operator=(VLMPipeline&&) noexcept = default;

bool VLMPipeline::create(
    std::unique_ptr<VLMModel> model, std::unique_ptr<VisionProcessor> processor, Tokenizer& tokenizer) {
    if (!model || !processor) return false;

    impl_->model     = std::move(model);
    impl_->processor = std::move(processor);
    impl_->tokenizer = &tokenizer;
    impl_->ready     = true;
    return true;
}

bool VLMPipeline::isReady() const { return impl_ && impl_->ready; }

void VLMPipeline::reset() {
    if (impl_->model) impl_->model->resetKVCache();
}

void VLMPipeline::setBosTokenId(int32_t token_id) { impl_->bos_token_id = token_id; }

std::string VLMPipeline::applyChatTemplate(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    return impl_->processor->apply_chat_template(messages, add_generation_prompt);
}

GenerateResult VLMPipeline::generate(
    const std::string& formatted_prompt, const GenerationConfig& gen_cfg, std::function<bool(const char*)> on_token) {
    return generate(formatted_prompt, /*image_paths=*/{}, gen_cfg, std::move(on_token));
}

GenerateResult VLMPipeline::generate(const std::string& formatted_prompt, const std::vector<std::string>& image_paths,
    const GenerationConfig& gen_cfg, std::function<bool(const char*)> on_token) {
    GenerateResult result;
    if (!impl_->ready || !impl_->model || !impl_->processor || !impl_->tokenizer) {
        result.stop_reason = "error";
        return result;
    }

    VLMInput             vlm_input;
    std::vector<int32_t> prompt_tokens;

    try {
        BatchFeatures bf     = impl_->processor->process(formatted_prompt, image_paths);
        vlm_input.pixel_data = toPixelData(bf);
        prompt_tokens.assign(bf.input_ids.cbegin(), bf.input_ids.cend());
    } catch (...) {
        result.stop_reason = "error";
        return result;
    }

    // Prepend BOS only on the first turn
    if (impl_->bos_token_id >= 0 && impl_->model->nPast() == 0 &&
        (prompt_tokens.empty() || prompt_tokens.front() != impl_->bos_token_id)) {
        prompt_tokens.insert(prompt_tokens.begin(), impl_->bos_token_id);
    }

    result.prompt_tokens = static_cast<int64_t>(prompt_tokens.size());

    // Inject the tokenizer; LLMModel needs it for grammar/EOG resolution.
    GenerationConfig effective_cfg = gen_cfg;
    effective_cfg.tokenizer        = impl_->tokenizer;

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
        auto output_tokens = impl_->model->generate(prompt_tokens, vlm_input, effective_cfg, on_each_token);
        auto t_end         = Clock::now();

        const int64_t total  = static_cast<int64_t>(output_tokens.size());
        const char*   reason = user_stopped ? "user" : (total >= gen_cfg.max_tokens ? "length" : "eos");
        finalize_generate_result(result, full_text, total, t_start, t_first_token, t_end, got_first, reason);
        result.media_ms = impl_->model->lastMediaMs();
        return result;
    } catch (const ContextLengthExceededError&) {
        const auto t_end = Clock::now();
        finalize_generate_result(
            result, full_text, streamed_tokens, t_start, t_first_token, t_end, got_first, "context_length");
        return result;
    }
}

void VLMPipeline::saveKVCache(const std::string& path) const {
    if (impl_->model) impl_->model->saveKVCacheToFile(path);
}

void VLMPipeline::loadKVCache(const std::string& path) {
    if (impl_->model) impl_->model->loadKVCacheFromFile(path);
}

size_t VLMPipeline::nPast() const { return impl_->model ? impl_->model->nPast() : 0; }

}  // namespace geniex

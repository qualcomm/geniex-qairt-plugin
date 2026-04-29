// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/vlm_pipeline.h"

#include <chrono>
#include <sstream>
#include <utility>

#include "geniex-proc/processor.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace geniex {

namespace {

PixelData toPixelData(const BatchFeatures& bf) {
    PixelData pd;
    if (bf.image_grid_thw.dimension() == 0 ||
        bf.image_grid_thw.shape()[0] == 0) {
        return pd;
    }
    pd.pixel_values.assign(bf.pixel_values.cbegin(), bf.pixel_values.cend());

    const size_t n = bf.image_grid_thw.shape()[0];
    pd.image_grid_thw.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pd.image_grid_thw[i] = {
            static_cast<int32_t>(bf.image_grid_thw(i, 0)),
            static_cast<int32_t>(bf.image_grid_thw(i, 1)),
            static_cast<int32_t>(bf.image_grid_thw(i, 2)),
        };
    }
    return pd;
}

} // namespace


struct VLMPipeline::Impl {
    std::unique_ptr<VLMModel>         model;
    std::unique_ptr<VisionProcessor>  processor;
    Tokenizer*                        tokenizer = nullptr;
    bool                              ready     = false;
};

VLMPipeline::VLMPipeline()  : impl_(std::make_unique<Impl>()) {}
VLMPipeline::~VLMPipeline() = default;
VLMPipeline::VLMPipeline(VLMPipeline&&) noexcept            = default;
VLMPipeline& VLMPipeline::operator=(VLMPipeline&&) noexcept = default;

bool VLMPipeline::create(std::unique_ptr<VLMModel>        model,
                         std::unique_ptr<VisionProcessor> processor,
                         Tokenizer&                       tokenizer) {
    if (!model || !processor) return false;

    impl_->model     = std::move(model);
    impl_->processor = std::move(processor);
    impl_->tokenizer = &tokenizer;
    impl_->ready     = true;
    return true;
}

bool VLMPipeline::isReady() const {
    return impl_ && impl_->ready;
}

void VLMPipeline::reset() {
    if (impl_->model) impl_->model->resetKVCache();
}

std::string VLMPipeline::applyChatTemplate(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt) const
{
    return impl_->processor->apply_chat_template(messages, add_generation_prompt);
}

GenerateResult VLMPipeline::generate(
    const std::string&               formatted_prompt,
    const GenerationConfig&          gen_cfg,
    std::function<bool(const char*)> on_token)
{
    return generate(formatted_prompt, /*image_paths=*/{}, gen_cfg, std::move(on_token));
}

GenerateResult VLMPipeline::generate(
    const std::string&               formatted_prompt,
    const std::vector<std::string>&  image_paths,
    const GenerationConfig&          gen_cfg,
    std::function<bool(const char*)> on_token)
{
    GenerateResult result;
    if (!impl_->ready || !impl_->model || !impl_->processor || !impl_->tokenizer) {
        result.stop_reason = "error";
        return result;
    }

    VLMInput             vlm_input;
    std::vector<int32_t> prompt_tokens;

    try {
        BatchFeatures bf = impl_->processor->process(formatted_prompt, image_paths);
        vlm_input.pixel_data = toPixelData(bf);
        prompt_tokens.assign(bf.input_ids.cbegin(), bf.input_ids.cend());
    } catch (...) {
        result.stop_reason = "error";
        return result;
    }

    result.prompt_tokens = static_cast<int64_t>(prompt_tokens.size());

    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();
    Clock::time_point t_first_token;
    bool got_first    = false;
    bool user_stopped = false;

    std::ostringstream full_text;
    std::vector<int32_t> output_tokens;

    try {
        output_tokens = impl_->model->generate(
            prompt_tokens,
            vlm_input,
            gen_cfg,
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
    } catch (...) {
        result.stop_reason = "error";
        return result;
    }

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

void VLMPipeline::saveKVCache(const std::string& path) const {
    if (impl_->model) impl_->model->saveKVCacheToFile(path);
}

void VLMPipeline::loadKVCache(const std::string& path) {
    if (impl_->model) impl_->model->loadKVCacheFromFile(path);
}

size_t VLMPipeline::nPast() const {
    return impl_->model ? impl_->model->nPast() : 0;
}

} // namespace geniex

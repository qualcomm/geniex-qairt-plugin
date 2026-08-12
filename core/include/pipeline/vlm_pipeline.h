// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "geniex-proc/types.h"  // ChatMessage
#include "geniex_export.h"
#include "pipeline/llm_pipeline.h"  // GenerateResult
#include "types.h"
#include "vlm/vlm_model.h"
#include "vlm/vlm_types.h"

namespace geniex {
class VisionProcessor;
class Tokenizer;
}  // namespace geniex

namespace geniex {

// High-level wrapper that pairs a VLMModel with a VisionProcessor (which owns
// the tokenizer and chat template) and exposes a generate / reset workflow
// analogous to LLMPipeline.
class GENIEX_VLM_API VLMPipeline {
   public:
    VLMPipeline();
    ~VLMPipeline();

    VLMPipeline(VLMPipeline&&) noexcept;
    VLMPipeline& operator=(VLMPipeline&&) noexcept;
    VLMPipeline(const VLMPipeline&)            = delete;
    VLMPipeline& operator=(const VLMPipeline&) = delete;

    bool create(std::unique_ptr<VLMModel> model, std::unique_ptr<VisionProcessor> processor, Tokenizer& tokenizer);

    // Polymorphic overload: accepts any VLMModel subclass without slicing.
    template <typename ModelT, std::enable_if_t<std::is_base_of_v<VLMModel, ModelT>, int> = 0>
    bool create(std::unique_ptr<ModelT> model, std::unique_ptr<VisionProcessor> processor, Tokenizer& tokenizer) {
        return create(std::unique_ptr<VLMModel>(model.release()), std::move(processor), tokenizer);
    }

    bool isReady() const;

    // Clears KV state and resets to the start of a new conversation.
    void reset();

    // Prepends `token_id` once on the first turn.
    // Pass -1 (default) to disable BOS prepending.
    void setBosTokenId(int32_t token_id);

    // Formats messages into a prompt string using the processor's chat template.
    // Pure text — no image I/O is performed. Each mm_content entry in each
    // message is replaced by the processor's image marker.
    std::string applyChatTemplate(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;

    // The text sentinel applyChatTemplate() emits in place of each image, and
    // that generate() pairs positionally with `image_paths`.
    //
    // Exposed for callers that assemble a continuation turn by hand instead of
    // re-rendering the whole transcript: with a stateful KV cache only the new
    // turn is fed, so such a caller must emit this marker itself to attach an
    // image to that turn.
    const std::string& imageMarker() const;

    // Run inference on a pre-formatted prompt (output of applyChatTemplate).
    // `image_paths` must match the image marker occurrences in `formatted_prompt`.
    // on_token is called with each decoded text piece; return false to stop early.
    GenerateResult generate(const std::string& formatted_prompt, const std::vector<std::string>& image_paths = {},
        const GenerationConfig& gen_cfg = {}, std::function<bool(const char*)> on_token = nullptr);

    // Text-only convenience overload.
    GenerateResult generate(const std::string& formatted_prompt, const GenerationConfig& gen_cfg,
        std::function<bool(const char*)> on_token = nullptr);

    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geniex

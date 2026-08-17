// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "geniex_export.h"
#include "llm/llm_model.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_input_provider.h"
#include "vlm/vlm_types.h"

namespace geniex {

// LLMModel subclass that encodes multimodal inputs and injects their embeddings
// before calling the LLM generate loop.
class GENIEX_VLM_API VLMModel : public LLMModel {
   public:
    explicit VLMModel(LLMSpec spec);

    // Variant for families that need genie_config.json at construction time --
    // e.g. Gemma4, whose per-layer embedding stream and dual RoPE are described
    // there and cannot be inferred from the graphs alone.
    VLMModel(LLMSpec spec, ParsedGenieConfig gc);

    // Return false from the callback to stop generation early.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input,
        const GenerationConfig& gen_cfg = {}, std::function<bool(int32_t)> token_callback = nullptr);

    // encodeVision() wall time (ms) from the last generate(); 0 if no media.
    double lastMediaMs() const { return last_media_ms_; }

   protected:
    bool onInitialized() override;

    virtual std::vector<float> encodeVision(const PixelData& pixel_data) = 0;

    // Makes this turn's embeddings visible to the decoder, and tears that down
    // again once generate() returns.
    //
    // The default pair covers the common case: a float32 table in RAM, looked up
    // up-front and scattered into. Families whose table cannot be materialised
    // that way override both -- Gemma4's LUTs are quantized and several GB, so
    // they stay memory-mapped and the vision rows are layered on as a positional
    // override instead.
    //
    // Absolute prompt positions start at nPast(), which is still the pre-prefill
    // value when these are called.
    virtual void prepareEmbeddings(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input);
    virtual void releaseEmbeddings();

    // Called after embedding injection, before LLMModel::generate().
    virtual void preparePositions(const std::vector<int32_t>& input_ids, const VLMInput& vlm_input, size_t n_past);

    // Called after LLMModel::generate() returns to reset position provider state.
    virtual void clearPositions();

    // Overwrites rows in input_embeds where input_ids == target_token_id
    // with consecutive rows from multimodal_embeds.
    static void maskedScatter(std::vector<float>& input_embeds, const std::vector<float>& multimodal_embeds,
        const std::vector<int32_t>& input_ids, int32_t target_token_id, size_t hidden_size);

    // Must be called before initialize().
    void setEmbeddingProvider(std::unique_ptr<PrecomputedEmbeddingProvider> provider);

    std::unique_ptr<VisionEncoder> vision_encoder_;

    int32_t image_token_id_ = 0;

    // encodeVision() wall time (ms) for the current generate(); reset to 0 at
    // its top. A subclass overriding prepareEmbeddings() must accumulate here.
    double last_media_ms_ = 0.0;

   private:
    PrecomputedEmbeddingProvider* emb_provider_ = nullptr;  // non-owning; owned by input_providers_
};

}  // namespace geniex

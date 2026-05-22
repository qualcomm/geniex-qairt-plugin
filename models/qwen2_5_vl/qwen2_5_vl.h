// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "geniex-proc/qwen2vl.h"
#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "pipeline/chat_template.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_input_provider.h"
#include "vlm/vlm_model.h"

namespace geniex {
namespace qwen2_5_vl {

// ── Vision-tower internals (not exposed in the bundle's config.json) ─────────
// These describe the qwen2-vl ViT itself, not the LLM tower; they are the same
// across every Qwen2.5-VL bundle and would only change if the family changed.
static constexpr int   kVitWindowSize = 112;  // pixels
static constexpr int   kVitRopeDim    = 40;   // half_dim*2 = cos/sin emb dim
static constexpr float kVitRopeTheta  = 10000.0f;

// Vision-related token IDs. These are family-level constants because the
// runtime no longer reads HuggingFace config.json, and current Qwen2.5-VL
// `genie_config.json` files don't carry them either.
//   <|vision_start|> 151652   <|vision_end|> 151653   <|image_pad|> 151655
static constexpr int32_t kVisionStartTokenId = 151652;
static constexpr int32_t kImageTokenId       = 151655;

// Single-graph QNN vision encoder for Qwen2.5-VL. Inputs/outputs match the
// graph schema declared in the bundle's metadata.json `vision_encoder.bin`.
class Qwen25VLVisionEncoder : public QnnVisionEncoder {
   public:
    // Configures the encoder with vision-preprocessing parameters from the
    // bundle's metadata.json. Must be called before initialize().
    void setPreprocessing(const ParsedVisionPreprocessing& vp);

    // LLM hidden size — needed to assemble per-image output buffers. Set from
    // ParsedQAIRTMetadata.hidden_size before initialize().
    void setHiddenSize(size_t hidden) { hidden_size_ = hidden; }

    std::vector<float> encode(const PixelData& pixel_data) override;

   private:
    int    image_width_         = 0;
    int    image_height_        = 0;
    int    patch_size_          = 0;
    int    temporal_patch_size_ = 0;
    int    spatial_merge_size_  = 0;
    size_t hidden_size_         = 0;
};

class Qwen25VLModel : public VLMModel {
   public:
    // Builds an empty model from an LLMSpec; vision encoder, MRoPE provider,
    // and image-token-ID are wired in by makeModel().
    explicit Qwen25VLModel(LLMSpec spec);

    void setVisionEncoder(std::unique_ptr<Qwen25VLVisionEncoder> vis);
    void setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider);
    void setVisionTokenIds(int32_t vision_start, int32_t image_pad);
    void setSpatialMergeSize(int sms) { spatial_merge_size_ = sms; }

   protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override;

    void preparePositions(const std::vector<int32_t>& input_ids, const VLMInput& vlm_input, size_t n_past) override;

    void clearPositions() override;

   private:
    MRoPEInputProvider*  mrope_provider_     = nullptr;  // non-owning
    std::vector<int32_t> mrope_deltas_       = {0, 0, 0};
    int32_t              vision_start_token_ = 0;
    int                  spatial_merge_size_ = 2;
};

// Full Qwen2.5-VL stack (vision encoder + LLM). Returns nullptr on failure.
GENIEX_VLM_API std::unique_ptr<Qwen25VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

// Convenience factory: builds the full pipeline (vision encoder + LLM + processor)
// from a runtime config and a model config. Image dimensions for the processor
// come from the bundle's metadata.json `vision_preprocessing` block.
inline std::optional<VLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    auto model = makeModel(runtime_cfg, config);
    if (!model) return std::nullopt;

    const auto bundle = bundleDirOf(config.llm_config);
    auto       meta   = parseQAIRTMetadata(bundle);
    if (!meta.vision_preprocessing) return std::nullopt;

    qwen2vl::Qwen2VLConfig proc_cfg;
    proc_cfg.fixed_height = meta.vision_preprocessing->image_height;
    proc_cfg.fixed_width  = meta.vision_preprocessing->image_width;

    auto processor = qwen2vl::Qwen2VLProcessor::create(config.llm_config.tokenizer_path, proc_cfg);
    if (!processor) return std::nullopt;

    Tokenizer& tok = processor->tokenizer();

    VLMPipeline pipe;
    if (!pipe.create(std::move(model), std::move(processor), tok)) return std::nullopt;
    return pipe;
}

}  // namespace qwen2_5_vl
}  // namespace geniex

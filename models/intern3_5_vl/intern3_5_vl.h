// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "geniex-proc/internvl.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_input_provider.h"
#include "vlm/vlm_model.h"

namespace geniex {
namespace intern3_5_vl {

// ── Vision-related token IDs ─────────────────────────────────────────────────
// <IMG_CONTEXT> is the placeholder whose embeddings the ViT output replaces; the
// Qwen-style delimiters wrap the run.
static constexpr int32_t kVisionStartTokenId = 151652;
static constexpr int32_t kVisionEndTokenId   = 151653;
static constexpr int32_t kImageTokenId       = 151671;

// Single-graph QNN vision encoder for InternVL3.5.
//
// This export bakes the vision-tower position encoding and attention masks into
// the graph, so the only input is the image itself:
//   pixel_values   [1, 3, H, W]                    (planar CHW, normalized)
//   image_features [1, num_image_tokens, hidden]
class Intern35VLVisionEncoder : public QnnVisionEncoder {
   public:
    // Configures the encoder with vision-preprocessing parameters from the
    // bundle's metadata.json. Must be called before encode().
    void setPreprocessing(const ParsedVisionPreprocessing& vp);

    // LLM hidden size — needed to size per-image output buffers. Set from
    // ParsedQAIRTMetadata.hidden_size before encode().
    void setHiddenSize(size_t hidden) { hidden_size_ = hidden; }

    std::vector<float> encode(const PixelData& pixel_data) override;

   private:
    int    image_width_        = 0;
    int    image_height_       = 0;
    int    patch_size_         = 0;
    int    spatial_merge_size_ = 0;
    size_t hidden_size_        = 0;
};

// InternVL3.5 (Qwen3 text tower + InternViT).
//
// The exported LLM shards take plain 1-D `position_ids_cos/sin` (half_dim =
// head_dim/2) rather than 3-D MRoPE, so positions need no per-turn state —
// nothing for preparePositions()/clearPositions() to do.
//
// The RoPE provider is registered by makeModel() rather than by
// LLMModel::createInputProviders(), which is skipped once input_providers_ is
// non-empty (VLMModel's constructor already registers the embedding provider).
class Intern35VLModel : public VLMModel {
   public:
    // Builds an empty model from an LLMSpec; the vision encoder, RoPE provider
    // and image-token ID are wired in by makeModel().
    explicit Intern35VLModel(LLMSpec spec);

    void setVisionEncoder(std::unique_ptr<Intern35VLVisionEncoder> vis);
    void setImageTokenId(int32_t image_token) { image_token_id_ = image_token; }

   protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override;
};

// Full InternVL3.5 stack (vision encoder + LLM). Returns nullptr on failure.
GENIEX_VLM_API std::unique_ptr<Intern35VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

// Convenience factory: builds the full pipeline (vision encoder + LLM +
// processor). Preprocessing geometry and normalization constants come from the
// bundle's metadata.json `vision_preprocessing` block — notably the mean/std,
// which are ImageNet values here, not CLIP values.
inline std::optional<VLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    auto model = makeModel(runtime_cfg, config);
    if (!model) return std::nullopt;

    const auto bundle = bundleDirOf(config.llm_config);
    auto       meta   = parseQAIRTMetadata(bundle);
    auto       gc     = parseGenieConfig(bundle);
    if (!meta.vision_preprocessing) return std::nullopt;

    const auto& vp = *meta.vision_preprocessing;

    internvl::InternVLConfig proc_cfg;
    proc_cfg.image_size         = vp.image_width;
    proc_cfg.patch_size         = vp.patch_size;
    proc_cfg.spatial_merge_size = vp.spatial_merge_size;
    if (vp.normalize_mean.size() == 3) proc_cfg.image_mean = vp.normalize_mean;
    if (vp.normalize_std.size() == 3) proc_cfg.image_std = vp.normalize_std;

    auto processor = internvl::InternVLProcessor::create(config.llm_config.tokenizer_path, proc_cfg);
    if (!processor) return std::nullopt;

    Tokenizer& tok = processor->tokenizer();

    VLMPipeline pipe;
    if (!pipe.create(std::move(model), std::move(processor), tok)) return std::nullopt;

    // Not calling pipe.setBosTokenId(): this is a ChatML text tower, whose
    // prompts must start directly with `<|im_start|>`. The bundle advertises
    // bos-token 151643 (`<|endoftext|>`), but that is a padding/EOS id, not a
    // sequence-start marker; prepending it shifts every position by one.
    (void)gc;
    return pipe;
}

}  // namespace intern3_5_vl
}  // namespace geniex

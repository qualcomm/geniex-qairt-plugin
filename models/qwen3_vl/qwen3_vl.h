// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "geniex-proc/qwen2vl.h"
#include "llm/input_provider.h"
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
namespace qwen3_vl {

// ── Vision-tower internals (not exposed in the bundle's config.json) ─────────
// Qwen3-VL's ViT runs full (non-windowed) attention over the patches of a
// single image, applies 2-D rotary position embeddings, and emits both the
// merged image features and the intermediate "DeepStack" features. The RoPE
// cos/sin width comes from the vision_encoder.bin schema (position_ids_cos
// has last-dim 32 ⇒ kVitRopeDim = 32).
static constexpr int   kVitRopeDim   = 32;
static constexpr float kVitRopeTheta = 10000.0f;

// Vision-related token IDs. Family-level constants — the runtime no longer
// reads HuggingFace config.json and the bundle's genie_config.json doesn't
// carry them. Shared with the Qwen2.x VL family.
//   <|vision_start|> 151652   <|vision_end|> 151653   <|image_pad|> 151655
static constexpr int32_t kVisionStartTokenId = 151652;
static constexpr int32_t kImageTokenId       = 151655;

// Single-graph QNN vision encoder for Qwen3-VL. In addition to the merged
// image features it produces the per-level DeepStack visual embeddings, which
// the model injects into the first decoder shard.
class Qwen3VLVisionEncoder : public QnnVisionEncoder {
   public:
    // embeds_name_prefix: DeepStack output tensor name prefix (tensor k is
    // "<prefix><k>"). Must match the prefix given to DeepstackInputProvider so
    // the ViT read side and the decoder write side stay in sync.
    explicit Qwen3VLVisionEncoder(std::string embeds_name_prefix = "deepstack_visual_embeds_")
        : deepstack_prefix_(std::move(embeds_name_prefix)) {}

    // Configures the encoder with vision-preprocessing parameters from the
    // bundle's metadata.json. Must be called before initialize().
    void setPreprocessing(const ParsedVisionPreprocessing& vp);

    // LLM hidden size — needed to assemble per-image output buffers. Set from
    // ParsedQAIRTMetadata.hidden_size before initialize().
    void setHiddenSize(size_t hidden) { hidden_size_ = hidden; }

    // Detects the number of DeepStack output tensors once the graphs are ready.
    bool initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) override;

    // Returns the merged image features, flat [num_image_tokens * hidden_size].
    std::vector<float> encode(const PixelData& pixel_data) override;

    // Per-level DeepStack embeddings captured during the most recent encode(),
    // each flat [num_image_tokens * hidden_size] concatenated across images.
    const std::vector<std::vector<float>>& deepstackEmbeds() const { return deepstack_embeds_; }

   private:
    int    image_width_         = 0;
    int    image_height_        = 0;
    int    patch_size_          = 0;
    int    temporal_patch_size_ = 0;
    int    spatial_merge_size_  = 0;
    size_t hidden_size_         = 0;

    // DeepStack output tensor name prefix (tensor k is "<prefix><k>"); set by ctor.
    std::string deepstack_prefix_;

    // Number of deepstack output tensors on the vision graph (detected in initialize()).
    size_t                          num_deepstack_levels_ = 0;
    std::vector<std::vector<float>> deepstack_embeds_;
};

class Qwen3VLModel : public VLMModel {
   public:
    // Builds an empty model from an LLMSpec; vision encoder, MRoPE provider,
    // deepstack provider, and image-token-ID are wired in by makeModel().
    explicit Qwen3VLModel(LLMSpec spec);

    // Non-copyable / non-movable: holds raw observer pointers into objects it
    // owns (vision encoder) or into the InputProvider list. A copy or move would
    // leave those dangling.
    Qwen3VLModel(const Qwen3VLModel&)            = delete;
    Qwen3VLModel& operator=(const Qwen3VLModel&) = delete;
    Qwen3VLModel(Qwen3VLModel&&)                 = delete;
    Qwen3VLModel& operator=(Qwen3VLModel&&)      = delete;

    void setVisionEncoder(std::unique_ptr<Qwen3VLVisionEncoder> vis);
    void setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider);
    void setDeepstackProvider(std::unique_ptr<DeepstackInputProvider> provider);
    void setVisionTokenIds(int32_t vision_start, int32_t image_pad);
    void setSpatialMergeSize(int sms) { spatial_merge_size_ = sms; }

    // Also clears MRoPE deltas and DeepStack state so a reset after a
    // mid-prefill error doesn't leave stale positions/embeddings behind.
    void resetKVCache() override;

   protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override;

    void preparePositions(const std::vector<int32_t>& input_ids, const VLMInput& vlm_input, size_t n_past) override;

    void clearPositions() override;

   private:
    Qwen3VLVisionEncoder*   vision_encoder_raw_ = nullptr;  // non-owning view into vision_encoder_
    MRoPEInputProvider*     mrope_provider_     = nullptr;  // non-owning
    DeepstackInputProvider* deepstack_provider_ = nullptr;  // non-owning
    std::vector<int32_t>    mrope_deltas_       = {0, 0, 0};
    int32_t                 vision_start_token_ = 0;
    int                     spatial_merge_size_ = 2;
};

// Full Qwen3-VL stack (vision encoder + LLM). Returns nullptr on failure.
GENIEX_VLM_API std::unique_ptr<Qwen3VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

// Convenience factory: builds the full pipeline (vision encoder + LLM + processor)
// from a runtime config and a model config. Image dimensions for the processor
// come from the bundle's metadata.json `vision_preprocessing` block.
inline std::optional<VLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    auto model = makeModel(runtime_cfg, config);
    if (!model) return std::nullopt;

    const auto bundle = bundleDirOf(config.llm_config);
    auto       meta   = parseQAIRTMetadata(bundle);
    if (!meta.vision_preprocessing) {
        GENIEX_LOG_ERROR("qwen3_vl::makePipeline: bundle has no vision_preprocessing block");
        return std::nullopt;
    }

    qwen2vl::Qwen2VLConfig proc_cfg;
    proc_cfg.fixed_height        = meta.vision_preprocessing->image_height;
    proc_cfg.fixed_width         = meta.vision_preprocessing->image_width;
    proc_cfg.patch_size          = meta.vision_preprocessing->patch_size;
    proc_cfg.temporal_patch_size = meta.vision_preprocessing->temporal_patch_size;
    proc_cfg.merge_size          = meta.vision_preprocessing->spatial_merge_size;
    if (!meta.vision_preprocessing->normalize_mean.empty())
        proc_cfg.image_mean = meta.vision_preprocessing->normalize_mean;
    if (!meta.vision_preprocessing->normalize_std.empty())
        proc_cfg.image_std = meta.vision_preprocessing->normalize_std;

    auto processor = qwen2vl::Qwen2VLProcessor::create(config.llm_config.tokenizer_path, proc_cfg);
    if (!processor) return std::nullopt;

    Tokenizer& tok = processor->tokenizer();

    VLMPipeline pipe;
    if (!pipe.create(std::move(model), std::move(processor), tok)) return std::nullopt;
    return pipe;
}

}  // namespace qwen3_vl
}  // namespace geniex

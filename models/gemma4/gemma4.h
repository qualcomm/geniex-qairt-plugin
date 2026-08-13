// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 (E2B / E4B), multimodal.
//
// A Gemma4 bundle that ships a vision encoder context binary is a VLM, so the
// multimodal stack lives here beside the decoder instead of in its own header:
// one model family, one header, matching qwen2_5_vl / qwen3_vl / intern3_5_vl.
//
// Every published `gemma_4_*` bundle ships a vision_encoder context binary, so
// the family is always multimodal and there is no text-only decoder here. A
// text-only prompt is simply a turn with no mm_content, which VLMPipeline
// already handles (prepareEmbeddings() returns early when pixel_values is
// empty), so one class covers both uses.
//
// Declarations only; every body lives in gemma4.cpp. The provider wiring, the
// bundle-path resolution and the graph tensor names are implementation details
// of that translation unit and are deliberately not exposed here.
//
// Contents:
//   Gemma4VisionEncoder     the Visual Embedding Generator (VEG) graph
//   Gemma4VLMModel          the decoder            -> makeVLMPipeline()

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "llm/input_provider.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_model.h"
#include "vlm/vlm_types.h"

namespace geniex {
namespace gemma4 {

// ═════════════════════════════════════════════════════════════════════════════
// Vision encoder
// ═════════════════════════════════════════════════════════════════════════════

// Gemma4 Visual Embedding Generator (VEG).
//
// A single-graph QNN model: patch_embedder -> 16-layer ViT encoder -> pooler ->
// embed_vision. It consumes the patch-budget tensors produced by
// geniex::gemma4::Gemma4Processor and emits soft tokens already in the decoder's
// embedding space, so its output can be spliced straight into `inputs_embeds`.
//
//   pixel_values       [1, max_patches, patch_size^2 * 3]  float32
//   image_position_ids [1, max_patches, 2]                 int32   (x, y); -1 = pad
//   vision_embedding   [1, n_soft_tokens, hidden]          float32
//
// The exported graph fixes n_soft_tokens (256 for the v73 export), which is why
// the processor pins a square geometry — see Gemma4Config::force_square_size.
class Gemma4VisionEncoder : public QnnVisionEncoder {
   public:
    bool initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) override;

    // Geometry read off the loaded graph; zero until initialize() succeeds.
    // makeVLMModel() cross-checks these against the processor's config.
    size_t maxPatches() const { return max_patches_; }
    size_t patchDim() const { return patch_dim_; }
    size_t numSoftTokens() const { return n_soft_tokens_; }
    size_t hiddenSize() const { return hidden_size_; }

    // Runs every image in the batch and concatenates their soft tokens, so the
    // result lines up with the prompt's image-token run in order. Gemma4 is a
    // patch-budget encoder, so geometry comes from image_position_ids and
    // image_grid_thw is always empty.
    std::vector<float> encode(const PixelData& pixel_data) override;

   private:
    // One VEG pass over a single image. Callers validate the buffer extents.
    void runOne(const float* pixel_values, const int32_t* image_position_ids, float* out);

    size_t max_patches_   = 0;
    size_t patch_dim_     = 0;
    size_t n_soft_tokens_ = 0;
    size_t hidden_size_   = 0;
};

// ═════════════════════════════════════════════════════════════════════════════
// Decoder
// ═════════════════════════════════════════════════════════════════════════════

// Gemma3/4 differs from a Llama-style decoder in four ways, all handled here +
// in the core runtime (see llm_model.cpp / input_provider.cpp / llm_utils.cpp):
//   1. Per-layer input embedding — a SECOND embedding stream (`per_layer_inputs`)
//      from its own LUT, injected in addition to `inputs_embeds`.
//   2. Dual RoPE — global layers use `position_ids_*` (partial-rotary, big theta),
//      local/sliding-window layers use `swa_position_ids_*` (full-rotary, small
//      theta). Base createInputProviders() installs the global one; this subclass
//      adds the local one.
//   3. Sliding-window (local) attention — a second `swa_attention_mask` and a
//      second `swa_*` KV cache. Both are handled generically by the core runtime
//      (get_sliding_window_mask + multi-KV-block updateKV), auto-detected from
//      the graph tensors.
//   4. Partial-rotary / proportional RoPE — parsed from genie_config.json's
//      rope-scaling (rope-type "proportional"), reused via PartialRoPEProvider.
//
// Beyond those, the class' job is to append the extra CPU-side providers the
// base factory doesn't know about (per-layer embedding + local RoPE) and to
// splice the vision encoder's output in; everything else (shards, both KV
// blocks, both masks, global RoPE, main embedding) is inferred from the graphs
// and genie_config.json exactly like the other families.
//
// Gemma4 cannot use VLMModel's default embedding path. That path assumes a
// PrecomputedEmbeddingProvider: a float32 table held in RAM, looked up in bulk
// for the whole prompt and then scattered into. Gemma4's tables are quantized
// ufixed16 and very large (E4B: 1.28 GB main + 5.4 GB per-layer), so they stay
// memory-mapped behind EmbeddingInputProvider and are converted per row as each
// chunk is written. This class therefore overrides prepareEmbeddings() /
// releaseEmbeddings() and expresses the same effect positionally:
//
//   * main stream      -> setEmbeddingOverride(image_positions, vision_rows)
//   * per-layer stream -> setTokenSubstitution(image_positions, pad_token_id)
//
// Both take the full list of image-token positions, not a single span: a prompt
// with several attachments carries one run per image, separated by the chat
// template's end-of-image / begin-of-image text.
//
// The second half is not optional. `per_layer_inputs` is a plain LUT lookup on
// the input ids, but Gemma4 rewrites multimodal positions to the pad id before
// that lookup:
//     llm_input_ids = where(multimodal_mask, pad_token_id, llm_input_ids)
// Letting the image token's own row through corrupts the per-layer input at
// every layer for every image position, and the model then answers as though no
// image were attached ("Please provide the image you are referring to").
// Verified on the Genie reference pipeline: image-token row and dequantized-zero
// both refuse; the PAD row describes the image.
class Gemma4VLMModel : public VLMModel {
   public:
    Gemma4VLMModel(LLMSpec spec, ParsedGenieConfig gc, std::filesystem::path bundle_dir);

    void setVisionEncoder(std::unique_ptr<Gemma4VisionEncoder> vis);
    void setImageTokenId(int32_t image_token);

   protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override;

    // Appends the extra providers the decoder needs beyond the base factory's.
    // setEmbeddingProvider() is deliberately never called, so
    // LLMModel::createInputProviders() still installs the quantized
    // memory-mapped `inputs_embeds` provider rather than VLMModel's in-RAM
    // default, and we keep a handle on it for the splice.
    void createInputProviders() override;

    // Runs the encoder and installs its rows as a positional override on the two
    // embedding streams, instead of materialising the prompt's text embeddings.
    void prepareEmbeddings(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input) override;
    void releaseEmbeddings() override;

   private:
    // Absolute positions of EVERY image token in the prompt, in order, offset by
    // `base` (the turn's starting KV position).
    std::vector<size_t> findImagePositions(const std::vector<int32_t>& ids, size_t base) const;

    std::filesystem::path bundle_dir_;

    // Non-owning; owned by input_providers_. Set in createInputProviders().
    EmbeddingInputProvider* main_embed_provider_     = nullptr;
    EmbeddingInputProvider* perlayer_embed_provider_ = nullptr;
};

// Full Gemma4 multimodal stack (VEG + decoder). Returns nullptr on failure.
GENIEX_VLM_API std::unique_ptr<Gemma4VLMModel> makeVLMModel(
    const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

// Convenience factory: VEG + decoder + processor, ready for VLMPipeline.
//
// Unlike the Qwen/InternVL families, Gemma4's metadata.json carries no
// `vision_preprocessing` block, so preprocessing geometry comes from
// Gemma4Config's defaults, which are pinned to the exported VEG geometry
// (force_square_size / patch_size / pooling_kernel_size). makeVLMModel()
// cross-checks those defaults against the VEG's own tensor shapes and fails
// loudly on a mismatch, rather than silently producing garbage soft tokens.
GENIEX_VLM_API std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

}  // namespace gemma4
}  // namespace geniex

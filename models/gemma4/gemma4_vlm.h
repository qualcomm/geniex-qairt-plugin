// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma4/gemma4_common.h"
#include "gemma4/gemma4_vision.h"
#include "geniex-proc/gemma4.h"
#include "llm/input_provider.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_model.h"

namespace geniex {
namespace gemma4 {

// Gemma4 as a first-class VLMModel, so `gemma_4_*` routes through
// makeVLMPipeline() like the other VLM families, with no per-model flag.
//
// VLMModel's default embedding path assumes a float32 table held in RAM.
// Gemma4's are quantized and very large (E4B: 1.28 GB main + 5.4 GB per-layer),
// so they stay memory-mapped behind EmbeddingInputProvider and are converted
// per row. prepareEmbeddings()/releaseEmbeddings() express the same effect
// positionally instead:
//
//   * main stream      -> setEmbeddingOverride(img_start, vision_rows)
//   * per-layer stream -> setTokenSubstitution(img_start, n_rows, pad_token_id)
//
// The second half is not optional. `per_layer_inputs` is a plain LUT lookup,
// but Gemma4 rewrites multimodal positions to the pad id before it:
//     llm_input_ids = where(multimodal_mask, pad_token_id, llm_input_ids)
// Letting the image token's own row through corrupts every layer at every image
// position, and the model answers as though no image were attached.
class Gemma4VLMModel : public VLMModel {
   public:
    Gemma4VLMModel(LLMSpec spec, ParsedGenieConfig gc, std::filesystem::path bundle_dir)
        : VLMModel(std::move(spec), std::move(gc)), bundle_dir_(std::move(bundle_dir)) {}

    void setVisionEncoder(std::unique_ptr<Gemma4VisionEncoder> vis) { vision_encoder_ = std::move(vis); }
    void setImageTokenId(int32_t image_token) { image_token_id_ = image_token; }

   protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override {
        if (!vision_encoder_) throw std::runtime_error("gemma4: no vision encoder registered");
        return vision_encoder_->encode(pixel_data);
    }

    // Same extra providers as the text-only Gemma4Model. setEmbeddingProvider()
    // is deliberately never called, so LLMModel still installs the quantized
    // main `inputs_embeds` provider and we keep a handle on it.
    void createInputProviders() override {
        LLMModel::createInputProviders();

        main_embed_provider_ = findEmbeddingProvider("inputs_embeds");
        if (!main_embed_provider_) main_embed_provider_ = findEmbeddingProvider("input_embeds");
        if (!main_embed_provider_) {
            GENIEX_LOG_ERROR("gemma4 VLM: main embedding provider NOT FOUND — vision splice unavailable");
        }

        auto extra = buildGemma4Providers(
            gc_, bundle_dir_, shard_count_, [this](size_t s) -> const Graph& { return graph(graphIndex(0, s, 0)); });

        if (extra.perlayer) {
            perlayer_embed_provider_ = extra.perlayer.get();
            input_providers_.push_back(std::move(extra.perlayer));
        }
        if (extra.local_rope) input_providers_.push_back(std::move(extra.local_rope));
        if (extra.global_rope) input_providers_.push_back(std::move(extra.global_rope));
    }

    // Installs the encoder's rows as a positional override on both embedding
    // streams, instead of materialising the prompt's text embeddings.
    void prepareEmbeddings(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input) override {
        if (vlm_input.pixel_data.pixel_values.empty()) return;  // text-only turn
        if (!main_embed_provider_) {
            throw std::runtime_error("gemma4: main embedding provider not available");
        }

        // Absolute positions, so multi-turn offsets the image span correctly.
        const size_t base      = nPast();
        const size_t img_start = findImageRun(prompt_tokens);

        auto         rows   = encodeVision(vlm_input.pixel_data);
        const size_t hidden = spec_.hidden_size;
        if (hidden == 0 || rows.size() % hidden != 0) {
            throw std::runtime_error("gemma4: vision encoder produced " + std::to_string(rows.size()) +
                                     " floats, not a multiple of hidden_size " + std::to_string(hidden));
        }
        const size_t n_rows = rows.size() / hidden;

        // Soft tokens must line up 1:1 with the prompt's image-token run, or the
        // override would shift text positions.
        if (n_rows != image_run_len_) {
            throw std::runtime_error("gemma4: prompt has " + std::to_string(image_run_len_) +
                                     " image tokens but the encoder produced " + std::to_string(n_rows));
        }

        main_embed_provider_->setEmbeddingOverride(base + img_start, std::move(rows));
        if (perlayer_embed_provider_) {
            const int32_t pad = gc_.pad_token_id >= 0 ? gc_.pad_token_id : 0;
            perlayer_embed_provider_->setTokenSubstitution(base + img_start, n_rows, pad);
        }
        GENIEX_LOG_DEBUG("gemma4 VLM: spliced {} vision rows at absolute position {}", n_rows, base + img_start);
    }

    void releaseEmbeddings() override {
        if (main_embed_provider_) main_embed_provider_->clearEmbeddingOverride();
        if (perlayer_embed_provider_) perlayer_embed_provider_->clearTokenSubstitution();
        image_run_len_ = 0;
    }

   private:
    // Start of the contiguous image-token run, recording its length.
    size_t findImageRun(const std::vector<int32_t>& ids) {
        const auto first = std::find(ids.begin(), ids.end(), image_token_id_);
        if (first == ids.end()) {
            throw std::runtime_error("gemma4: an image was supplied but the prompt carries no image token (id " +
                                     std::to_string(image_token_id_) + ")");
        }
        const size_t start = static_cast<size_t>(std::distance(ids.begin(), first));
        size_t       n     = 0;
        while (start + n < ids.size() && ids[start + n] == image_token_id_) ++n;
        image_run_len_ = n;
        return start;
    }

    std::filesystem::path bundle_dir_;

    // Non-owning; owned by input_providers_. Set in createInputProviders().
    EmbeddingInputProvider* main_embed_provider_     = nullptr;
    EmbeddingInputProvider* perlayer_embed_provider_ = nullptr;

    size_t image_run_len_ = 0;
};

// Full Gemma4 multimodal stack (VEG + decoder). Returns nullptr on failure.
GENIEX_VLM_API std::unique_ptr<Gemma4VLMModel> makeVLMModel(
    const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

// Convenience factory: VEG + decoder + processor, ready for VLMPipeline.
//
// Gemma4's metadata.json carries no `vision_preprocessing` block, so
// preprocessing geometry comes from Gemma4Config's defaults. makeVLMModel()
// cross-checks them against the VEG's own tensor shapes and fails loudly on a
// mismatch rather than producing garbage soft tokens.
GENIEX_VLM_API std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config);

}  // namespace gemma4
}  // namespace geniex

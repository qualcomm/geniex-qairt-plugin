// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "model.h"
#include "types.h"
#include "vlm/vision_encoder.h"

namespace geniex {
namespace gemma4 {

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
    bool initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) override {
        if (!QnnVisionEncoder::initialize(runtime_cfg, model_cfg)) return false;
        if (graphCount() == 0) {
            GENIEX_LOG_ERROR("gemma4 VEG: context binary exposes no graphs");
            return false;
        }

        const Graph& g = graph(0);
        for (const char* name : {kPixelValues, kPositionIds}) {
            if (!g.hasInput(name)) {
                GENIEX_LOG_ERROR("gemma4 VEG: graph is missing input '{}'", name);
                return false;
            }
        }
        if (!g.hasOutput(kVisionEmbedding)) {
            GENIEX_LOG_ERROR("gemma4 VEG: graph is missing output '{}'", kVisionEmbedding);
            return false;
        }

        const auto& pv  = g.inputSpec(kPixelValues);
        const auto& out = g.outputSpec(kVisionEmbedding);
        if (pv.shape.size() != 3 || out.shape.size() != 3) {
            GENIEX_LOG_ERROR("gemma4 VEG: unexpected tensor ranks");
            return false;
        }
        max_patches_   = pv.shape[1];
        patch_dim_     = pv.shape[2];
        n_soft_tokens_ = out.shape[1];
        hidden_size_   = out.shape[2];

        GENIEX_LOG_INFO("gemma4 VEG: pixel_values [1,{},{}] -> vision_embedding [1,{},{}]",
            max_patches_,
            patch_dim_,
            n_soft_tokens_,
            hidden_size_);
        return true;
    }

    size_t maxPatches() const { return max_patches_; }
    size_t patchDim() const { return patch_dim_; }
    size_t numSoftTokens() const { return n_soft_tokens_; }
    size_t hiddenSize() const { return hidden_size_; }

    // VisionEncoder entry point: runs every image in the batch and concatenates
    // their soft tokens, so the result lines up with the prompt's image-token
    // run in order. Gemma4 is a patch-budget encoder, so geometry comes from
    // image_position_ids and image_grid_thw is always empty.
    std::vector<float> encode(const PixelData& pixel_data) override {
        if (max_patches_ == 0 || patch_dim_ == 0) {
            throw std::runtime_error("gemma4 VEG: encode() before initialize()");
        }

        const size_t per_image_px  = max_patches_ * patch_dim_;
        const size_t per_image_pos = max_patches_ * 2;

        if (pixel_data.pixel_values.empty() || pixel_data.pixel_values.size() % per_image_px != 0) {
            throw std::runtime_error("gemma4 VEG: pixel_values has " + std::to_string(pixel_data.pixel_values.size()) +
                                     " floats, expected a multiple of " + std::to_string(per_image_px));
        }
        const size_t n_images = pixel_data.pixel_values.size() / per_image_px;

        if (pixel_data.image_position_ids.size() != n_images * per_image_pos) {
            throw std::runtime_error("gemma4 VEG: image_position_ids has " +
                                     std::to_string(pixel_data.image_position_ids.size()) + " ints, expected " +
                                     std::to_string(n_images * per_image_pos) + " (= " + std::to_string(n_images) +
                                     " images * " + std::to_string(per_image_pos) + ")");
        }

        const size_t       per_image_out = n_soft_tokens_ * hidden_size_;
        std::vector<float> out(n_images * per_image_out);

        for (size_t img = 0; img < n_images; ++img) {
            runOne(pixel_data.pixel_values.data() + img * per_image_px,
                pixel_data.image_position_ids.data() + img * per_image_pos,
                out.data() + img * per_image_out);
        }
        return out;
    }

    // Runs the encoder on ONE image. `pixel_values` must hold max_patches *
    // patch_dim floats and `image_position_ids` 2 * max_patches ints, exactly as
    // Gemma4Processor emits them.
    // Returns a flat [n_soft_tokens * hidden_size] buffer.
    std::vector<float> encode(const std::vector<float>& pixel_values, const std::vector<int32_t>& image_position_ids) {
        const size_t want_px  = max_patches_ * patch_dim_;
        const size_t want_pos = max_patches_ * 2;
        if (pixel_values.size() != want_px || image_position_ids.size() != want_pos) {
            throw std::runtime_error("gemma4 VEG: expected pixel_values " + std::to_string(want_px) +
                                     " and position_ids " + std::to_string(want_pos) + ", got " +
                                     std::to_string(pixel_values.size()) + " / " +
                                     std::to_string(image_position_ids.size()));
        }

        std::vector<float> out(n_soft_tokens_ * hidden_size_);
        runOne(pixel_values.data(), image_position_ids.data(), out.data());
        return out;
    }

   private:
    // One VEG pass. Callers have already validated the buffer extents.
    void runOne(const float* pixel_values, const int32_t* image_position_ids, float* out) {
        Graph& g = graph(0);
        g.write(kPixelValues, pixel_values, max_patches_ * patch_dim_);
        g.write(kPositionIds, image_position_ids, max_patches_ * 2);

        std::map<std::string, std::pair<double, uint16_t>> time_log;
        if (!g.execute(time_log)) {
            throw std::runtime_error("gemma4 VEG: graph execution failed");
        }

        g.read(kVisionEmbedding, out, n_soft_tokens_ * hidden_size_);
    }

    static constexpr const char* kPixelValues     = "pixel_values";
    static constexpr const char* kPositionIds     = "image_position_ids";
    static constexpr const char* kVisionEmbedding = "vision_embedding";

    size_t max_patches_   = 0;
    size_t patch_dim_     = 0;
    size_t n_soft_tokens_ = 0;
    size_t hidden_size_   = 0;
};

}  // namespace gemma4
}  // namespace geniex

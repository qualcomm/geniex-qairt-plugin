// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "intern3_5_vl.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "logging.h"
#include "utils.h"

namespace geniex {
namespace intern3_5_vl {

void Intern35VLVisionEncoder::setPreprocessing(const ParsedVisionPreprocessing& vp) {
    image_width_        = vp.image_width;
    image_height_       = vp.image_height;
    patch_size_         = vp.patch_size;
    spatial_merge_size_ = vp.spatial_merge_size;
}

std::vector<float> Intern35VLVisionEncoder::encode(const PixelData& pixel_data) {
    if (pixel_data.image_grid_thw.empty()) {
        throw std::runtime_error("Intern35VLVisionEncoder: empty image_grid_thw");
    }
    if (image_width_ == 0 || image_height_ == 0 || patch_size_ == 0 || spatial_merge_size_ == 0 || hidden_size_ == 0) {
        throw std::runtime_error("Intern35VLVisionEncoder: setPreprocessing/setHiddenSize not called before encode");
    }

    const int    grid_h  = image_height_ / patch_size_;
    const int    grid_w  = image_width_ / patch_size_;
    const size_t sm_unit = static_cast<size_t>(spatial_merge_size_) * spatial_merge_size_;

    const size_t num_patches      = static_cast<size_t>(grid_h) * grid_w;
    const size_t num_image_tokens = num_patches / sm_unit;

    // The graph takes a whole planar-CHW image, not patchified rows.
    const size_t per_image_pixels = static_cast<size_t>(3) * image_height_ * image_width_;
    const size_t per_image_tokens = num_image_tokens * hidden_size_;

    // Every tile must match the traced geometry: the graph has fixed shapes.
    for (const auto& thw : pixel_data.image_grid_thw) {
        if (thw[0] != 1 || thw[1] != grid_h || thw[2] != grid_w) {
            throw std::runtime_error("Intern35VLVisionEncoder: grid_thw = (" + std::to_string(thw[0]) + "," +
                                     std::to_string(thw[1]) + "," + std::to_string(thw[2]) + "), expected (1," +
                                     std::to_string(grid_h) + "," + std::to_string(grid_w) + ") for every image");
        }
    }

    const size_t n_images = pixel_data.image_grid_thw.size();
    if (pixel_data.pixel_values.size() != n_images * per_image_pixels) {
        throw std::runtime_error("Intern35VLVisionEncoder: pixel_values has " +
                                 std::to_string(pixel_data.pixel_values.size()) + " floats, expected " +
                                 std::to_string(n_images * per_image_pixels) + " (= " + std::to_string(n_images) +
                                 " images * " + std::to_string(per_image_pixels) + ")");
    }

    Graph& g = graph(0);

    std::vector<float> image_features(n_images * per_image_tokens);
    TimeLog            tl;

    for (size_t img = 0; img < n_images; ++img) {
        g.write("pixel_values", pixel_data.pixel_values.data() + img * per_image_pixels, per_image_pixels);

        if (!g.execute(tl)) {
            throw std::runtime_error(
                "Intern35VLVisionEncoder: vision_encoder graph execute failed on image " + std::to_string(img));
        }

        // Output token order already matches the LLM's <IMG_CONTEXT> run: this
        // export applies pixel-unshuffle inside the graph and does no window
        // permutation, so there is nothing to un-reorder.
        g.read("image_features", image_features.data() + img * per_image_tokens, per_image_tokens);
    }

    return image_features;
}

Intern35VLModel::Intern35VLModel(LLMSpec spec) : VLMModel(std::move(spec)) {
    setEmbeddingProvider(std::make_unique<PrecomputedEmbeddingProvider>("inputs_embeds"));
}

void Intern35VLModel::setVisionEncoder(std::unique_ptr<Intern35VLVisionEncoder> vis) {
    vision_encoder_ = std::move(vis);
}

std::vector<float> Intern35VLModel::encodeVision(const PixelData& pixel_data) {
    return vision_encoder_->encode(pixel_data);
}

std::unique_ptr<Intern35VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    try {
        const auto bundle = bundleDirOf(config.llm_config);
        auto       meta   = parseQAIRTMetadata(bundle);
        auto       gc     = parseGenieConfig(bundle);
        auto       spec   = buildSpecSkeleton(gc);

        if (!meta.vision_preprocessing) {
            GENIEX_LOG_ERROR("intern3_5_vl::makeModel: bundle has no vision_preprocessing block");
            return nullptr;
        }

        auto vis_enc = std::make_unique<Intern35VLVisionEncoder>();
        vis_enc->setPreprocessing(*meta.vision_preprocessing);
        vis_enc->setHiddenSize(meta.hidden_size);
        if (!vis_enc->initialize(runtime_cfg, config.vision_config)) return nullptr;

        auto model = std::make_unique<Intern35VLModel>(std::move(spec));
        model->setVisionEncoder(std::move(vis_enc));
        model->setImageTokenId(kImageTokenId);

        // Plain 1-D RoPE. head_dim comes from metadata.json; the graph's
        // position_ids_cos last dim is head_dim/2.
        if (meta.head_dim == 0) {
            GENIEX_LOG_ERROR("intern3_5_vl::makeModel: metadata.json has no head_dim");
            return nullptr;
        }
        GENIEX_LOG_INFO("intern3_5_vl: 1-D RoPE provider (head_dim={}, theta={})", meta.head_dim, gc.rope_theta);
        model->addInputProvider(makeRoPEProvider(meta.head_dim, gc));

        if (!model->initialize(runtime_cfg, config.llm_config)) return nullptr;

        return model;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("intern3_5_vl::makeModel failed: {}", e.what());
        return nullptr;
    }
}

}  // namespace intern3_5_vl
}  // namespace geniex

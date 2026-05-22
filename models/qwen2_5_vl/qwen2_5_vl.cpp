// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "qwen2_5_vl.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "logging.h"
#include "utils.h"
#include "vlm/vit_utils.h"
#include "vlm/vlm_utils.h"

namespace geniex {
namespace qwen2_5_vl {

void Qwen25VLVisionEncoder::setPreprocessing(const ParsedVisionPreprocessing& vp) {
    image_width_         = vp.image_width;
    image_height_        = vp.image_height;
    patch_size_          = vp.patch_size;
    temporal_patch_size_ = vp.temporal_patch_size;
    spatial_merge_size_  = vp.spatial_merge_size;
}

std::vector<float> Qwen25VLVisionEncoder::encode(const PixelData& pixel_data) {
    if (pixel_data.image_grid_thw.empty()) {
        throw std::runtime_error("Qwen25VLVisionEncoder: empty image_grid_thw");
    }
    if (image_width_ == 0 || image_height_ == 0 || patch_size_ == 0 || hidden_size_ == 0) {
        throw std::runtime_error("Qwen25VLVisionEncoder: setPreprocessing/setHiddenSize not called before encode");
    }

    const int    grid_t           = 1;
    const int    grid_h           = image_height_ / patch_size_;
    const int    grid_w           = image_width_ / patch_size_;
    const size_t num_patches      = static_cast<size_t>(grid_t * grid_h * grid_w);
    const size_t sm_unit          = static_cast<size_t>(spatial_merge_size_ * spatial_merge_size_);
    const size_t n_groups         = num_patches / sm_unit;
    const size_t num_image_tokens = num_patches / sm_unit;
    const size_t patch_feature_sz = static_cast<size_t>(3 * temporal_patch_size_ * patch_size_ * patch_size_);
    const size_t per_image_pixels = num_patches * patch_feature_sz;
    const size_t per_image_tokens = num_image_tokens * hidden_size_;

    for (const auto& thw : pixel_data.image_grid_thw) {
        if (thw[0] != grid_t || thw[1] != grid_h || thw[2] != grid_w) {
            throw std::runtime_error("Qwen25VLVisionEncoder: grid_thw = (" + std::to_string(thw[0]) + "," +
                                     std::to_string(thw[1]) + "," + std::to_string(thw[2]) + "), expected (" +
                                     std::to_string(grid_t) + "," + std::to_string(grid_h) + "," +
                                     std::to_string(grid_w) + ") for every image");
        }
    }

    const size_t n_images = pixel_data.image_grid_thw.size();
    Graph&       g        = graph(0);

    const auto window_index =
        qwen_vit::computeWindowIndex(grid_t, grid_h, grid_w, spatial_merge_size_, kVitWindowSize, patch_size_);
    const auto reverse_index = qwen_vit::reverseWindowIndex(window_index);
    const auto cu_window =
        qwen_vit::computeCuWindowSeqlens(grid_t, grid_h, grid_w, spatial_merge_size_, kVitWindowSize, patch_size_);

    const auto inv_freq = qwen_vit::makeInvFreq(kVitRopeDim, kVitRopeTheta);
    auto [rope_cos_nat, rope_sin_nat] =
        qwen_vit::computePatchRoPE(grid_t, grid_h, grid_w, spatial_merge_size_, inv_freq);
    const auto rope_cos = qwen_vit::windowReorder(rope_cos_nat, n_groups, sm_unit, kVitRopeDim, window_index);
    const auto rope_sin = qwen_vit::windowReorder(rope_sin_nat, n_groups, sm_unit, kVitRopeDim, window_index);

    constexpr float kAllowed    = 0.0f;
    constexpr float kBlocked    = -1e9f;
    const auto      window_mask = qwen_vit::buildBlockAttentionMask(num_patches, cu_window, kAllowed, kBlocked);
    const auto      full_mask =
        qwen_vit::buildBlockAttentionMask(num_patches, {0, static_cast<int64_t>(num_patches)}, kAllowed, kBlocked);

    if (pixel_data.pixel_values.size() != n_images * per_image_pixels) {
        throw std::runtime_error("Qwen25VLVisionEncoder: pixel_values has " +
                                 std::to_string(pixel_data.pixel_values.size()) + " floats, expected " +
                                 std::to_string(n_images * per_image_pixels) + " (= " + std::to_string(n_images) +
                                 " images * " + std::to_string(per_image_pixels) + ")");
    }

    std::vector<float> image_features(n_images * per_image_tokens);
    TimeLog            tl;

    for (size_t img = 0; img < n_images; ++img) {
        const float* pixels_natural = pixel_data.pixel_values.data() + img * per_image_pixels;

        g.write("pixel_values", pixels_natural, per_image_pixels);
        g.write("position_ids_cos", rope_cos.data(), rope_cos.size());
        g.write("position_ids_sin", rope_sin.data(), rope_sin.size());
        g.write("window_attention_mask", window_mask.data(), window_mask.size());
        g.write("full_attention_mask", full_mask.data(), full_mask.size());

        if (!g.execute(tl)) {
            throw std::runtime_error(
                "Qwen25VLVisionEncoder: vision_encoder graph execute failed on image " + std::to_string(img));
        }

        std::vector<float> tokens_reordered(per_image_tokens);
        g.read("image_features", tokens_reordered.data(), tokens_reordered.size());

        float* out_slot = image_features.data() + img * per_image_tokens;
        for (size_t i = 0; i < reverse_index.size(); ++i) {
            const size_t src = static_cast<size_t>(reverse_index[i]);
            std::memcpy(out_slot + i * hidden_size_,
                tokens_reordered.data() + src * hidden_size_,
                hidden_size_ * sizeof(float));
        }
    }

    return image_features;
}

Qwen25VLModel::Qwen25VLModel(LLMSpec spec) : VLMModel(std::move(spec)) {
    setEmbeddingProvider(std::make_unique<PrecomputedEmbeddingProvider>("inputs_embeds"));
}

void Qwen25VLModel::setVisionEncoder(std::unique_ptr<Qwen25VLVisionEncoder> vis) { vision_encoder_ = std::move(vis); }

void Qwen25VLModel::setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider) {
    mrope_provider_ = provider.get();
    addInputProvider(std::move(provider));
}

void Qwen25VLModel::setVisionTokenIds(int32_t vision_start, int32_t image_pad) {
    vision_start_token_ = vision_start;
    image_token_id_     = image_pad;
}

std::vector<float> Qwen25VLModel::encodeVision(const PixelData& pixel_data) {
    return vision_encoder_->encode(pixel_data);
}

void Qwen25VLModel::preparePositions(const std::vector<int32_t>& input_ids, const VLMInput& vlm_input, size_t n_past) {
    if (!mrope_provider_) return;

    std::vector<ImageGrid> image_grids(
        vlm_input.pixel_data.image_grid_thw.begin(), vlm_input.pixel_data.image_grid_thw.end());

    auto mrope = computeMRoPEPositions(input_ids,
        image_grids,
        /*audio_segments=*/{},
        spatial_merge_size_,
        vision_start_token_,
        image_token_id_,
        /*audio_start=*/0,
        /*audio_token=*/0);

    const size_t seq_len = input_ids.size();

    for (size_t dim = 0; dim < 3; ++dim)
        for (size_t t = 0; t < seq_len; ++t)
            mrope.position_ids[dim * seq_len + t] += static_cast<int32_t>(n_past) + mrope_deltas_[dim];

    mrope_provider_->setPositionIds(mrope.position_ids, seq_len, n_past);

    for (int d = 0; d < 3; ++d) mrope_deltas_[d] += mrope.deltas[d];
    mrope_provider_->setMropeDeltas(mrope_deltas_);
}

void Qwen25VLModel::clearPositions() {
    if (mrope_provider_) mrope_provider_->clearPositionIds();
}

std::unique_ptr<Qwen25VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    try {
        const auto bundle = bundleDirOf(config.llm_config);
        auto       meta   = parseQAIRTMetadata(bundle);
        auto       gc     = parseGenieConfig(bundle);
        auto       spec   = buildSpec(meta, gc);

        if (!meta.vision_preprocessing) {
            GENIEX_LOG_ERROR("qwen2_5_vl::makeModel: bundle has no vision_preprocessing block");
            return nullptr;
        }

        // mrope_section comes from genie_config.json's
        // dialog.engine.model.positional-encoding.rope-scaling (rope-type
        // "qwen2vl-mrope"). Default to {16,24,24} if the bundle declares
        // mrope but the section is missing.
        std::vector<int> mrope_section;
        if (auto* mrope = std::get_if<MRopeScaling>(&gc.rope_scaling)) {
            mrope_section = mrope->mrope_section;
        }
        if (mrope_section.empty()) mrope_section = {16, 24, 24};

        auto vis_enc = std::make_unique<Qwen25VLVisionEncoder>();
        vis_enc->setPreprocessing(*meta.vision_preprocessing);
        vis_enc->setHiddenSize(meta.hidden_size);
        if (!vis_enc->initialize(runtime_cfg, config.vision_config)) return nullptr;

        auto model = std::make_unique<Qwen25VLModel>(std::move(spec));
        model->setVisionEncoder(std::move(vis_enc));
        model->setMRoPEProvider(
            std::make_unique<MRoPEInputProvider>(mrope_section, gc.rope_theta, MRoPEInterleaving::BLOCK));

        // Vision token IDs are family-level constants (see qwen2_5_vl.h).
        model->setVisionTokenIds(kVisionStartTokenId, kImageTokenId);
        model->setSpatialMergeSize(meta.vision_preprocessing->spatial_merge_size);

        if (!model->initialize(runtime_cfg, config.llm_config)) return nullptr;

        return model;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("qwen2_5_vl::makeModel failed: {}", e.what());
        return nullptr;
    }
}

}  // namespace qwen2_5_vl
}  // namespace geniex

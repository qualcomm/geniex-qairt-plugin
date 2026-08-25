// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "qwen3_vl.h"

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
namespace qwen3_vl {

void Qwen3VLVisionEncoder::setPreprocessing(const ParsedVisionPreprocessing& vp) {
    image_width_         = vp.image_width;
    image_height_        = vp.image_height;
    patch_size_          = vp.patch_size;
    temporal_patch_size_ = vp.temporal_patch_size;
    spatial_merge_size_  = vp.spatial_merge_size;
}

bool Qwen3VLVisionEncoder::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    if (!QnnVisionEncoder::initialize(runtime_cfg, model_cfg)) return false;
    inferSpecFromGraphs();
    return true;
}

void Qwen3VLVisionEncoder::inferSpecFromGraphs() {
    Graph& g = graph(0);

    // Detect the RoPE cos/sin embed dim from the graph's actual tensor shape
    // instead of a hardcoded constant (see qwen3_vl.h comment on kVitRopeTheta
    // for why this can't come from genie_config.json). Fail loudly rather than
    // guess a default: a wrong dim silently corrupts vision positions without
    // any other observable symptom (see PR history for the 4B-vs-8B bug this
    // fixes).
    if (!g.hasInput("position_ids_cos")) {
        throw std::runtime_error("inferSpecFromGraphs: vision graph has no 'position_ids_cos' input");
    }
    const auto& shape = g.inputSpec("position_ids_cos").shape;
    if (shape.empty()) {
        throw std::runtime_error("inferSpecFromGraphs: 'position_ids_cos' has an empty shape");
    }
    rope_dim_ = static_cast<int>(shape.back());

    // The graph structure is fixed, so detect the DeepStack output count once
    // here rather than on every encode() (which loops over images).
    num_deepstack_levels_ = 0;
    while (g.hasOutput(deepstack_prefix_ + std::to_string(num_deepstack_levels_))) ++num_deepstack_levels_;
    if (num_deepstack_levels_ == 0) {
        GENIEX_LOG_WARN(
            "inferSpecFromGraphs: no '{}<k>' outputs found on the vision graph; "
            "DeepStack injection will be a no-op",
            deepstack_prefix_);
    }
}

std::vector<float> Qwen3VLVisionEncoder::encode(const PixelData& pixel_data) {
    if (pixel_data.image_grid_thw.empty()) {
        throw std::runtime_error("Qwen3VLVisionEncoder: empty image_grid_thw");
    }
    if (image_width_ == 0 || image_height_ == 0 || patch_size_ == 0 || hidden_size_ == 0) {
        throw std::runtime_error("Qwen3VLVisionEncoder: setPreprocessing/setHiddenSize not called before encode");
    }

    const int    grid_t           = 1;
    const int    grid_h           = image_height_ / patch_size_;
    const int    grid_w           = image_width_ / patch_size_;
    const size_t num_patches      = static_cast<size_t>(grid_t * grid_h * grid_w);
    const size_t sm_unit          = static_cast<size_t>(spatial_merge_size_ * spatial_merge_size_);
    const size_t num_image_tokens = num_patches / sm_unit;
    const size_t patch_feature_sz = static_cast<size_t>(3 * temporal_patch_size_ * patch_size_ * patch_size_);
    const size_t per_image_pixels = num_patches * patch_feature_sz;
    const size_t per_image_tokens = num_image_tokens * hidden_size_;

    for (const auto& thw : pixel_data.image_grid_thw) {
        if (thw[0] != grid_t || thw[1] != grid_h || thw[2] != grid_w) {
            throw std::runtime_error("Qwen3VLVisionEncoder: grid_thw = (" + std::to_string(thw[0]) + "," +
                                     std::to_string(thw[1]) + "," + std::to_string(thw[2]) + "), expected (" +
                                     std::to_string(grid_t) + "," + std::to_string(grid_h) + "," +
                                     std::to_string(grid_w) + ") for every image");
        }
    }

    const size_t n_images = pixel_data.image_grid_thw.size();
    Graph&       g        = graph(0);

    // Qwen3-VL's ViT uses full (non-windowed) attention, so there is no window
    // reordering of patches: RoPE cos/sin and the output features stay in
    // natural patch order. computePatchRoPE() already yields natural order.
    const auto inv_freq       = qwen_vit::makeInvFreq(rope_dim_, kVitRopeTheta);
    auto [rope_cos, rope_sin] = qwen_vit::computePatchRoPE(grid_t, grid_h, grid_w, spatial_merge_size_, inv_freq);

    // Single image ⇒ one attention block spanning all patches. Both the full
    // and window attention masks are therefore "all allowed" (0.0).
    constexpr float kAllowed = 0.0f;
    constexpr float kBlocked = -1e9f;
    const auto      full_mask =
        qwen_vit::buildBlockAttentionMask(num_patches, {0, static_cast<int64_t>(num_patches)}, kAllowed, kBlocked);

    if (pixel_data.pixel_values.size() != n_images * per_image_pixels) {
        throw std::runtime_error("Qwen3VLVisionEncoder: pixel_values has " +
                                 std::to_string(pixel_data.pixel_values.size()) + " floats, expected " +
                                 std::to_string(n_images * per_image_pixels) + " (= " + std::to_string(n_images) +
                                 " images * " + std::to_string(per_image_pixels) + ")");
    }

    std::vector<float> image_features(n_images * per_image_tokens);
    deepstack_embeds_.assign(num_deepstack_levels_, std::vector<float>(n_images * per_image_tokens));
    TimeLog tl;

    for (size_t img = 0; img < n_images; ++img) {
        const float* pixels_natural = pixel_data.pixel_values.data() + img * per_image_pixels;

        g.write("pixel_values", pixels_natural, per_image_pixels);
        g.write("position_ids_cos", rope_cos.data(), rope_cos.size());
        g.write("position_ids_sin", rope_sin.data(), rope_sin.size());
        if (g.hasInput("window_attention_mask")) g.write("window_attention_mask", full_mask.data(), full_mask.size());
        if (g.hasInput("full_attention_mask")) g.write("full_attention_mask", full_mask.data(), full_mask.size());

        if (!g.execute(tl)) {
            throw std::runtime_error(
                "Qwen3VLVisionEncoder: vision_encoder graph execute failed on image " + std::to_string(img));
        }

        float* feat_slot = image_features.data() + img * per_image_tokens;
        g.read("image_features", feat_slot, per_image_tokens);

        for (size_t k = 0; k < num_deepstack_levels_; ++k) {
            float* ds_slot = deepstack_embeds_[k].data() + img * per_image_tokens;
            g.read(deepstack_prefix_ + std::to_string(k), ds_slot, per_image_tokens);
        }
    }

    return image_features;
}

Qwen3VLModel::Qwen3VLModel(LLMSpec spec) : VLMModel(std::move(spec)) {
    setEmbeddingProvider(std::make_unique<PrecomputedEmbeddingProvider>("inputs_embeds"));
}

void Qwen3VLModel::setVisionEncoder(std::unique_ptr<Qwen3VLVisionEncoder> vis) {
    vision_encoder_raw_ = vis.get();
    vision_encoder_     = std::move(vis);
}

void Qwen3VLModel::setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider) {
    mrope_provider_ = provider.get();
    addInputProvider(std::move(provider));
}

void Qwen3VLModel::setDeepstackProvider(std::unique_ptr<DeepstackInputProvider> provider) {
    deepstack_provider_ = provider.get();
    addInputProvider(std::move(provider));
}

void Qwen3VLModel::setVisionTokenIds(int32_t vision_start, int32_t image_pad) {
    vision_start_token_ = vision_start;
    image_token_id_     = image_pad;
}

std::vector<float> Qwen3VLModel::encodeVision(const PixelData& pixel_data) {
    // Runs the ViT then hands the per-level deepstack embeddings to the provider
    // that feeds the first decoder shard. The returned image features are scattered
    // onto the image token positions by VLMModel::generate().
    auto image_features = vision_encoder_raw_->encode(pixel_data);

    if (deepstack_provider_) {
        deepstack_provider_->setEmbeds(vision_encoder_raw_->deepstackEmbeds(), spec_.hidden_size);
    }
    return image_features;
}

void Qwen3VLModel::preparePositions(const std::vector<int32_t>& input_ids, const VLMInput& vlm_input, size_t n_past) {
    // DeepStack visual-position mask: 1 where the token is an image token. The
    // compiled first shard scatters the deepstack features onto these rows.
    if (deepstack_provider_) {
        std::vector<uint8_t> visual_mask(input_ids.size(), 0);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (input_ids[i] == image_token_id_) visual_mask[i] = 1;
        }
        deepstack_provider_->setVisualMask(std::move(visual_mask), n_past);
    }

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

void Qwen3VLModel::clearPositions() {
    if (mrope_provider_) mrope_provider_->clearPositionIds();
    if (deepstack_provider_) deepstack_provider_->clear();
}

void Qwen3VLModel::resetKVCache() {
    LLMModel::resetKVCache();
    // Drop any in-flight prefill state and the accumulated MRoPE position
    // offset so a reset (e.g. after a mid-prefill error) starts clean.
    clearPositions();
    mrope_deltas_ = {0, 0, 0};
    if (mrope_provider_) mrope_provider_->resetMropeDeltas();
}

std::unique_ptr<Qwen3VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    try {
        const auto bundle = bundleDirOf(config.llm_config);
        auto       meta   = parseQAIRTMetadata(bundle);
        auto       gc     = parseGenieConfig(bundle);
        auto       spec   = buildSpecSkeleton(gc);

        if (!meta.vision_preprocessing) {
            GENIEX_LOG_ERROR("qwen3_vl::makeModel: bundle has no vision_preprocessing block");
            return nullptr;
        }

        // mrope_section comes from genie_config.json's
        // dialog.engine.model.positional-encoding.rope-scaling (rope-type
        // "qwen3vl-mrope"). Default to {24,20,20} if the bundle declares mrope
        // but the section is missing.
        std::vector<int> mrope_section;
        if (auto* mrope = std::get_if<MRopeScaling>(&gc.rope_scaling)) {
            mrope_section = mrope->mrope_section;
        }
        if (mrope_section.empty()) mrope_section = {24, 20, 20};

        // Single source of truth for the DeepStack tensor naming: the ViT read
        // side (encoder) and the decoder write side (provider) take the same
        // prefix via their constructors.
        const std::string kDeepstackPrefix = "deepstack_visual_embeds_";

        auto vis_enc = std::make_unique<Qwen3VLVisionEncoder>(kDeepstackPrefix);
        vis_enc->setPreprocessing(*meta.vision_preprocessing);
        vis_enc->setHiddenSize(meta.hidden_size);
        if (!vis_enc->initialize(runtime_cfg, config.vision_config)) return nullptr;

        auto model = std::make_unique<Qwen3VLModel>(std::move(spec));
        model->setVisionEncoder(std::move(vis_enc));

        // Qwen3-VL uses *interleaved* MRoPE (THWTHW… stride-3 layout), unlike
        // Qwen2.5-VL's block layout.
        model->setMRoPEProvider(
            std::make_unique<MRoPEInputProvider>(mrope_section, gc.rope_theta, MRoPEInterleaving::STRIDE));

        // DeepStack visual features feed the first decoder shard (same tensor
        // naming as the ViT outputs above).
        model->setDeepstackProvider(std::make_unique<DeepstackInputProvider>(kDeepstackPrefix));

        // Vision token IDs are family-level constants (see qwen3_vl.h).
        model->setVisionTokenIds(kVisionStartTokenId, kImageTokenId);
        model->setSpatialMergeSize(meta.vision_preprocessing->spatial_merge_size);

        if (!model->initialize(runtime_cfg, config.llm_config)) return nullptr;

        return model;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("qwen3_vl::makeModel failed: {}", e.what());
        return nullptr;
    }
}

}  // namespace qwen3_vl
}  // namespace geniex

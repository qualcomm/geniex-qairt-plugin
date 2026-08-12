// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 multimodal factories. Compiled into geniex_vlm.

#include "gemma4/gemma4.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "geniex-proc/gemma4.h"
#include "logging.h"
#include "utils.h"

namespace geniex {
namespace gemma4 {

std::unique_ptr<Gemma4VLMModel> makeVLMModel(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    try {
        const auto bundle = bundleDirOf(config.llm_config);
        auto       gc     = parseGenieConfig(bundle);
        auto       spec   = buildSpecSkeleton(gc);

        if (config.vision_config.model_paths.empty()) {
            GENIEX_LOG_ERROR("gemma4::makeVLMModel: no vision encoder context binary supplied");
            return nullptr;
        }

        auto vis_enc = std::make_unique<Gemma4VisionEncoder>();
        if (!vis_enc->initialize(runtime_cfg, config.vision_config)) return nullptr;

        // Gemma4's metadata.json has no vision_preprocessing block, so the
        // processor runs on Gemma4Config's defaults. Those defaults encode the
        // exported geometry, so cross-check them against what the VEG graph
        // actually declares rather than trusting them silently.
        const Gemma4Config proc_cfg;
        if (vis_enc->maxPatches() != static_cast<size_t>(proc_cfg.max_patches()) ||
            vis_enc->patchDim() != static_cast<size_t>(proc_cfg.patch_dim())) {
            GENIEX_LOG_ERROR(
                "gemma4::makeVLMModel: VEG expects pixel_values [1,{},{}] but the processor produces [1,{},{}] — "
                "Gemma4Config (max_soft_tokens * pooling_kernel_size^2, patch_size^2 * 3) does not match this export",
                vis_enc->maxPatches(),
                vis_enc->patchDim(),
                proc_cfg.max_patches(),
                proc_cfg.patch_dim());
            return nullptr;
        }
        // Soft-token count is NOT max_soft_tokens: that field is the *patch budget*
        // cap (max_patches = max_soft_tokens * pooling^2). The count the VEG emits
        // follows from force_square_size, since the processor pre-squares every
        // image to it and then pools:
        //     (force_square_size / patch_size)^2 / pooling_kernel_size^2
        // For the shipped defaults that is (768/16)^2 / 3^2 = 2304/9 = 256, which
        // is exactly what the export traces. Verify that identity, so a config
        // whose square size disagrees with the export is rejected here instead of
        // producing a soft-token run that silently misaligns with the prompt.
        if (proc_cfg.force_square_size <= 0) {
            GENIEX_LOG_ERROR(
                "gemma4::makeVLMModel: force_square_size is disabled, but the VEG is traced at a fixed {} soft "
                "tokens — an aspect-preserving resize would vary the count per image",
                vis_enc->numSoftTokens());
            return nullptr;
        }
        const size_t side_patches  = static_cast<size_t>(proc_cfg.force_square_size / proc_cfg.patch_size);
        const size_t pool          = static_cast<size_t>(proc_cfg.pooling_kernel_size);
        const size_t expected_soft = (side_patches * side_patches) / (pool * pool);
        if (vis_enc->numSoftTokens() != expected_soft) {
            GENIEX_LOG_ERROR(
                "gemma4::makeVLMModel: VEG emits {} soft tokens but the processor produces {} "
                "(force_square_size={} / patch_size={} -> {}x{} patches, pooled by {}^2) — "
                "Gemma4Config does not match this export",
                vis_enc->numSoftTokens(),
                expected_soft,
                proc_cfg.force_square_size,
                proc_cfg.patch_size,
                side_patches,
                side_patches,
                pool);
            return nullptr;
        }

        // The decoder needs the MAIN embedding LUT on the CPU side; the bundle
        // only names it in genie_config.json.
        const auto llm_cfg = withEmbeddingPath(config.llm_config, gc, bundle);

        auto model = std::make_unique<Gemma4VLMModel>(std::move(spec), gc, bundle);
        model->setVisionEncoder(std::move(vis_enc));

        if (!model->initialize(runtime_cfg, llm_cfg)) return nullptr;
        return model;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("gemma4::makeVLMModel failed: {}", e.what());
        return nullptr;
    }
}

std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    try {
        const auto bundle = bundleDirOf(config.llm_config);
        auto       gc     = parseGenieConfig(bundle);

        // Chat template lives in tokenizer_config.json; VLMPipeline drives
        // apply_chat_template() through the processor, so it must be supplied.
        std::string tok_cfg;
        if (config.llm_config.tokenizer_config_path && !config.llm_config.tokenizer_config_path->empty()) {
            tok_cfg = *config.llm_config.tokenizer_config_path;
        } else {
            const auto candidate = bundle / "tokenizer_config.json";
            if (std::filesystem::exists(candidate)) tok_cfg = candidate.string();
        }
        if (tok_cfg.empty()) {
            GENIEX_LOG_ERROR(
                "gemma4::makeVLMPipeline: tokenizer_config.json not found in {} — no chat template", bundle.string());
            return std::nullopt;
        }

        const Gemma4Config proc_cfg;
        auto               processor = Gemma4Processor::create(config.llm_config.tokenizer_path, tok_cfg, proc_cfg);
        if (!processor) {
            GENIEX_LOG_ERROR("gemma4::makeVLMPipeline: failed to create the Gemma4 processor");
            return std::nullopt;
        }

        // The image-token id is a property of this bundle's tokenizer, so read it
        // rather than hard-coding it: the run of these ids in the rendered prompt
        // is what the vision rows are spliced over.
        const auto image_ids = processor->tokenizer().encode(proc_cfg.image_token, false);
        if (image_ids.size() != 1) {
            GENIEX_LOG_ERROR("gemma4::makeVLMPipeline: image token '{}' encodes to {} ids, expected exactly 1",
                proc_cfg.image_token,
                image_ids.size());
            return std::nullopt;
        }

        auto model = makeVLMModel(runtime_cfg, config);
        if (!model) return std::nullopt;
        model->setImageTokenId(image_ids.front());

        Tokenizer& tok = processor->tokenizer();

        VLMPipeline pipe;
        if (!pipe.create(std::move(model), std::move(processor), tok)) return std::nullopt;

        // Gemma4's chat template does not emit BOS itself, and the text-only
        // pipeline prepends it the same way.
        pipe.setBosTokenId(gc.bos_token_id);
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("gemma4::makeVLMPipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace gemma4
}  // namespace geniex

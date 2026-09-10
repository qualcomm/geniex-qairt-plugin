// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 multimodal implementation: the VEG wrapper, the decoder subclass and
// the two factories. Compiled into geniex_vlm.

#include "gemma4/gemma4.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "geniex-proc/gemma4.h"
#include "graph.h"
#include "llm/llm_model.h"
#include "logging.h"
#include "model.h"
#include "utils.h"

namespace geniex {
namespace gemma4 {
namespace {

// ═════════════════════════════════════════════════════════════════════════════
// Provider wiring
//
// The decoder needs three extra CPU-side providers on top of what
// LLMModel::createInputProviders() installs, all resolved from
// genie_config.json. They are built here rather than inline so the wiring stays
// reviewable in one place — two of the three (the per-layer stream and the local
// RoPE table) are invisible in output quality until they are subtly wrong.
// ═════════════════════════════════════════════════════════════════════════════

// Local (sliding-window) RoPE bound to the swa_position_ids_* tensors. Mirrors
// makeRoPEProvider's variant handling but with the local theta/scaling.
std::unique_ptr<InputProvider> makeLocalRoPEProvider(const ParsedGenieConfig& gc, size_t head_dim) {
    return std::visit(
        [&](const auto& s) -> std::unique_ptr<InputProvider> {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, PartialRopeScaling>) {
                return std::make_unique<PartialRoPEInputProvider>(
                    head_dim, gc.local_rope_theta, s.rope_fraction, s.scale, kSwaPositionCos, kSwaPositionSin);
            } else {
                // Gemma local layers use plain (full-rotary) RoPE.
                return std::make_unique<RoPEInputProvider>(
                    head_dim, gc.local_rope_theta, kSwaPositionCos, kSwaPositionSin);
            }
        },
        gc.local_rope_scaling);
}

// The extra providers a Gemma4 decoder needs. Any member may be null when the
// bundle does not describe the corresponding stream.
struct Gemma4Providers {
    // Auxiliary per-token embedding stream (`per_layer_inputs`). Held as the
    // concrete type because the decoder must keep a handle on it to redirect
    // image positions to PAD — see Gemma4VLMModel::prepareEmbeddings().
    std::unique_ptr<EmbeddingInputProvider> perlayer;
    std::unique_ptr<InputProvider>          local_rope;
    std::unique_ptr<InputProvider>          global_rope;
};

// Builds the three providers described above.
//
// `head_dim_of(cos_tensor_name)` must resolve a RoPE cos tensor's head_dim by
// scanning the model's shard graphs (LLMModel::discoverRopeHeadDim); the head
// dims are read from the graph's tensors rather than assumed, since they differ
// across variants (E2B vs E4B) and across exports.
Gemma4Providers buildGemma4Providers(
    const ParsedGenieConfig& gc, const std::function<size_t(const char*)>& head_dim_of) {
    Gemma4Providers out;

    // (1) Per-layer embedding stream. A second embedding table feeding
    // `per_layer_inputs`; its row width is num_layers * per_layer_dim, NOT
    // spec.hidden_size, so it uses the explicit-config EmbeddingInputProvider.
    if (gc.perlayer_embedding_lut_path && gc.perlayer_embedding_size > 0) {
        const std::string& lut      = *gc.perlayer_embedding_lut_path;
        auto               provider = std::make_unique<EmbeddingInputProvider>(
            /*tensor_name=*/kPerLayerInputs,
            /*table_path=*/lut,
            /*row_hidden_size=*/gc.perlayer_embedding_size,
            /*pad_token_override=*/gc.pad_token_id >= 0 ? gc.pad_token_id : 0);
        // This is the table that makes the in-RAM path impossible: E2B's is
        // 2.35 GB as int8 and 9.4 GB dequantized. Quantized => mmap.
        if (gc.perlayer_embedding_quant.quantized()) {
            provider->setQuantization(gc.perlayer_embedding_quant);
        }
        out.perlayer = std::move(provider);
        GENIEX_LOG_INFO("gemma4: per-layer embedding provider ({} dims) -> {}", gc.perlayer_embedding_size, lut);
    }

    // (2) Local (swa) RoPE.
    if (gc.local_positional_encoding_present) {
        const size_t local_head_dim = head_dim_of(kSwaPositionCos);
        if (local_head_dim > 0) {
            out.local_rope = makeLocalRoPEProvider(gc, local_head_dim);
            GENIEX_LOG_INFO(
                "gemma4: local (swa) RoPE provider head_dim={} theta={}", local_head_dim, gc.local_rope_theta);
        } else {
            GENIEX_LOG_WARN("gemma4: local-positional-encoding set but no {} tensor found", kSwaPositionCos);
        }
    }

    // (3) Global RoPE — present when global-attention layers expose a separate
    // pair. Partial-rotary is applied inside the graph, so the CPU-side table is
    // plain full RoPE (not makeRoPEProvider).
    {
        const size_t global_head_dim = head_dim_of(kGlobalPositionCos);
        if (global_head_dim > 0) {
            out.global_rope = std::make_unique<RoPEInputProvider>(
                global_head_dim, gc.rope_theta, kGlobalPositionCos, kGlobalPositionSin);
            GENIEX_LOG_INFO("gemma4: global RoPE provider head_dim={} theta={}", global_head_dim, gc.rope_theta);
        } else {
            GENIEX_LOG_WARN("gemma4: no {} tensor found; global RoPE provider not installed", kGlobalPositionCos);
        }
    }

    return out;
}

// Gemma feeds CPU-side embeddings (`inputs_embeds`), so the base embedding
// provider needs model_cfg.embedding_path pointing at the MAIN embedding LUT.
// Fallback for a ModelConfig assembled without going through
// modelConfigFromDirectory, which already sets this from genie_config.
ModelConfig withEmbeddingPath(ModelConfig cfg, const ParsedGenieConfig& gc) {
    if (!cfg.embedding_path && gc.embedding_lut_path) {
        cfg.embedding_path = *gc.embedding_lut_path;
    }
    return cfg;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Gemma4VisionEncoder
// ═════════════════════════════════════════════════════════════════════════════

bool Gemma4VisionEncoder::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    if (!QnnVisionEncoder::initialize(runtime_cfg, model_cfg)) return false;
    if (graphCount() == 0) {
        GENIEX_LOG_ERROR("gemma4 VEG: context binary exposes no graphs");
        return false;
    }

    const Graph& g = graph(0);
    for (const char* name : {kPixelValues, kImagePositionIds}) {
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

std::vector<float> Gemma4VisionEncoder::encode(const PixelData& pixel_data) {
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

void Gemma4VisionEncoder::runOne(const float* pixel_values, const int32_t* image_position_ids, float* out) {
    Graph& g = graph(0);
    g.write(kPixelValues, pixel_values, max_patches_ * patch_dim_);
    g.write(kImagePositionIds, image_position_ids, max_patches_ * 2);

    std::map<std::string, std::pair<double, uint16_t>> time_log;
    if (!g.execute(time_log)) {
        throw std::runtime_error("gemma4 VEG: graph execution failed");
    }

    g.read(kVisionEmbedding, out, n_soft_tokens_ * hidden_size_);
}

// ═════════════════════════════════════════════════════════════════════════════
// Gemma4VLMModel
// ═════════════════════════════════════════════════════════════════════════════

Gemma4VLMModel::Gemma4VLMModel(LLMSpec spec, ParsedGenieConfig gc) : VLMModel(std::move(spec), std::move(gc)) {}

void Gemma4VLMModel::setVisionEncoder(std::unique_ptr<Gemma4VisionEncoder> vis) { vision_encoder_ = std::move(vis); }

void Gemma4VLMModel::setImageTokenId(int32_t image_token) { image_token_id_ = image_token; }

std::vector<float> Gemma4VLMModel::encodeVision(const PixelData& pixel_data) {
    if (!vision_encoder_) throw std::runtime_error("gemma4: no vision encoder registered");
    return vision_encoder_->encode(pixel_data);
}

void Gemma4VLMModel::createInputProviders() {
    LLMModel::createInputProviders();

    main_embed_provider_ = findEmbeddingProvider(kInputsEmbeds);
    if (!main_embed_provider_) main_embed_provider_ = findEmbeddingProvider(kInputsEmbedsAlt);
    if (!main_embed_provider_) {
        GENIEX_LOG_ERROR("gemma4 VLM: main embedding provider NOT FOUND — vision splice unavailable");
    } else {
        // Gemma4's embedding LUT was calibrated with round-to-nearest, so the
        // float->UFIXED write must not truncate; the core default is TowardZero.
        // Not opting in shifts every embedding row by up to one quantization
        // step and degrades output quality.
        main_embed_provider_->setRoundingMode(RoundingMode::Nearest);
    }

    auto extra = buildGemma4Providers(gc_, [this](const char* cos_name) { return discoverRopeHeadDim(cos_name); });

    if (extra.perlayer) {
        perlayer_embed_provider_ = extra.perlayer.get();
        perlayer_embed_provider_->setRoundingMode(RoundingMode::Nearest);
        input_providers_.push_back(std::move(extra.perlayer));
    }
    if (extra.local_rope) input_providers_.push_back(std::move(extra.local_rope));
    if (extra.global_rope) input_providers_.push_back(std::move(extra.global_rope));
}

void Gemma4VLMModel::prepareEmbeddings(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input) {
    if (vlm_input.pixel_data.pixel_values.empty()) return;  // text-only turn
    if (!main_embed_provider_) {
        throw std::runtime_error("gemma4: main embedding provider not available");
    }

    // Absolute prompt positions: this turn's tokens start at the current
    // nPast(), so a multi-turn conversation offsets the image positions
    // correctly and each round only overrides its own attachments.
    const auto positions = findImagePositions(prompt_tokens, nPast());

    auto         rows   = encodeVision(vlm_input.pixel_data);
    const size_t hidden = spec_.hidden_size;
    if (hidden == 0 || rows.size() % hidden != 0) {
        throw std::runtime_error("gemma4: vision encoder produced " + std::to_string(rows.size()) +
                                 " floats, not a multiple of hidden_size " + std::to_string(hidden));
    }
    const size_t n_rows = rows.size() / hidden;

    // The encoder's soft tokens must line up 1:1 with the prompt's image tokens,
    // in order; otherwise the override would land on text positions.
    if (n_rows != positions.size()) {
        throw std::runtime_error("gemma4: prompt has " + std::to_string(positions.size()) +
                                 " image tokens but the encoder produced " + std::to_string(n_rows) +
                                 " rows — every attached image must expand to its own soft-token run");
    }

    main_embed_provider_->setEmbeddingOverride(positions, std::move(rows));
    if (perlayer_embed_provider_) {
        const int32_t pad = gc_.pad_token_id >= 0 ? gc_.pad_token_id : 0;
        perlayer_embed_provider_->setTokenSubstitution(positions, pad);
    }
    GENIEX_LOG_DEBUG(
        "gemma4 VLM: spliced {} vision rows over positions [{}..{}]", n_rows, positions.front(), positions.back());
}

void Gemma4VLMModel::releaseEmbeddings() {
    if (main_embed_provider_) main_embed_provider_->clearEmbeddingOverride();
    if (perlayer_embed_provider_) perlayer_embed_provider_->clearTokenSubstitution();
}

// Deliberately not "first contiguous run": several attachments in one turn
// produce several runs, separated by the template's eoi/boi text. Scanning for a
// single run would return only the first image's span, and the 1:1 check against
// the encoder's row count would then reject the whole turn.
std::vector<size_t> Gemma4VLMModel::findImagePositions(const std::vector<int32_t>& ids, size_t base) const {
    std::vector<size_t> positions;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == image_token_id_) positions.push_back(base + i);
    }
    if (positions.empty()) {
        throw std::runtime_error("gemma4: an image was supplied but the prompt carries no image token (id " +
                                 std::to_string(image_token_id_) + ")");
    }
    return positions;
}

// ═════════════════════════════════════════════════════════════════════════════
// Factories
// ═════════════════════════════════════════════════════════════════════════════
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
        const auto llm_cfg = withEmbeddingPath(config.llm_config, gc);

        auto model = std::make_unique<Gemma4VLMModel>(std::move(spec), gc);
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

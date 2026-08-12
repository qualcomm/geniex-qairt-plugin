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
// Contents, in dependency order:
//   buildGemma4Providers()  the extra CPU-side input providers the decoder
//                           needs (per-layer embedding + dual RoPE)
//   Gemma4VisionEncoder     the Visual Embedding Generator (VEG) graph
//   Gemma4VLMModel          the decoder            -> makeVLMPipeline()

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "graph.h"
#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "model.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_model.h"

namespace geniex {
namespace gemma4 {

// ═════════════════════════════════════════════════════════════════════════════
// Shared provider wiring
//
// The decoder needs three extra CPU-side providers on top of what
// LLMModel::createInputProviders() installs, all resolved from genie_config.json.
// They are built here rather than inline so the wiring stays reviewable in one
// place — two of the three (the per-layer stream and the local RoPE table) are
// invisible in output quality until they are subtly wrong.
// ═════════════════════════════════════════════════════════════════════════════
namespace detail {

// Resolves a genie_config path against the bundle when it is not absolute.
inline std::filesystem::path resolveBundlePath(const std::filesystem::path& bundle_dir, const std::string& p) {
    std::filesystem::path pp(p);
    return pp.is_absolute() ? pp : (bundle_dir / pp);
}

// head_dim for a RoPE tensor pair, read off the first shard that exposes it.
// The graph's cos tensor carries head_dim/2 in its last dim.
inline size_t discoverHeadDim(
    size_t shard_count, const std::function<const Graph&(size_t)>& shard_graph, const char* cos_tensor) {
    for (size_t s = 0; s < shard_count; ++s) {
        const Graph& g = shard_graph(s);
        if (g.hasInput(cos_tensor)) return g.inputSpec(cos_tensor).shape.back() * 2;
    }
    return 0;
}

// Local (sliding-window) RoPE bound to the swa_position_ids_* tensors. Mirrors
// makeRoPEProvider's variant handling but with the local theta/scaling.
inline std::unique_ptr<InputProvider> makeLocalRoPEProvider(const ParsedGenieConfig& gc, size_t head_dim) {
    return std::visit(
        [&](const auto& s) -> std::unique_ptr<InputProvider> {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, PartialRopeScaling>) {
                return std::make_unique<PartialRoPEInputProvider>(head_dim,
                    gc.local_rope_theta,
                    s.rope_fraction,
                    s.scale,
                    "swa_position_ids_cos",
                    "swa_position_ids_sin");
            } else {
                // Gemma local layers use plain (full-rotary) RoPE.
                return std::make_unique<RoPEInputProvider>(
                    head_dim, gc.local_rope_theta, "swa_position_ids_cos", "swa_position_ids_sin");
            }
        },
        gc.local_rope_scaling);
}

}  // namespace detail

// The extra providers a Gemma4 decoder needs. Any member may be null when the
// bundle does not describe the corresponding stream.
struct Gemma4Providers {
    // Auxiliary per-token embedding stream (`per_layer_inputs`). Returned as the
    // concrete type because the decoder must keep a handle on it to redirect
    // image positions to PAD — see Gemma4VLMModel::prepareEmbeddings().
    std::unique_ptr<EmbeddingInputProvider> perlayer;
    std::unique_ptr<InputProvider>          local_rope;
    std::unique_ptr<InputProvider>          global_rope;
};

// Builds the three providers described above.
//
// `shard_graph(s)` must return shard s's prefill graph; the RoPE head dims are
// read from its tensors rather than assumed, since they differ across variants
// (E2B vs E4B) and across exports.
inline Gemma4Providers buildGemma4Providers(const ParsedGenieConfig& gc, const std::filesystem::path& bundle_dir,
    size_t shard_count, const std::function<const Graph&(size_t)>& shard_graph) {
    Gemma4Providers out;

    // (1) Per-layer embedding stream. A second embedding table feeding
    // `per_layer_inputs`; its row width is num_layers * per_layer_dim, NOT
    // spec.hidden_size, so it uses the explicit-config EmbeddingInputProvider.
    if (gc.perlayer_embedding_lut_path && gc.perlayer_embedding_size > 0) {
        const std::filesystem::path lut      = detail::resolveBundlePath(bundle_dir, *gc.perlayer_embedding_lut_path);
        auto                        provider = std::make_unique<EmbeddingInputProvider>(
            /*tensor_name=*/"per_layer_inputs",
            /*table_path=*/lut.string(),
            /*row_hidden_size=*/gc.perlayer_embedding_size,
            /*pad_token_override=*/gc.pad_token_id >= 0 ? gc.pad_token_id : 0);
        // This is the table that makes the in-RAM path impossible: E2B's is
        // 2.35 GB as int8 and 9.4 GB dequantized. Quantized => mmap.
        if (gc.perlayer_embedding_quant.quantized()) {
            provider->setQuantization(gc.perlayer_embedding_quant);
        }
        out.perlayer = std::move(provider);
        GENIEX_LOG_INFO(
            "gemma4: per-layer embedding provider ({} dims) -> {}", gc.perlayer_embedding_size, lut.string());
    }

    // (2) Local (swa) RoPE.
    if (gc.local_positional_encoding_present) {
        const size_t local_head_dim = detail::discoverHeadDim(shard_count, shard_graph, "swa_position_ids_cos");
        if (local_head_dim > 0) {
            out.local_rope = detail::makeLocalRoPEProvider(gc, local_head_dim);
            GENIEX_LOG_INFO(
                "gemma4: local (swa) RoPE provider head_dim={} theta={}", local_head_dim, gc.local_rope_theta);
        } else {
            GENIEX_LOG_WARN("gemma4: local-positional-encoding set but no swa_position_ids_cos tensor found");
        }
    }

    // (3) Global RoPE — present when global-attention layers expose a separate
    // `position_ids_global_cos/sin` pair. Partial-rotary is applied inside the
    // graph, so the CPU-side table is plain full RoPE (not makeRoPEProvider).
    {
        const size_t global_head_dim = detail::discoverHeadDim(shard_count, shard_graph, "position_ids_global_cos");
        if (global_head_dim > 0) {
            out.global_rope = std::make_unique<RoPEInputProvider>(
                global_head_dim, gc.rope_theta, "position_ids_global_cos", "position_ids_global_sin");
            GENIEX_LOG_INFO("gemma4: global RoPE provider head_dim={} theta={}", global_head_dim, gc.rope_theta);
        }
    }

    return out;
}

// Gemma feeds CPU-side embeddings (`inputs_embeds`), so the base embedding
// provider needs model_cfg.embedding_path pointing at the MAIN embedding LUT.
// modelConfigFromDirectory doesn't set it (most on-device-embedding models don't
// need it), so resolve it here from genie_config's dialog.embedding.lut-path.
inline ModelConfig withEmbeddingPath(
    ModelConfig cfg, const ParsedGenieConfig& gc, const std::filesystem::path& bundle) {
    if (!cfg.embedding_path && gc.embedding_lut_path) {
        cfg.embedding_path = detail::resolveBundlePath(bundle, *gc.embedding_lut_path).string();
    }
    return cfg;
}

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
    Gemma4VLMModel(LLMSpec spec, ParsedGenieConfig gc, std::filesystem::path bundle_dir)
        : VLMModel(std::move(spec), std::move(gc)), bundle_dir_(std::move(bundle_dir)) {}

    void setVisionEncoder(std::unique_ptr<Gemma4VisionEncoder> vis) { vision_encoder_ = std::move(vis); }
    void setImageTokenId(int32_t image_token) { image_token_id_ = image_token; }

   protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override {
        if (!vision_encoder_) throw std::runtime_error("gemma4: no vision encoder registered");
        return vision_encoder_->encode(pixel_data);
    }

    // The extra providers the decoder needs beyond the base factory's. We never
    // call setEmbeddingProvider(), so LLMModel::createInputProviders() still
    // installs the quantized memory-mapped `inputs_embeds` provider rather than
    // VLMModel's in-RAM default, and we keep a handle on it for the splice.
    void createInputProviders() override {
        LLMModel::createInputProviders();

        main_embed_provider_ = findEmbeddingProvider("inputs_embeds");
        if (!main_embed_provider_) main_embed_provider_ = findEmbeddingProvider("input_embeds");
        if (!main_embed_provider_) {
            GENIEX_LOG_ERROR("gemma4 VLM: main embedding provider NOT FOUND — vision splice unavailable");
        } else {
            // Gemma4's embedding LUT was calibrated with round-to-nearest, so the
            // float->UFIXED write must not truncate; the core default is
            // TowardZero. Not opting in shifts every embedding row by up to one
            // quantization step and degrades output quality.
            main_embed_provider_->setRoundingMode(RoundingMode::Nearest);
        }

        auto extra = buildGemma4Providers(
            gc_, bundle_dir_, shard_count_, [this](size_t s) -> const Graph& { return graph(graphIndex(0, s, 0)); });

        if (extra.perlayer) {
            perlayer_embed_provider_ = extra.perlayer.get();
            perlayer_embed_provider_->setRoundingMode(RoundingMode::Nearest);
            input_providers_.push_back(std::move(extra.perlayer));
        }
        if (extra.local_rope) input_providers_.push_back(std::move(extra.local_rope));
        if (extra.global_rope) input_providers_.push_back(std::move(extra.global_rope));
    }

    // Runs the encoder and installs its rows as a positional override on the two
    // embedding streams, instead of materialising the prompt's text embeddings.
    void prepareEmbeddings(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input) override {
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

        // The encoder's soft tokens must line up 1:1 with the prompt's image
        // tokens, in order; otherwise the override would land on text positions.
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

    void releaseEmbeddings() override {
        if (main_embed_provider_) main_embed_provider_->clearEmbeddingOverride();
        if (perlayer_embed_provider_) perlayer_embed_provider_->clearTokenSubstitution();
    }

   private:
    // Absolute positions of EVERY image token in the prompt, in order, offset by
    // `base` (the turn's starting KV position).
    //
    // Deliberately not "first contiguous run": several attachments in one turn
    // produce several runs, separated by the template's eoi/boi text. Scanning
    // for a single run would return only the first image's span, and the 1:1
    // check against the encoder's row count would then reject the whole turn.
    std::vector<size_t> findImagePositions(const std::vector<int32_t>& ids, size_t base) const {
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

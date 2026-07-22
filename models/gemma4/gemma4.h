// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace gemma4 {

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
// The subclass' only job is to append the two extra CPU-side providers the base
// factory doesn't know about (per-layer embedding + local RoPE); everything else
// (shards, both KV blocks, both masks, global RoPE, main embedding) is inferred
// from the graphs and genie_config.json exactly like the other families.
class Gemma4Model : public LLMModel {
   public:
    Gemma4Model(LLMSpec spec, ParsedGenieConfig gc, std::filesystem::path bundle_dir)
        : LLMModel(std::move(spec), std::move(gc)), bundle_dir_(std::move(bundle_dir)) {}

    // Splices a vision encoder's soft tokens into `inputs_embeds` at the prompt
    // positions occupied by image tokens. `rows` is flat [n_tokens * hidden].
    //
    // `per_layer_inputs` must be redirected at the same positions. It is a plain
    // LUT lookup on the input ids, but Gemma4Model.forward rewrites multimodal
    // positions to the pad id *before* that lookup:
    //     llm_input_ids = where(multimodal_mask, pad_token_id, llm_input_ids)
    // Letting the image token's own row through instead corrupts the per-layer
    // input at all 35 layers for every image position, and the model then replies
    // as though no image were attached ("Please provide the image you are
    // referring to"). Verified on the Genie reference pipeline: image-token row
    // and dequantized-zero both refuse; the PAD row describes the image.
    void setVisionEmbeddings(size_t start_position, std::vector<float> rows) {
        if (!main_embed_provider_) {
            throw std::runtime_error("gemma4: main embedding provider not available");
        }
        const size_t n_rows = spec_.hidden_size ? rows.size() / spec_.hidden_size : 0;
        main_embed_provider_->setEmbeddingOverride(start_position, std::move(rows));
        if (perlayer_embed_provider_ && n_rows) {
            const int32_t pad = gc_.pad_token_id >= 0 ? gc_.pad_token_id : 0;
            perlayer_embed_provider_->setTokenSubstitution(start_position, n_rows, pad);
        }
    }

    void clearVisionEmbeddings() {
        if (main_embed_provider_) main_embed_provider_->clearEmbeddingOverride();
        if (perlayer_embed_provider_) perlayer_embed_provider_->clearTokenSubstitution();
    }

   protected:
    void createInputProviders() override {
        // Base installs: main EmbeddingInputProvider("inputs_embeds") + one RoPE
        // provider bound to position_ids_cos/sin (PartialRoPE for Gemma's global
        // layers, from the "proportional" rope-scaling in genie_config.json).
        LLMModel::createInputProviders();

        // Remember the main embedding provider so the VLM path can splice vision
        // embeddings into it. Resolved by tensor name inside geniex_core, since
        // an RTTI match here would have to cross the DLL boundary.
        main_embed_provider_ = findEmbeddingProvider("inputs_embeds");
        if (!main_embed_provider_) main_embed_provider_ = findEmbeddingProvider("input_embeds");
        GENIEX_LOG_INFO("gemma4: main embedding provider {}",
            main_embed_provider_ ? "resolved (vision splice available)" : "NOT FOUND");

        // (1) Per-layer embedding stream. A second embedding table feeding
        // `per_layer_inputs`; its row width is num_layers * per_layer_dim, NOT
        // spec_.hidden_size, so it uses the explicit-config EmbeddingInputProvider.
        if (gc_.perlayer_embedding_lut_path && gc_.perlayer_embedding_size > 0) {
            const std::filesystem::path lut      = resolvePath(*gc_.perlayer_embedding_lut_path);
            auto                        provider = std::make_unique<EmbeddingInputProvider>(
                /*tensor_name=*/"per_layer_inputs",
                /*table_path=*/lut.string(),
                /*row_hidden_size=*/gc_.perlayer_embedding_size,
                /*pad_token_override=*/gc_.pad_token_id >= 0 ? gc_.pad_token_id : 0);
            // This is the table that makes the in-RAM path impossible: E2B's is
            // 2.35 GB as int8 and 9.4 GB dequantized. Quantized => mmap.
            if (gc_.perlayer_embedding_quant.quantized()) {
                provider->setQuantization(gc_.perlayer_embedding_quant);
            }
            // Kept so setVisionEmbeddings() can redirect image positions to PAD.
            perlayer_embed_provider_ = provider.get();
            input_providers_.push_back(std::move(provider));
            GENIEX_LOG_INFO(
                "gemma4: per-layer embedding provider ({} dims) -> {}", gc_.perlayer_embedding_size, lut.string());
        }

        // (2) Local (swa) RoPE. head_dim derived from the swa cos tensor (last dim = head_dim/2).
        if (gc_.local_positional_encoding_present) {
            size_t local_head_dim = 0;
            for (size_t s = 0; s < shard_count_; ++s) {
                const Graph& g = graph(graphIndex(0, s, 0));
                if (g.hasInput("swa_position_ids_cos")) {
                    local_head_dim = g.inputSpec("swa_position_ids_cos").shape.back() * 2;
                    break;
                }
            }
            if (local_head_dim > 0) {
                input_providers_.push_back(makeLocalRoPEProvider(local_head_dim));
                GENIEX_LOG_INFO(
                    "gemma4: local (swa) RoPE provider head_dim={} theta={}", local_head_dim, gc_.local_rope_theta);
            } else {
                GENIEX_LOG_WARN("gemma4: local-positional-encoding set but no swa_position_ids_cos tensor found");
            }
        }

        // (3) Global RoPE. This export uses `position_ids_global_cos/sin` rather than
        // the single-stream `position_ids_cos` the base class binds. Partial-rotary is
        // applied inside the graph, so the CPU-side table is plain full RoPE.
        {
            size_t global_head_dim = 0;
            for (size_t s = 0; s < shard_count_; ++s) {
                const Graph& g = graph(graphIndex(0, s, 0));
                if (g.hasInput("position_ids_global_cos")) {
                    global_head_dim = g.inputSpec("position_ids_global_cos").shape.back() * 2;
                    break;
                }
            }
            if (global_head_dim > 0) {
                input_providers_.push_back(std::make_unique<RoPEInputProvider>(
                    global_head_dim, gc_.rope_theta, "position_ids_global_cos", "position_ids_global_sin"));
                GENIEX_LOG_INFO(
                    "gemma4: global RoPE provider head_dim={} theta={}", global_head_dim, gc_.rope_theta);
            }
        }
    }

   private:
    std::filesystem::path bundle_dir_;

    // Non-owning; owned by input_providers_. Set in createInputProviders().
    EmbeddingInputProvider* main_embed_provider_     = nullptr;
    EmbeddingInputProvider* perlayer_embed_provider_ = nullptr;

    std::filesystem::path resolvePath(const std::string& p) const {
        std::filesystem::path pp(p);
        return pp.is_absolute() ? pp : (bundle_dir_ / pp);
    }

    // Builds the local RoPE provider from gc_.local_rope_* bound to the
    // swa_position_ids_* tensors. Mirrors makeRoPEProvider's variant handling,
    // but with the local theta/scaling and the swa tensor names.
    std::unique_ptr<InputProvider> makeLocalRoPEProvider(size_t head_dim) const {
        return std::visit(
            [&](const auto& s) -> std::unique_ptr<InputProvider> {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<T, PartialRopeScaling>) {
                    return std::make_unique<PartialRoPEInputProvider>(head_dim,
                        gc_.local_rope_theta,
                        s.rope_fraction,
                        s.scale,
                        "swa_position_ids_cos",
                        "swa_position_ids_sin");
                } else {
                    // Gemma local layers use plain (full-rotary) RoPE.
                    return std::make_unique<RoPEInputProvider>(
                        head_dim, gc_.local_rope_theta, "swa_position_ids_cos", "swa_position_ids_sin");
                }
            },
            gc_.local_rope_scaling);
    }
};

// Gemma feeds CPU-side embeddings (`inputs_embeds`), so the base embedding
// provider needs model_cfg.embedding_path pointing at the MAIN embedding LUT.
// modelConfigFromDirectory doesn't set it (most on-device-embedding models don't
// need it), so resolve it here from genie_config's dialog.embedding.lut-path.
inline ModelConfig withEmbeddingPath(
    ModelConfig cfg, const ParsedGenieConfig& gc, const std::filesystem::path& bundle) {
    if (!cfg.embedding_path && gc.embedding_lut_path) {
        std::filesystem::path p(*gc.embedding_lut_path);
        cfg.embedding_path = (p.is_absolute() ? p : (bundle / p)).string();
    }
    return cfg;
}

// Returns the concrete Gemma4Model (NOT sliced to LLMModel) so LLMPipeline's
// templated create<ModelT> preserves the createInputProviders() override.
inline Gemma4Model makeModel(const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       gc     = parseGenieConfig(bundle);
    auto       spec   = buildSpecSkeleton(gc);
    // Gemma carries a second (sliding-window) KV cache; core inferSpecFromGraphs
    // auto-detects and appends it from the swa_key_* tensors, so we don't need to
    // declare it here.
    return Gemma4Model(std::move(spec), std::move(gc), bundle);
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    try {
        const auto bundle = bundleDirOf(model_cfg);
        auto       gc     = parseGenieConfig(bundle);
        const auto cfg    = withEmbeddingPath(model_cfg, gc, bundle);

        LLMPipeline pipe;
        if (!pipe.create(makeModel(cfg), runtime_cfg, cfg)) return std::nullopt;
        pipe.setBosTokenId(gc.bos_token_id);
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("gemma4::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace gemma4
}  // namespace geniex

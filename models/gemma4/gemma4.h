// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "gemma4/gemma4_common.h"
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
        if (main_embed_provider_) main_embed_provider_->setRoundingMode(RoundingMode::Nearest);
        GENIEX_LOG_INFO("gemma4: main embedding provider {}",
            main_embed_provider_ ? "resolved (vision splice available)" : "NOT FOUND");

        // The per-layer embedding stream and the two RoPE tables are identical in
        // the text and multimodal paths, so they are built by the shared helper.
        auto extra = buildGemma4Providers(
            gc_, bundle_dir_, shard_count_, [this](size_t s) -> const Graph& { return graph(graphIndex(0, s, 0)); });

        if (extra.perlayer) {
            // Kept so setVisionEmbeddings() can redirect image positions to PAD.
            perlayer_embed_provider_ = extra.perlayer.get();
            perlayer_embed_provider_->setRoundingMode(RoundingMode::Nearest);
            input_providers_.push_back(std::move(extra.perlayer));
        }
        if (extra.local_rope) input_providers_.push_back(std::move(extra.local_rope));
        if (extra.global_rope) input_providers_.push_back(std::move(extra.global_rope));
    }

   private:
    std::filesystem::path bundle_dir_;

    // Non-owning; owned by input_providers_. Set in createInputProviders().
    EmbeddingInputProvider* main_embed_provider_     = nullptr;
    EmbeddingInputProvider* perlayer_embed_provider_ = nullptr;
};

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

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "graph.h"
#include "llm/input_provider.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "types.h"

namespace geniex {
namespace gemma4 {

// Provider wiring shared by the text-only decoder (Gemma4Model) and the
// multimodal one (Gemma4VLMModel).
//
// Both need the same extra CPU-side providers on top of what
// LLMModel::createInputProviders() installs, resolved from the same
// genie_config.json fields. Constructing them here keeps the VLM path from
// drifting from the text path -- a real risk, because the per-layer stream and
// the local RoPE table are invisible in output quality until subtly wrong.
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
    // concrete type because the vision path must keep a handle on it to redirect
    // image positions to PAD — see Gemma4Model::setVisionEmbeddings().
    std::unique_ptr<EmbeddingInputProvider> perlayer;
    std::unique_ptr<InputProvider>          local_rope;
    std::unique_ptr<InputProvider>          global_rope;
};

// Builds the providers described above.
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

}  // namespace gemma4
}  // namespace geniex

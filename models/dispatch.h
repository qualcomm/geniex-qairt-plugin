// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// model_id-driven pipeline dispatcher.
//
// Replaces the per-variant llm_model_registry / vlm_model_registry tables: the
// runtime reads `metadata.json`'s `model_id` field and routes to the matching
// family factory via prefix matching. Adding a new variant of an existing
// family requires no source change — drop the bundle into modelfiles/<name>/
// and call makeLLMPipeline / makeVLMPipeline.
//
// Dispatch rules (in priority order):
//   metadata.json `model_id` prefix          → factory
//   ─────────────────────────────────────────────────────────────
//   qwen2_5_vl_*                             → qwen2_5_vl::makePipeline (VLM)
//   qwen3_vl_*                               → qwen3_vl::makePipeline   (VLM)
//   qwen3_*                                  → qwen3::makePipeline
//   qwen2_5_*                                → qwen2_5::makePipeline
//   falcon_v3_*                              → falcon3::makePipeline
//   llama_v3_*_ssd                           → llama3_2_3b_ssd::makePipeline
//   llama_v3_*                               → llama3::makePipeline
//   phi_3_5_*                                → phi3_5::makePipeline
//
// LLM vs VLM, SSD vs plain Llama, and Falcon3 vs Llama-3 are all decided
// purely from `model_id`. We do not need `dialog.type`, the bundle's per-
// graph filename pattern (ar*_cl* vs partN_of_M.bin), or
// `architectures` / `_name_or_path` from config.json.

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "falcon3/falcon3.h"
#include "llama3/llama3.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "phi3_5/phi3_5.h"
#include "pipeline/llm_pipeline.h"
#include "pipeline/vlm_pipeline.h"
#include "qwen2_5/qwen2_5.h"
#include "qwen2_5_vl/qwen2_5_vl.h"
#include "qwen3/qwen3.h"
#include "qwen3_vl/qwen3_vl.h"
#include "types.h"

namespace geniex {

namespace dispatch_detail {

inline std::string modelIdOf(const ModelConfig& model_cfg) {
    try {
        const auto bundle = bundleDirOf(model_cfg);
        return parseQAIRTMetadata(bundle).model_id;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("dispatch: cannot read metadata.json: {}", e.what());
        return {};
    }
}

inline bool startsWith(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

inline bool endsWith(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Auto-discovers the SSD forecast prefix file under the bundle if the caller
// didn't set model_cfg.forecast_prefix_path explicitly.
inline ModelConfig autoDiscoverForecastPrefix(ModelConfig model_cfg) {
    if (model_cfg.forecast_prefix_path.has_value()) return model_cfg;
    try {
        const auto bundle    = bundleDirOf(model_cfg);
        const auto candidate = bundle / "forecast-prefix" / "kv-cache.primary.qnn-htp";
        if (std::filesystem::exists(candidate)) {
            model_cfg.forecast_prefix_path = candidate.string();
        }
    } catch (...) {
    }
    return model_cfg;
}

}  // namespace dispatch_detail

// Single LLM entry point. Routes by metadata.json's `model_id` prefix.
// Returns std::nullopt for unknown / VLM model_ids.
inline std::optional<LLMPipeline> makeLLMPipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg_in) {
    using namespace dispatch_detail;
    const std::string model_id = modelIdOf(model_cfg_in);
    if (model_id.empty()) return std::nullopt;

    // SSD: model_id ends in "_ssd". Auto-populate the forecast prefix path.
    if (endsWith(model_id, "_ssd")) {
        const auto cfg = autoDiscoverForecastPrefix(model_cfg_in);
        return llama3_2_3b_ssd::makePipeline(runtime_cfg, cfg);
    }

    if (startsWith(model_id, "qwen3_") && !startsWith(model_id, "qwen3_vl_"))
        return qwen3::makePipeline(runtime_cfg, model_cfg_in);
    if (startsWith(model_id, "qwen2_5_")) return qwen2_5::makePipeline(runtime_cfg, model_cfg_in);
    if (startsWith(model_id, "falcon_v3_")) return falcon3::makePipeline(runtime_cfg, model_cfg_in);
    if (startsWith(model_id, "llama_v3_")) return llama3::makePipeline(runtime_cfg, model_cfg_in);
    if (startsWith(model_id, "phi_3_5_")) return phi3_5::makePipeline(runtime_cfg, model_cfg_in);

    GENIEX_LOG_ERROR("dispatch: no LLM factory matches model_id '{}'", model_id);
    return std::nullopt;
}

// Single VLM entry point. Routes by metadata.json's `model_id` prefix.
inline std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    using namespace dispatch_detail;
    const std::string model_id = modelIdOf(config.llm_config);
    if (model_id.empty()) return std::nullopt;

    if (startsWith(model_id, "qwen2_5_vl_")) return qwen2_5_vl::makePipeline(runtime_cfg, config);
    if (startsWith(model_id, "qwen3_vl_")) return qwen3_vl::makePipeline(runtime_cfg, config);

    GENIEX_LOG_ERROR("dispatch: no VLM factory matches model_id '{}'", model_id);
    return std::nullopt;
}

}  // namespace geniex

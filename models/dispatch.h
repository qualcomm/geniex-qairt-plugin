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
//   makeVLMPipeline:
//   qwen2_5_vl_*                             → qwen2_5_vl::makePipeline
//   qwen3_vl_*                               → qwen3_vl::makePipeline
//   intern3_5_vl_*                           → intern3_5_vl::makePipeline
//   gemma_4_*                                → gemma4::makeVLMPipeline
//
//   makeLLMPipeline:
//   llama_v3_*_ssd                           → llama3_2_3b_ssd::makePipeline
//   gemma_4_*                                → gemma4::makePipeline
//   (vision bundle)                          → refused; use makeVLMPipeline
//   (dialog.type != "basic")                 → refused; no factory here
//   everything else                          → llm_family::makePipeline
//
// The LLM table only lists what needs a *specialized* factory; plain
// decoder-only families are the fallback and appear nowhere. They carry no code
// of their own either — they share llm_family::makePipeline
// (core/include/pipeline/llm_family.h). Adding one is nothing at all: drop the
// bundle in. The lone exception is BOS, which Tokenizer does not expose, so the
// qwen3_ prefix survives to set it.
//
// The fallback is guarded rather than unconditional. A vision or speculative
// bundle would otherwise build a pipeline that loads and runs while computing
// the wrong thing, so both are detected from the bundle's own sidecars and
// refused with an error — see makeLLMPipeline.
//
// The two tables are independent, and a family may appear in both, as gemma_4_*
// does: one bundle serves text-only and multimodal use, and the entry point is
// chosen upstream from the bundle's `supports_vision` flag, not here.
//
// SSD vs plain Llama and Falcon3 vs Llama-3 are decided purely from `model_id`.
// LLM vs VLM and plain vs speculative additionally consult the bundle's vision
// fields and `dialog.type`. We do not need the bundle's per-graph filename
// pattern (ar*_cl* vs partN_of_M.bin), or
// `architectures` / `_name_or_path` from config.json.

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "gemma4/gemma4.h"
#include "gemma4/gemma4_vlm.h"
#include "intern3_5_vl/intern3_5_vl.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "pipeline/llm_family.h"
#include "pipeline/llm_pipeline.h"
#include "pipeline/vlm_pipeline.h"
#include "qwen2_5_vl/qwen2_5_vl.h"
#include "qwen3_vl/qwen3_vl.h"
#include "types.h"

namespace geniex {

namespace dispatch_detail {

// What dispatch needs from a bundle's sidecars, read in one pass.
struct BundleFacts {
    std::string model_id;
    bool        multimodal  = false;  // ships a vision tower
    std::string dialog_type = "basic";
};

inline std::optional<BundleFacts> bundleFactsOf(const ModelConfig& model_cfg) {
    try {
        const auto bundle = bundleDirOf(model_cfg);
        const auto meta   = parseQAIRTMetadata(bundle);

        BundleFacts f;
        f.model_id   = meta.model_id;
        f.multimodal = !meta.vision_encoder_graph.empty() || meta.vision_preprocessing.has_value();
        // parseGenieConfig tolerates a missing file, so this cannot throw on
        // bundles that ship no genie_config.json.
        f.dialog_type = parseGenieConfig(bundle).dialog_type;
        return f;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("dispatch: cannot read metadata.json: {}", e.what());
        return std::nullopt;
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

// Single LLM entry point. Routes the families that need a specialized factory,
// then falls back to the generic one, so a plain decoder-only bundle needs no
// entry here at all.
inline std::optional<LLMPipeline> makeLLMPipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg_in) {
    using namespace dispatch_detail;
    const auto facts = bundleFactsOf(model_cfg_in);
    if (!facts) return std::nullopt;
    const std::string& model_id = facts->model_id;

    // ── Families needing a specialized factory ──────────────────────────────
    // SSD: model_id ends in "_ssd". Auto-populate the forecast prefix path.
    if (endsWith(model_id, "_ssd")) {
        const auto cfg = autoDiscoverForecastPrefix(model_cfg_in);
        return llama3_2_3b_ssd::makePipeline(runtime_cfg, cfg);
    }

    if (startsWith(model_id, "gemma_4_")) return gemma4::makePipeline(runtime_cfg, model_cfg_in);

    // ── Bundles the generic factory must not silently accept ────────────────
    // Refused rather than fallen through, because the generic factory would
    // build a pipeline that loads and runs but computes the wrong thing:
    //
    //  - A vision bundle needs its encoder and (Qwen-VL) a 3-D MRoPE provider.
    //    The generic path wires plain RoPE, so positions would be wrong with no
    //    error anywhere. Use makeVLMPipeline for these.
    //  - A non-"basic" dialog.type (ssd-q1, eaglet, spd, lade, kv-share,
    //    multistream) selects a speculative or multi-engine decode loop that
    //    LLMModel does not implement.
    //
    // Both are read from the bundle rather than pattern-matched on model_id, so
    // a new multimodal or speculative family is caught without touching this
    // file.
    if (facts->multimodal) {
        GENIEX_LOG_ERROR("dispatch: '{}' is a multimodal bundle; use makeVLMPipeline", model_id);
        return std::nullopt;
    }
    if (facts->dialog_type != "basic") {
        GENIEX_LOG_ERROR(
            "dispatch: '{}' needs dialog.type '{}', which has no LLM factory here", model_id, facts->dialog_type);
        return std::nullopt;
    }

    // ── Everything else: plain decoder-only, fully described by its bundle ───
    // The lone remaining per-family knob is BOS, which Tokenizer does not
    // expose; Qwen3's model needs one its chat template does not emit.
    const bool prepend_bos = startsWith(model_id, "qwen3_");
    return llm_family::makePipeline(runtime_cfg, model_cfg_in, {prepend_bos});
}

// Single VLM entry point. Routes by metadata.json's `model_id` prefix.
inline std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    using namespace dispatch_detail;
    const auto facts = bundleFactsOf(config.llm_config);
    if (!facts) return std::nullopt;
    const std::string& model_id = facts->model_id;

    if (startsWith(model_id, "qwen2_5_vl_")) return qwen2_5_vl::makePipeline(runtime_cfg, config);
    if (startsWith(model_id, "qwen3_vl_")) return qwen3_vl::makePipeline(runtime_cfg, config);
    if (startsWith(model_id, "intern3_5_vl_")) return intern3_5_vl::makePipeline(runtime_cfg, config);
    if (startsWith(model_id, "gemma_4_")) return gemma4::makeVLMPipeline(runtime_cfg, config);

    GENIEX_LOG_ERROR("dispatch: no VLM factory matches model_id '{}'", model_id);
    return std::nullopt;
}

}  // namespace geniex

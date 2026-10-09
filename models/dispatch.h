// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Bundle-driven pipeline dispatcher: routes by metadata.json's `model_id`,
// `architectures` (falling back to config.json's `architectures[0]` for
// bundles that predate that field), `geniex.dialog_type` (assumed "basic"
// for a bundle that predates the `geniex` block), and `geniex.supports_vision`
// (falling back to vision_encoder_graph/vision_preprocessing otherwise).
//
//   makeVLMPipeline (architecture OR model_id prefix; model_id prefix matching will be removed in the future):
//   Qwen2_5_VLForConditionalGeneration | qwen2_5_vl_*   → qwen2_5_vl
//   Qwen3VLForConditionalGeneration    | qwen3_vl_*     → qwen3_vl
//   InternVLChatModel                  | intern3_5_vl_* → intern3_5_vl
//   Gemma4ForConditionalGeneration     | gemma_4_*      → gemma4::makeVLMPipeline
//
//   makeLLMPipeline:
//   llama_v3_*_ssd                             → llama3_2_3b_ssd::makePipeline
//   (vision bundle)                            → refused; use makeVLMPipeline
//   (dialog.type != "basic")                   → refused; no factory here
//   Phi3/Qwen2/Llama/Qwen3ForCausalLM          → auto_llm (named rows, same factory)
//   anything else                              → auto_llm
//

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "gemma4/gemma4.h"
#include "intern3_5_vl/intern3_5_vl.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "pipeline/auto_llm.h"
#include "pipeline/llm_pipeline.h"
#include "pipeline/vlm_pipeline.h"
#include "qwen2_5_vl/qwen2_5_vl.h"
#include "qwen3_vl/qwen3_vl.h"
#include "types.h"

namespace geniex {

namespace dispatch_detail {

struct BundleFacts {
    std::string model_id;
    bool        multimodal  = false;
    std::string dialog_type = "basic";
    std::string architecture;  // metadata.json's architectures (or config.json's architectures[0]); "" if absent
};

inline std::optional<BundleFacts> bundleFactsOf(const ModelConfig& model_cfg) {
    try {
        const auto bundle = bundleDirOf(model_cfg);
        const auto meta   = parseQAIRTMetadata(bundle);

        BundleFacts f;
        f.model_id = meta.model_id;
        // geniex.supports_vision, falling back to the structural heuristic for
        // a bundle that predates the field.
        f.multimodal =
            meta.supports_vision.value_or(!meta.vision_encoder_graph.empty() || meta.vision_preprocessing.has_value());
        // meta.dialog_type stays "basic" for a bundle whose metadata.json
        // carries no `geniex` block at all (predates the migration); report
        // it rather than silently reading genie_config.json.
        f.dialog_type = !meta.dialog_type.empty() ? meta.dialog_type : "basic";
        if (meta.dialog_type.empty()) {
            GENIEX_LOG_WARN(
                "dispatch: metadata.json in {} has no geniex.dialog_type; assuming \"basic\"", bundle.string());
        }
        // meta.architecture is empty when the bundle predates the `architectures`
        // field -- fall back to config.json's architectures[0] (not a
        // genie_config.json read).
        f.architecture = !meta.architecture.empty() ? meta.architecture : parseModelArchitecture(bundle);
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

inline std::optional<LLMPipeline> makeLLMPipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg_in) {
    using namespace dispatch_detail;
    const auto facts = bundleFactsOf(model_cfg_in);
    if (!facts) return std::nullopt;
    const std::string& model_id = facts->model_id;

    if (endsWith(model_id, "_ssd")) {
        const auto cfg = autoDiscoverForecastPrefix(model_cfg_in);
        return llama3_2_3b_ssd::makePipeline(runtime_cfg, cfg);
    }
    if (facts->multimodal) {
        GENIEX_LOG_ERROR("dispatch: '{}' is a multimodal bundle; use makeVLMPipeline", model_id);
        return std::nullopt;
    }
    if (facts->dialog_type != "basic") {
        GENIEX_LOG_ERROR(
            "dispatch: '{}' needs dialog.type '{}', which has no LLM factory here", model_id, facts->dialog_type);
        return std::nullopt;
    }

    if (facts->architecture == "Phi3ForCausalLM") {  // Phi model family
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "Qwen2ForCausalLM") {  // Qwen2 model family
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "LlamaForCausalLM") {  // Llama3 model family
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "Qwen3ForCausalLM") {  // Qwen3 model family, need to prepend BOS
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in, {/*prepend_bos=*/true});
    }

    return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
}

inline std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    using namespace dispatch_detail;
    const auto facts = bundleFactsOf(config.llm_config);
    if (!facts) return std::nullopt;
    const std::string& model_id = facts->model_id;

    if (facts->architecture == "Qwen2_5_VLForConditionalGeneration" || startsWith(model_id, "qwen2_5_vl_")) {
        return qwen2_5_vl::makePipeline(runtime_cfg, config);
    }
    if (facts->architecture == "Qwen3VLForConditionalGeneration" || startsWith(model_id, "qwen3_vl_")) {
        return qwen3_vl::makePipeline(runtime_cfg, config);
    }
    if (facts->architecture == "InternVLChatModel" || startsWith(model_id, "intern3_5_vl_")) {
        return intern3_5_vl::makePipeline(runtime_cfg, config);
    }
    if (facts->architecture == "Gemma4ForConditionalGeneration" || startsWith(model_id, "gemma_4_")) {
        return gemma4::makeVLMPipeline(runtime_cfg, config);
    }

    GENIEX_LOG_ERROR("dispatch: no VLM factory matches model_id '{}'", model_id);
    return std::nullopt;
}

}  // namespace geniex

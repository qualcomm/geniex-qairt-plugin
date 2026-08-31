// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Bundle-driven pipeline dispatcher. Reads facts straight from the bundle --
// metadata.json's `model_id`, config.json's `architectures[0]`,
// genie_config.json's `dialog.type` -- and routes to the matching factory.
// Adding a variant of an existing family needs no source change: drop the
// bundle into modelfiles/<name>/ and call makeLLMPipeline / makeVLMPipeline.
//
// Plain LLM families route on `architectures[0]` (e.g. "Phi3ForCausalLM")
// rather than `model_id`: it's the standard HuggingFace field AIHM's exports
// already carry, so a model_id naming change on that side can't break
// dispatch here. SSD and VLM-family routing stay on `model_id` -- neither a
// decode strategy nor (reliably, see below) a VLM family has an equivalent
// architecture signal.
//
// Dispatch rules (in priority order):
//   metadata.json `model_id` prefix          → factory
//   ─────────────────────────────────────────────────────────────
//   makeVLMPipeline (architecture OR model_id prefix; either alone routes):
//   Qwen2_5_VLForConditionalGeneration | qwen2_5_vl_*   → qwen2_5_vl
//   Qwen3VLForConditionalGeneration    | qwen3_vl_*     → qwen3_vl
//   InternVLChatModel                  | intern3_5_vl_* → intern3_5_vl
//   Gemma4ForConditionalGeneration     | gemma_4_*      → gemma4::makeVLMPipeline
//
//   makeLLMPipeline:
//   llama_v3_*_ssd                             → llama3_2_3b_ssd::makePipeline
//   Gemma4ForConditionalGeneration | gemma_4_* → gemma4::makePipeline
//   (vision bundle)                            → refused; use makeVLMPipeline
//   (dialog.type != "basic")                   → refused; no factory here
//   Phi3ForCausalLM / Qwen2ForCausalLM / LlamaForCausalLM / Qwen3ForCausalLM
//                                               → auto_llm (named for discoverability)
//   anything else                              → auto_llm (unnamed fallback)
//
// The named LLM rows are all the same factory -- naming them just documents
// which families are validated; deleting a row changes nothing. Falcon3 and
// SmolLM2 both report "LlamaForCausalLM" too (confirmed live on HuggingFace),
// so architecture can't split them from Llama-3, which is harmless since all
// three share a factory anyway. The only row with different *behavior* is
// Qwen3ForCausalLM: its model needs a leading BOS its chat template doesn't
// emit, which Tokenizer has no accessor to detect otherwise.
//
// The LLM fallback is guarded, not unconditional: a vision or speculative
// bundle would otherwise get a pipeline that loads and runs while computing
// the wrong thing (wrong RoPE, or a decode loop LLMModel doesn't implement),
// so both are refused based on the bundle's own sidecars rather than model_id,
// catching any future multimodal/speculative family for free.
//
// That guard stays on metadata.json's vision fields, not architecture, even
// though every VLM family GenieX supports has a real composite HF class (see
// the VLM table above) -- for two reasons. First, AIHM's exports don't
// populate it correctly yet (Qwen3-VL/Gemma4 carry no `architectures` key;
// InternVL reports its inner Qwen3 tower's class, not the composite one), and
// bundles already exported before that's fixed won't retroactively gain it.
// Second, even fixed, architecture answers "which VLM family", not this
// guard's question "is this multimodal at all" -- keying the guard on it would
// mean maintaining a denylist of every known VLM class instead of reading the
// bundle's actual vision signal.
//
// Because of that gap, the VLM table's architecture checks are additive with
// model_id, never a replacement: only Qwen2.5-VL's export currently populates
// the field, so the other three routes still run on model_id alone today.

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
#include "pipeline/auto_llm.h"
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
    std::string architecture;  // config.json's architectures[0]; "" if absent/unparsable
};

inline std::optional<BundleFacts> bundleFactsOf(const ModelConfig& model_cfg) {
    try {
        const auto bundle = bundleDirOf(model_cfg);
        const auto meta   = parseQAIRTMetadata(bundle);

        BundleFacts f;
        f.model_id   = meta.model_id;
        f.multimodal = !meta.vision_encoder_graph.empty() || meta.vision_preprocessing.has_value();
        // Both tolerate a missing file, so neither throws on a bundle without one.
        f.dialog_type  = parseGenieConfig(bundle).dialog_type;
        f.architecture = parseModelArchitecture(bundle);
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

// Single LLM entry point. Routes families needing a specialized factory, then
// falls back to the generic one, so a plain decoder-only bundle needs no entry
// here at all.
inline std::optional<LLMPipeline> makeLLMPipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg_in) {
    using namespace dispatch_detail;
    const auto facts = bundleFactsOf(model_cfg_in);
    if (!facts) return std::nullopt;
    const std::string& model_id = facts->model_id;

    // SSD is a decode strategy, not an architecture -- model_id is the only signal.
    if (endsWith(model_id, "_ssd")) {
        const auto cfg = autoDiscoverForecastPrefix(model_cfg_in);
        return llama3_2_3b_ssd::makePipeline(runtime_cfg, cfg);
    }

    if (facts->architecture == "Gemma4ForConditionalGeneration" || startsWith(model_id, "gemma_4_")) {
        return gemma4::makePipeline(runtime_cfg, model_cfg_in);
    }

    // Refused rather than falling through to the generic factory below, which
    // would build a pipeline that loads and runs while computing the wrong
    // thing (see the header comment).
    if (facts->multimodal) {
        GENIEX_LOG_ERROR("dispatch: '{}' is a multimodal bundle; use makeVLMPipeline", model_id);
        return std::nullopt;
    }
    if (facts->dialog_type != "basic") {
        GENIEX_LOG_ERROR(
            "dispatch: '{}' needs dialog.type '{}', which has no LLM factory here", model_id, facts->dialog_type);
        return std::nullopt;
    }

    // Plain decoder-only, fully described by its bundle. Named per architecture
    // purely for discoverability -- every row is the same factory. The only
    // behavioural knob is Qwen3's BOS (see header comment).
    if (facts->architecture == "Phi3ForCausalLM") {  // Phi-3.5, Phi-4
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "Qwen2ForCausalLM") {  // Qwen2.5
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "LlamaForCausalLM") {  // Llama-3, Falcon3, SmolLM2
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "Qwen3ForCausalLM") {
        return auto_llm::makePipeline(runtime_cfg, model_cfg_in, {/*prepend_bos=*/true});
    }

    return auto_llm::makePipeline(runtime_cfg, model_cfg_in);  // unrecognised architecture, no BOS
}

// Single VLM entry point. See the header comment for why the architecture
// checks are additive with model_id rather than a replacement.
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

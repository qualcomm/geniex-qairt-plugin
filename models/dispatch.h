// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Bundle-driven pipeline dispatcher.
//
// Replaces the per-variant llm_model_registry / vlm_model_registry tables. The
// runtime reads facts straight from the bundle -- metadata.json's `model_id`
// for the few things still keyed by it, config.json's `architectures[0]` for
// the model family, genie_config.json's `dialog.type` for the decode strategy
// -- and routes to the matching factory. Adding a new variant of an existing
// family requires no source change: drop the bundle into modelfiles/<name>/
// and call makeLLMPipeline / makeVLMPipeline.
//
// The family signal is deliberately config.json's `architectures[0]`
// (e.g. "Phi3ForCausalLM", "Qwen3ForCausalLM") rather than `model_id`: it is
// the standard HuggingFace field ai-hub-models' exports already carry, so
// ai-hub-models does not have to pick a model_id GenieX happens to
// pattern-match, and a name change on that side cannot silently break
// dispatch here. `model_id` still decides SSD and VLM-family routing below,
// where no equivalent architecture signal exists (see the note at
// makeLLMPipeline's SSD branch and at makeVLMPipeline).
//
// Dispatch rules (in priority order):
//   metadata.json `model_id` prefix          → factory
//   ─────────────────────────────────────────────────────────────
//   makeVLMPipeline:
//   architectures[0] == Qwen2_5_VLForConditionalGeneration → qwen2_5_vl
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
//   everything else                          → auto_model::makePipeline
//
// The LLM table only lists what needs a *specialized* factory; plain
// decoder-only families are the fallback and appear nowhere -- not even Llama,
// Qwen2.5, Qwen3, Falcon3 or Phi. They carry no code of their own either: they
// share auto_model::makePipeline (core/include/pipeline/auto_model.h). Adding
// one is nothing at all: drop the bundle in. The lone remaining knob is BOS,
// which config.json's architecture decides -- see makeLLMPipeline.
//
// The fallback is guarded rather than unconditional. A vision or speculative
// bundle would otherwise build a pipeline that loads and runs while computing
// the wrong thing, so both are detected from the bundle's own sidecars and
// refused with an error — see makeLLMPipeline.
//
// The two tables are independent, and a family may appear in both, as gemma_4_*
// does: one bundle serves text-only and multimodal use, and which entry point to
// call is the caller's choice, not something decided here.
//
// That choice is the reason the multimodal guard exists. Nothing hands a vision
// bundle to makeLLMPipeline deliberately, but nothing stops it either: callers
// have no flag to key off, and a generic "point it at a bundle directory" tool
// cannot tell them apart. `architectures[0]` does not help here either, and not
// uniformly for the same reason across today's five supported VLM families:
//   - Gemma-4-E4B-it, Qwen3-VL-4B/8B-Instruct ship no `architectures` key at all
//     (their config.json's `model_type` is "gemma4_text" / "qwen3_vl_text").
//   - InternVL-3.5-2B's config.json reports its text tower's own class,
//     "Qwen3ForCausalLM" -- byte-identical to a standalone Qwen3 LLM's, with no
//     other field (model_type is plain "qwen3"; no auto_map) hinting it is a VLM
//     tower rather than the whole model. AIHM's export ought to report something
//     that names the composite model instead, e.g. "InternVLChatModel".
//   - Qwen2.5-VL-7B-Instruct is the one exception: its config.json does carry a
//     distinct, correctly composite class, "Qwen2_5_VLForConditionalGeneration".
//     Still not relied on here, both to keep one rule for all five families and
//     because nothing guarantees every future VLM export will follow suit.
// So multimodality is read from metadata.json's vision_preprocessing /
// vision_encoder_graph fields instead, which is what actually differs. Before
// the fallback existed a vision model_id simply matched nothing and errored
// out; the guard keeps that behaviour now that the fallback would otherwise
// take it.
//
// SSD vs plain Llama and the VLM families are decided purely from `model_id`,
// for lack of a config.json signal: a decode strategy or a vision tower is not
// something HuggingFace's `architectures` field encodes, and some exports omit
// the field entirely. Plain vs speculative additionally consults the bundle's
// `dialog.type`. We do not need the bundle's per-graph filename pattern
// (ar*_cl* vs partN_of_M.bin) or `_name_or_path` from config.json.

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
#include "pipeline/auto_model.h"
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
    // config.json's architectures[0] (e.g. "Qwen3ForCausalLM"); "" if
    // config.json is missing, unparsable, or carries no architectures array --
    // some exports (Qwen3-VL, Gemma4) omit it entirely. Those bundles are
    // routed by model_id before this field would matter; see makeLLMPipeline.
    std::string architecture;
};

inline std::optional<BundleFacts> bundleFactsOf(const ModelConfig& model_cfg) {
    try {
        const auto bundle = bundleDirOf(model_cfg);
        const auto meta   = parseQAIRTMetadata(bundle);

        BundleFacts f;
        f.model_id   = meta.model_id;
        f.multimodal = !meta.vision_encoder_graph.empty() || meta.vision_preprocessing.has_value();
        // parseGenieConfig tolerates a missing file, so this cannot throw on
        // bundles that ship no genie_config.json. Same for parseModelArchitecture
        // and a missing config.json.
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
    // SSD: model_id ends in "_ssd". Kept on model_id -- speculative decoding is
    // a decode-loop strategy, not a model architecture, so config.json's
    // architectures[0] carries no equivalent signal (an SSD bundle's tower is
    // architecturally still plain Llama). Auto-populate the forecast prefix path.
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
    // expose; Qwen3's model needs one its chat template does not emit. Keyed
    // off config.json's architecture, not model_id, so ai-hub-models is free to
    // name model_id however it likes -- an empty/unrecognised architecture
    // (config.json missing, or an export that omits the field) defaults to no
    // BOS, matching every other family.
    const bool prepend_bos = (facts->architecture == "Qwen3ForCausalLM");
    return auto_model::makePipeline(runtime_cfg, model_cfg_in, {prepend_bos});
}

// Single VLM entry point. Mostly routes by metadata.json's `model_id` prefix --
// not config.json's architecture, which names the text tower's decoder for most
// of these families (e.g. InternVL's Qwen3 tower reports plain
// "Qwen3ForCausalLM") and so cannot tell them apart, and which some VLM exports
// (Qwen3-VL, Gemma4) omit entirely -- see the note above makeLLMPipeline for the
// full picture across all five families.
//
// Qwen2.5-VL is the one exception: its export's `architectures[0]` names the
// actual composite model, "Qwen2_5_VLForConditionalGeneration" -- a real
// HuggingFace/transformers class, not something AIHM invented, so it is
// checked first and independent of model_id. This is currently redundant with
// the qwen2_5_vl_ prefix below (today's bundles satisfy both), but means
// dispatch keeps working even if AIHM's model_id naming changes.
//
// Add `facts->architecture == "<verified string>" ||` to a family's line below,
// and delete its model_id branch, once its own export actually populates a
// correctly composite class the same way Qwen2.5-VL's does. Do not add a guess:
// none of Qwen3-VL, InternVL or Gemma4 report anything usable today (see the
// note above makeLLMPipeline), and a guessed string that never matches the
// export AIHM actually ships is silent dead code -- it looks like verified
// support without being any. Until a family's export is checked, its model_id
// branch is load-bearing, not just a fallback.
inline std::optional<VLMPipeline> makeVLMPipeline(const QnnRuntimeConfig& runtime_cfg, const VLMConfig& config) {
    using namespace dispatch_detail;
    const auto facts = bundleFactsOf(config.llm_config);
    if (!facts) return std::nullopt;
    const std::string& model_id = facts->model_id;

    if (facts->architecture == "Qwen2_5_VLForConditionalGeneration" || startsWith(model_id, "qwen2_5_vl_")) {
        return qwen2_5_vl::makePipeline(runtime_cfg, config);
    }
    if (startsWith(model_id, "qwen3_vl_")) return qwen3_vl::makePipeline(runtime_cfg, config);
    if (startsWith(model_id, "intern3_5_vl_")) return intern3_5_vl::makePipeline(runtime_cfg, config);
    if (startsWith(model_id, "gemma_4_")) return gemma4::makeVLMPipeline(runtime_cfg, config);

    GENIEX_LOG_ERROR("dispatch: no VLM factory matches model_id '{}'", model_id);
    return std::nullopt;
}

}  // namespace geniex

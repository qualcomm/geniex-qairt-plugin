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
//   makeVLMPipeline (architecture check is `||`-ed with the model_id prefix
//   beside it, so the row still fires from either signal alone):
//   Qwen2_5_VLForConditionalGeneration | qwen2_5_vl_* → qwen2_5_vl
//   Qwen3VLForConditionalGeneration    | qwen3_vl_*   → qwen3_vl
//   InternVLChatModel                  | intern3_5_vl_* → intern3_5_vl
//   Gemma4ForConditionalGeneration     | gemma_4_*    → gemma4::makeVLMPipeline
//
//   makeLLMPipeline:
//   llama_v3_*_ssd                           → llama3_2_3b_ssd::makePipeline
//   Gemma4ForConditionalGeneration | gemma_4_* → gemma4::makePipeline
//   (vision bundle)                          → refused; use makeVLMPipeline
//   (dialog.type != "basic")                 → refused; no factory here
//   architectures[0] == Phi3ForCausalLM      → auto_model  (Phi-3.5, Phi-4)
//   architectures[0] == Qwen2ForCausalLM     → auto_model  (Qwen2.5)
//   architectures[0] == LlamaForCausalLM     → auto_model  (Llama-3, Falcon3, SmolLM2)
//   architectures[0] == Qwen3ForCausalLM     → auto_model  (BOS)
//   anything else                            → auto_model  (no BOS)
//
// The named LLM rows are all the exact same factory -- they exist so a reader
// can see which families are actually validated, not because the routing
// differs. Falcon3 and SmolLM2 both report the same "LlamaForCausalLM" as
// Llama-3 itself (confirmed live against tiiuae/Falcon3-7B-Instruct and
// HuggingFaceTB/SmolLM2-1.7B on HuggingFace), so architecture alone cannot
// split them -- harmless today since they share a factory regardless; only
// model_id could split them if that ever stopped being true. Every string here
// is confirmed against a real bundle or a live HuggingFace config.json, never
// assumed from a model's name -- see the VLM note below for why a guessed one
// would be worse than an anonymous fallback. A new plain family needs no row
// at all: it is served by "anything else" until (optionally) promoted once its
// architecture is confirmed. The lone behavioural knob across all of them is
// BOS, which Tokenizer does not expose -- Qwen3's model needs one its chat
// template does not emit; every other row is otherwise identical.
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
// cannot tell them apart.
//
// The guard is on metadata.json, not `architectures[0]`, even though a real,
// correctly composite HuggingFace class exists for every VLM family GenieX
// supports -- Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration,
// InternVLChatModel, Gemma4ForConditionalGeneration, all confirmed live against
// the families' official repos (see makeVLMPipeline, which does use them,
// alongside model_id). Two separate reasons the guard still doesn't:
//   - Bundle vintage. AIHM's exports don't populate config.json accurately yet
//     for Qwen3-VL / Gemma4 (no `architectures` key at all) or InternVL (reports
//     its inner text tower's plain "Qwen3ForCausalLM", not the composite class
//     above) -- confirmed against real exports of each. AIHM is fixing this, but
//     bundles already exported before the fix ships don't retroactively gain it,
//     so a guard that depends on it would stay unreliable for however long those
//     are still in use.
//   - Even fixed, architecture answers "which VLM family" -- useful for
//     makeVLMPipeline's routing, not for this guard's question, "is this
//     multimodal at all". Keying the guard on architecture would mean
//     maintaining a complete, current denylist of every known VLM class and
//     missing any family not yet added to it. metadata.json's
//     vision_preprocessing / vision_encoder_graph fields answer the actual
//     question directly, for any family, without a list to maintain.
// Before the fallback existed a vision model_id simply matched nothing and
// errored out; the guard keeps that behaviour now that the fallback would
// otherwise take it.
//
// SSD vs plain Llama is decided purely from `model_id`: a decode strategy is
// not something HuggingFace's `architectures` field encodes at all, unlike a
// VLM family. Plain vs speculative additionally consults the bundle's
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

    if (facts->architecture == "Gemma4ForConditionalGeneration" || startsWith(model_id, "gemma_4_")) {
        return gemma4::makePipeline(runtime_cfg, model_cfg_in);
    }

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

    // ── Plain decoder-only, fully described by its bundle ────────────────────
    // Named per architecture purely for discoverability -- every row below is
    // the exact same factory, so there is nothing to keep in sync. Each string
    // is confirmed against a real bundle or a live HuggingFace config.json (see
    // the header comment); an unverified guess would be worse than no row at
    // all -- it would read as confirmed support that was never checked. Add a
    // row once a new family's real architecture string is confirmed; until
    // then it is served correctly, just anonymously, by the fallback.
    if (facts->architecture == "Phi3ForCausalLM") {  // Phi-3.5, Phi-4
        return auto_model::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "Qwen2ForCausalLM") {  // Qwen2.5
        return auto_model::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "LlamaForCausalLM") {  // Llama-3, Falcon3, SmolLM2 -- indistinguishable by
                                                      // architecture; model_id would be the only way to
                                                      // split them if that ever mattered.
        return auto_model::makePipeline(runtime_cfg, model_cfg_in);
    }
    if (facts->architecture == "Qwen3ForCausalLM") {
        // The lone behavioural knob across every row here: Tokenizer exposes no
        // add_bos_token accessor, and Qwen3's model needs a leading BOS its
        // chat template does not emit.
        return auto_model::makePipeline(runtime_cfg, model_cfg_in, {/*prepend_bos=*/true});
    }

    // Unrecognised architecture -- a new family, or config.json missing /
    // lacking the field entirely (some real exports ship neither; see the VLM
    // note below). Still served correctly, just with no name here yet and no
    // BOS.
    return auto_model::makePipeline(runtime_cfg, model_cfg_in);
}

// Single VLM entry point. Routes by config.json's architecture OR metadata.json's
// `model_id` prefix -- either signal is enough on its own, so this keeps working
// however a caller's bundle carries the information:
//
//   Family      | real HF architecture              | model_id prefix
//   Qwen2.5-VL  | Qwen2_5_VLForConditionalGeneration | qwen2_5_vl_
//   Qwen3-VL    | Qwen3VLForConditionalGeneration    | qwen3_vl_
//   InternVL    | InternVLChatModel                  | intern3_5_vl_
//   Gemma4      | Gemma4ForConditionalGeneration     | gemma_4_
//
// All four architecture strings are confirmed live against the families'
// official HuggingFace repos (Qwen/Qwen2.5-VL-7B-Instruct,
// Qwen/Qwen3-VL-4B-Instruct, OpenGVLab/InternVL3_5-2B, google/gemma-4-e4b-it),
// not guessed from a naming convention.
//
// Only Qwen2.5-VL's export currently populates config.json with this value --
// the other three ship no `architectures` key (Qwen3-VL, Gemma4) or the wrong
// one (InternVL reports its inner Qwen3 tower's class, not this composite one)
// -- so today the architecture check is live only for Qwen2.5-VL and the other
// three routes still run on model_id alone. AIHM is updating those exports to
// carry the correct field; once any one does, its row here needs no further
// change -- the `||` already covers it. Nothing here is load-bearing in a way
// that would need reverting either way: architecture is additive next to
// model_id, never a replacement for it, so an unpopulated or wrong field simply
// falls through to model_id exactly as before this row existed.
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

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Device-only EAGLE bring-up: registers the quantized embedding providers and
// drives the target/draft engines' QNN initialize(). Separated from
// eagle_model.cpp so that file's CPU-reachable decode logic can be measured by
// the coverage surface without pulling in QNN backend load. Mirrors the
// model.cpp / model_init.cpp split.

#include <memory>
#include <stdexcept>
#include <string>

#include "llm/eagle_model.h"
#include "llm/input_provider.h"
#include "llm/llm_utils.h"  // isSpecialTensor
#include "logging.h"

namespace geniex {

namespace {

// Index of the engine's body shard -- the one that emits the EAGLE feature
// (last hidden state). For a body+LM-head split that is the second-to-last
// shard; a single-shard engine is its own body.
size_t bodyShardIndex(const LLMSpec& s) { return s.shards.size() >= 2 ? s.shards.size() - 2 : 0; }

// Draft body input that receives the seeding hidden state: the one non-special
// float input that is NOT the embedding entry (shard-0's state input). The
// embedding input, attention mask, and KV inputs are all excluded, leaving the
// feature seed. Throws if it can't be uniquely identified so a mis-inferred
// name can't later no-op silently in decodeBatch's feature override.
std::string inferDraftFeatureInput(SpeculativeLLMModel& draft, const std::string& embed_name) {
    const LLMSpec& s        = draft.spec();
    const size_t   body_idx = bodyShardIndex(s);
    const Graph&   g        = draft.graph(draft.graphIndex(/*phase=*/0, body_idx, /*cl_idx=*/0));

    std::string feature_input;
    for (const auto& t : g.inputSpecs()) {
        if (t.name == embed_name || isSpecialTensor(t.name)) continue;
        if (!feature_input.empty())
            throw std::runtime_error("EagleModel: draft body '" + g.name() +
                                     "' has more than one non-special feature input (" + feature_input + ", " + t.name +
                                     "); cannot infer draft_feature_input");
        feature_input = t.name;
    }
    if (feature_input.empty())
        throw std::runtime_error("EagleModel: draft body '" + g.name() +
                                 "' exposes no feature-seed input besides the embedding entry '" + embed_name + "'");
    return feature_input;
}

}  // namespace

// Fills EagleConfig's graph tensor bindings from the loaded engines, so the
// export-specific names live in the graphs rather than a hard-coded adapter
// table. inferSpecFromGraphs() has already resolved each shard's state I/O and
// the LM-head's logits output; this reads them back and validates the derived
// names against the graphs. Runs after both engines initialize().
void EagleModel::inferTensorBindings(SpeculativeLLMModel& target, SpeculativeLLMModel& draft, EagleConfig& cfg) {
    const LLMSpec& ts = target.spec();
    const LLMSpec& ds = draft.spec();

    cfg.target_embed_name = ts.shards.front().in_state_name;
    cfg.draft_embed_name  = ds.shards.front().in_state_name;

    // Body shard's state output is the hidden feature EAGLE seeds/reads.
    cfg.target_feature_output = ts.shards[bodyShardIndex(ts)].out_state_name;
    cfg.draft_feature_output  = ds.shards[bodyShardIndex(ds)].out_state_name;

    // The draft LM head's state output is its logits (the body's is the feature,
    // so the two differ -- that is why the logits name must be tracked at all).
    cfg.draft_logits_name = ds.shards.back().out_state_name;

    cfg.draft_feature_input = inferDraftFeatureInput(draft, cfg.draft_embed_name);

    // Validate every binding actually resolves to a graph tensor; a silent
    // mismatch would otherwise surface only as lost acceptance at run time.
    const Graph& tgt_body = target.graph(target.graphIndex(0, bodyShardIndex(ts), 0));
    const Graph& drf_body = draft.graph(draft.graphIndex(0, bodyShardIndex(ds), 0));
    const Graph& drf_head = draft.graph(draft.graphIndex(0, ds.shards.size() - 1, 0));
    if (!tgt_body.hasOutput(cfg.target_feature_output) || !drf_body.hasOutput(cfg.draft_feature_output) ||
        !drf_body.hasInput(cfg.draft_feature_input) || !drf_head.hasOutput(cfg.draft_logits_name))
        throw std::runtime_error("EagleModel: inferred a tensor binding absent from the loaded graphs");

    GENIEX_LOG_INFO(
        "EagleModel: inferred tensor bindings (target_embed='{}' draft_embed='{}' target_feat_out='{}' "
        "draft_feat_in='{}' draft_feat_out='{}' draft_logits='{}')",
        cfg.target_embed_name,
        cfg.draft_embed_name,
        cfg.target_feature_output,
        cfg.draft_feature_input,
        cfg.draft_feature_output,
        cfg.draft_logits_name);
}

bool EagleModel::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& target_cfg) {
    // A shared RoPE base is fed raw to both engines' rotary embedding; 0 (the
    // "unset" sentinel) yields inf/nan rotations, so the bundle parser must have
    // supplied it and proven target/draft agree.
    if (!(cfg_.rope_theta > 0.0f))
        throw std::runtime_error(
            "EagleModel: EagleConfig::rope_theta must be > 0 (got " + std::to_string(cfg_.rope_theta) + ")");

    // Bind each engine's quantized embedding to the tensor its graphs expose,
    // resolved on the standard createInputProviders() seam once the spec is
    // inferred -- so the export-specific embedding name is never hard-coded. The
    // draft carries its own embedding table; the target uses the bundle's.
    target().setEmbeddingBinding(cfg_.embedding_quant, /*table_path=*/"");
    if (!target().initialize(runtime_cfg, target_cfg)) {
        GENIEX_LOG_ERROR("EagleModel: target engine initialize() failed");
        return false;
    }

    ModelConfig draft_cfg = target_cfg;
    draft_cfg.model_paths = cfg_.draft_model_paths;
    draft().setEmbeddingBinding(cfg_.embedding_quant, cfg_.draft_embedding_path);
    if (!draft().initialize(runtime_cfg, draft_cfg)) {
        GENIEX_LOG_ERROR("EagleModel: draft engine initialize() failed");
        return false;
    }

    // Both engines' graphs are now loaded and their specs inferred: resolve the
    // remaining feature/logits tensor names from the graphs.
    inferTensorBindings(target(), draft(), cfg_);

    // Multi-CL bundles are safe but unsupported: the speculation loop never
    // promotes, so a long prompt throws ContextLengthExceededError mid-generation
    // rather than growing the context. Nothing rejects this at load, so surface
    // it here to set expectations at the right time instead of on a long prompt.
    if (target().spec().context_lengths.size() > 1 || draft().spec().context_lengths.size() > 1) {
        GENIEX_LOG_WARN(
            "EagleModel: multi-CL bundle (target {} CL, draft {} CL); EAGLE speculation does not promote, so "
            "generation throws once the initial context length is exhausted",
            target().spec().context_lengths.size(),
            draft().spec().context_lengths.size());
    }

    ready_       = true;
    initialized_ = true;
    return true;
}

}  // namespace geniex

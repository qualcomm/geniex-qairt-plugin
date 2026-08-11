// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llm/llm_types.h"  // QuantizedLutSpec

namespace geniex {

// Configuration for EAGLE speculative decoding.
//
// A small draft engine autoregressively proposes candidate tokens conditioned
// on the target engine's last-layer hidden states, and the target verifies them
// in one batched forward. Under greedy sampling the accepted output is identical
// to plain target decoding, so speculation only affects speed.
struct EagleConfig {
    // ── Algorithm policy ──────────────────────────────────────────────────
    // Max draft tokens proposed per speculation round (dialog.eaglet "draft-len").
    // Must be >= 1; the verification batch (1 + draft_len) must fit the target's
    // decode width, validated once both graphs are loaded.
    size_t draft_len = 6;

    // Fan-out of the speculative token tree per level (dialog.eaglet "n-branches").
    // 1 degenerates to a linear chain; higher values raise the acceptance rate at
    // the cost of a wider target verify batch. Clamped so the tree never exceeds
    // max_verify_tokens or the target's decode width.
    size_t n_branches = 1;

    // Upper bound on tree nodes the target verifies in one forward
    // (dialog.eaglet "max-tokens-target-can-evaluate"). Also clamped to the
    // target's decode width at load time.
    size_t max_verify_tokens = 32;

    // ── Assets (required bundle metadata) ─────────────────────────────────
    // Context binaries for the draft engine (target uses ModelConfig::model_paths).
    std::vector<std::string> draft_model_paths;

    // Draft token-embedding table. EAGLE embeds proposed tokens with the draft
    // model's own embedding weights, which differ from the target's even though
    // both cover the full vocabulary. Empty falls back to the target table
    // (ModelConfig::embedding_path), which is only correct when the two engines
    // genuinely share embedding weights.
    std::string draft_embedding_path;

    // Draft logit index -> full-vocab token id. Empty means the draft already
    // emits full-vocabulary ids (identity). When non-empty it must cover at
    // least the draft LM-head vocabulary (one target id per draft logit index)
    // and every entry must index a target token.
    std::vector<int32_t> draft_token_map;

    // Shared RoPE base frequency. EAGLE runs both engines on one value, so the
    // parser must supply it and prove target and draft agree; 0 means unset.
    float rope_theta = 0.0f;

    // Quantized embedding LUT spec (both engines consume embeddings as input).
    QuantizedLutSpec embedding_quant;

    // ── Graph tensor bindings (bundle-specific; required) ─────────────────
    // Each engine's embedding input tensor (its first shard's state input).
    // Kept out of core defaults on purpose: the names are export-specific and
    // belong to the model adapter that parses the bundle.
    std::string target_embed_name;
    std::string draft_embed_name;

    // Target body hidden-state output that seeds the draft.
    std::string target_feature_output;
    // Draft body input that receives the seeding hidden state.
    std::string draft_feature_input;
    // Draft body hidden-state output that conditions the next draft step.
    std::string draft_feature_output;
    // Draft LM-head logits output. The draft's designated shard "state" output
    // is its hidden feature, so the vocabulary logits are a distinct tensor and
    // must be named explicitly (else argmax runs over the hidden state).
    std::string draft_logits_name;
};

// Per-generation speculative-decoding metrics, captured for the most recent
// EagleModel::generate() call.
struct EagleStats {
    // Speculation rounds (target verify passes) executed this generation.
    size_t iterations = 0;
    // Tokens emitted this generation (excludes the prompt), i.e. output.size().
    size_t generated_tokens = 0;

    // Mean tokens accepted per round (generated_tokens / iterations): 1.0 means
    // no speedup, rising toward 1 + draft_len as the draft matches. This is a
    // per-round throughput multiplier, NOT a [0,1] acceptance fraction. Zero
    // when no speculation round ran.
    float meanAcceptedTokensPerRound() const {
        return iterations == 0 ? 0.0f : static_cast<float>(generated_tokens) / static_cast<float>(iterations);
    }
};

}  // namespace geniex

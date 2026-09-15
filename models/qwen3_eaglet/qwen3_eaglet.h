// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm/eagle_model.h"
#include "llm/eagle_types.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "types.h"
#include "utils/detail/json.hpp"

namespace geniex {
namespace qwen3_eaglet {

using json = qualla::json;

// Reads metadata.json's `geniex` block for the eaglet-specific fields the
// two-engine driver needs: the draft engine's paths, the trimmed draft-token
// map, and the shared RoPE base. Mirrors genie_config.json's old `dialog.*`
// schema key-for-key, just nested under `geniex` instead of `dialog`.
inline EagleConfig parseEagletConfig(const std::filesystem::path& bundle_dir, const ParsedGenieConfig& gc) {
    const auto    meta_path = bundle_dir / "metadata.json";
    std::ifstream f(meta_path);
    if (!f) throw std::runtime_error("qwen3_eaglet: cannot open " + meta_path.string());
    json root = json::parse(f);
    if (!root.contains("geniex") || root["geniex"].value("dialog_type", "") != "eaglet") {
        throw std::runtime_error(
            "qwen3_eaglet: metadata.json's geniex.dialog_type is not \"eaglet\" in " + bundle_dir.string());
    }
    const json& gx = root.at("geniex");

    EagleConfig cfg;
    cfg.embedding_quant = gc.embedding_quant;

    if (gx.contains("eaglet")) {
        cfg.draft_len         = gx["eaglet"].value("draft-len", cfg.draft_len);
        cfg.n_branches        = gx["eaglet"].value("n-branches", cfg.n_branches);
        cfg.max_verify_tokens = gx["eaglet"].value("max-tokens-target-can-evaluate", cfg.max_verify_tokens);
    }

    // Resolve the two engines by role: collect the draft's ctx-bins + token map,
    // and read each engine's RoPE base so we can prove they agree (EAGLE drives
    // both on one shared value).
    std::string          draft_token_map;
    std::optional<float> target_theta;
    std::optional<float> draft_theta;
    for (const json& eng : gx.at("engine")) {
        const std::string role = eng.value("role", "");
        const json*       pe   = (eng.contains("model") && eng["model"].contains("positional-encoding"))
                                     ? &eng["model"]["positional-encoding"]
                                     : nullptr;
        if (role == "target") {
            if (pe && pe->contains("rope-theta")) target_theta = pe->at("rope-theta").get<float>();
        } else if (role == "draft") {
            if (eng.contains("model") && eng["model"].contains("binary")) {
                for (const auto& b : eng["model"]["binary"].at("ctx-bins"))
                    cfg.draft_model_paths.push_back((bundle_dir / b.get<std::string>()).string());
            }
            if (eng.contains("model")) draft_token_map = eng["model"].value("draft-token-map", "");
            if (pe && pe->contains("rope-theta")) draft_theta = pe->at("rope-theta").get<float>();
        }
    }
    if (cfg.draft_model_paths.empty()) {
        throw std::runtime_error(
            "qwen3_eaglet: no draft engine (role=draft) with ctx-bins in metadata.json's geniex.engine");
    }

    // The draft embeds proposed tokens with its own weights. The metadata only
    // declares the shared (target) embedding table, so the draft table is
    // resolved by the export convention "draft_" + the declared lut filename.
    {
        const std::string target_lut =
            gx.value("embedding", json::object()).value("lut-path", "quantized_embedding_table.bin");
        const auto draft_lut = bundle_dir / ("draft_" + target_lut);
        if (std::filesystem::exists(draft_lut)) cfg.draft_embedding_path = draft_lut.string();
    }
    if (!target_theta || !draft_theta) {
        throw std::runtime_error(
            "qwen3_eaglet: both target and draft engines must declare positional-encoding rope-theta");
    }
    if (*target_theta != *draft_theta) {
        throw std::runtime_error("qwen3_eaglet: target/draft rope-theta mismatch (" + std::to_string(*target_theta) +
                                 " vs " + std::to_string(*draft_theta) + "); EAGLE requires a shared value");
    }
    cfg.rope_theta = *draft_theta;

    // Load the trimmed draft-vocab → full-vocab id map (ordered JSON array/object of ints).
    if (!draft_token_map.empty()) {
        std::ifstream tf(bundle_dir / draft_token_map);
        if (!tf) throw std::runtime_error("qwen3_eaglet: cannot open draft-token-map " + draft_token_map);
        json tm = json::parse(tf);

        // Declared vocab sizes bound the map: a key indexes the draft vocab, a
        // value indexes the full target vocab. Validating here (rather than on
        // every proposal) turns a malformed bundle into one clear load-time error
        // and stops an out-of-range key from driving a multi-GB allocation or an
        // out-of-range value from reaching the embedding lookup. Absent counts
        // (0) disable the corresponding bound rather than reject the bundle.
        const json&  ctx          = gx.contains("context") ? gx.at("context") : gx;
        const size_t draft_nvocab = ctx.value("draft-n-vocab", 0);
        const size_t full_nvocab  = ctx.value("n-vocab", 0);
        auto         check_value  = [&](int32_t value) {
            if (value < 0 || (full_nvocab && static_cast<size_t>(value) >= full_nvocab))
                throw std::runtime_error("qwen3_eaglet: draft-token-map value " + std::to_string(value) +
                                         " is out of range for the target vocab (" + std::to_string(full_nvocab) +
                                         ") in " + draft_token_map);
            return value;
        };

        if (tm.is_array()) {
            cfg.draft_token_map.reserve(tm.size());
            for (const auto& v : tm) cfg.draft_token_map.push_back(check_value(v.get<int32_t>()));
        } else {
            // Object keyed by decimal draft index. nlohmann stores object keys
            // sorted lexicographically ("0","1","10",...), so iteration order does
            // NOT match numeric index -- place each value at its parsed key or the
            // draft-token map ends up scrambled and every proposal is wrong.
            // Parse keys ourselves: a non-numeric key or a negative/oversized index
            // is a malformed bundle, reported as a named error rather than a bare
            // std::stoi throw or an absurd allocation.
            auto parse_key = [&](const std::string& key) -> int32_t {
                size_t consumed = 0;
                long   parsed   = 0;
                try {
                    parsed = std::stol(key, &consumed);
                } catch (const std::exception&) {
                    consumed = 0;  // fall through to the shared diagnostic
                }
                if (consumed != key.size() || parsed < 0 ||
                    (draft_nvocab && static_cast<size_t>(parsed) >= draft_nvocab))
                    throw std::runtime_error(
                        "qwen3_eaglet: draft-token-map has a non-numeric, negative, or "
                        "out-of-range key '" +
                        key + "' (draft vocab " + std::to_string(draft_nvocab) + ") in " + draft_token_map);
                return static_cast<int32_t>(parsed);
            };
            int32_t max_key = -1;
            for (auto it = tm.begin(); it != tm.end(); ++it) max_key = std::max(max_key, parse_key(it.key()));
            cfg.draft_token_map.assign(static_cast<size_t>(max_key + 1), 0);
            for (auto it = tm.begin(); it != tm.end(); ++it)
                cfg.draft_token_map[static_cast<size_t>(parse_key(it.key()))] = check_value(it.value().get<int32_t>());
        }
    }

    return cfg;
}

// Builds and fully initializes both engines. The returned model is ready for
// generate(): the target is initialized via ModelConfig::model_paths and the
// draft via the paths parsed from metadata.json's geniex.engine.
inline std::unique_ptr<EagleModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       meta   = parseQAIRTMetadata(bundle);
    if (meta.dialog_type != "eaglet") {
        throw std::runtime_error("qwen3_eaglet::makeModel requires metadata.json's geniex.dialog_type == \"eaglet\"");
    }
    auto gc = runtimeConfigFromMetadata(meta);

    EagleConfig ecfg        = parseEagletConfig(bundle, gc);
    LLMSpec     target_spec = buildSpecSkeleton(gc);
    LLMSpec     draft_spec  = buildSpecSkeleton(gc);

    GENIEX_LOG_INFO(
        "qwen3_eaglet: config parsed (draft_len={} n_branches={} max_verify={} rope_theta={} draft_paths={} "
        "token_map={})",
        ecfg.draft_len,
        ecfg.n_branches,
        ecfg.max_verify_tokens,
        ecfg.rope_theta,
        ecfg.draft_model_paths.size(),
        ecfg.draft_token_map.size());

    auto model = std::make_unique<EagleModel>(std::move(target_spec), std::move(draft_spec), ecfg);
    GENIEX_LOG_INFO("qwen3_eaglet: initializing target + draft engines...");
    if (!model->initialize(runtime_cfg, model_cfg)) {
        throw std::runtime_error("qwen3_eaglet: engine initialize() failed");
    }
    GENIEX_LOG_INFO("qwen3_eaglet: engines ready.");
    return model;
}

}  // namespace qwen3_eaglet
}  // namespace geniex

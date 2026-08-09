// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/llm_spec_loader.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>

#include "llm/llm_utils.h"  // isKVTensor / isSpecialTensor
#include "logging.h"
#include "utils/detail/json.hpp"

namespace geniex {
namespace {

using json = qualla::json;

// Reads file into a json document; throws on missing file or parse error.
json loadJson(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("llm_spec_loader: cannot open " + path.string());
    }
    try {
        return json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error("llm_spec_loader: failed to parse " + path.string() + ": " + e.what());
    }
}

// Captures (ar, cl, shard, total) and an optional phase prefix from an LLM-
// style graph name. Accepts forms with or without a phase prefix.
struct GraphNameParts {
    std::string phase_prefix;
    size_t      ar    = 0;
    size_t      cl    = 0;
    size_t      shard = 0;
    size_t      total = 0;
};

bool parseGraphName(const std::string& name, GraphNameParts& out) {
    static const std::regex re(R"((?:([a-zA-Z]+)_)?ar(\d+)_cl(\d+)_(\d+)_of_(\d+))");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return false;
    out.phase_prefix = m[1].matched ? m[1].str() : "";
    out.ar           = std::stoul(m[2].str());
    out.cl           = std::stoul(m[3].str());
    out.shard        = std::stoul(m[4].str());
    out.total        = std::stoul(m[5].str());
    return true;
}

// Captures (shard, total) from `partN_of_M.bin`-style VLM keys.
bool parsePartShardName(const std::string& name, size_t& shard, size_t& total) {
    static const std::regex re(R"((?:.*_)?part(\d+)_of_(\d+)\.bin)");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return false;
    shard = std::stoul(m[1].str());
    total = std::stoul(m[2].str());
    return true;
}

// Parses the integer N from "past_key_<N>_in" or "past_value_<N>_in".
std::optional<size_t> parsePastIndex(const std::string& name) {
    static const std::regex re(R"(past_(?:key|value)_(\d+)_(?:in|out))");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    return std::stoul(m[1].str());
}

template <typename T>
std::optional<T> getOpt(const json& j, const std::string& key) {
    if (!j.contains(key) || j.at(key).is_null()) return std::nullopt;
    return j.at(key).get<T>();
}

// Reads the shape of one tensor entry inside metadata.json's inputs/outputs
// map. Returns an empty vector if the entry is missing or malformed.
std::vector<size_t> readShape(const json& tensor_entry) {
    std::vector<size_t> shape;
    if (!tensor_entry.is_object() || !tensor_entry.contains("shape") || !tensor_entry.at("shape").is_array()) {
        return shape;
    }
    for (const auto& d : tensor_entry.at("shape")) {
        if (d.is_number_integer() || d.is_number_unsigned()) shape.push_back(d.get<size_t>());
    }
    return shape;
}

// Per-shard hyperparameter signal from one metadata.json graph entry.
struct ShardWiring {
    std::set<size_t>    kv_layer_indices;
    std::vector<size_t> in_state_shape;    // first non-special input shape (for hidden_size)
    std::vector<size_t> out_state_shape;   // first non-special output shape (fallback)
    std::vector<size_t> past_key_shape;    // first past_key_* shape (for num_kv_heads / head_dim)
    std::vector<size_t> logits_shape;      // for vocab_size
    std::string         first_input_name;  // raw JSON key of the first non-special input
};

ShardWiring readShardWiring(const json& graph_entry, const std::string& /*diag_label*/) {
    ShardWiring w;
    bool        in_state_seen = false, out_state_seen = false;
    if (graph_entry.contains("inputs") && graph_entry.at("inputs").is_object()) {
        for (auto it = graph_entry.at("inputs").begin(); it != graph_entry.at("inputs").end(); ++it) {
            const std::string& key = it.key();
            if (isSpecialTensor(key)) {
                if (auto idx = parsePastIndex(key)) w.kv_layer_indices.insert(*idx);
                if (key.rfind("past_key_", 0) == 0 && w.past_key_shape.empty()) {
                    w.past_key_shape = readShape(it.value());
                }
                continue;
            }
            if (!in_state_seen) {
                w.in_state_shape   = readShape(it.value());
                w.first_input_name = key;
                in_state_seen      = true;
            }
        }
    }
    if (graph_entry.contains("outputs") && graph_entry.at("outputs").is_object()) {
        for (auto it = graph_entry.at("outputs").begin(); it != graph_entry.at("outputs").end(); ++it) {
            const std::string& key = it.key();
            if (key == "logits") w.logits_shape = readShape(it.value());
            if (isSpecialTensor(key)) {
                if (key.rfind("past_key_", 0) == 0 && w.past_key_shape.empty()) {
                    w.past_key_shape = readShape(it.value());
                }
                continue;
            }
            if (!out_state_seen) {
                w.out_state_shape = readShape(it.value());
                out_state_seen    = true;
            }
        }
    }
    return w;
}

void parseVisionPreprocessing(const json& j, ParsedVisionPreprocessing& out) {
    out.image_width         = j.value("image_width", 0);
    out.image_height        = j.value("image_height", 0);
    out.patch_size          = j.value("patch_size", 0);
    out.temporal_patch_size = j.value("temporal_patch_size", 0);
    out.spatial_merge_size  = j.value("spatial_merge_size", 0);
    if (j.contains("normalize_mean") && j.at("normalize_mean").is_array()) {
        for (const auto& v : j.at("normalize_mean")) out.normalize_mean.push_back(v.get<float>());
    }
    if (j.contains("normalize_std") && j.at("normalize_std").is_array()) {
        for (const auto& v : j.at("normalize_std")) out.normalize_std.push_back(v.get<float>());
    }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// parseQAIRTMetadata
// ─────────────────────────────────────────────────────────────────────────────
ParsedQAIRTMetadata parseQAIRTMetadata(const std::filesystem::path& bundle_dir) {
    auto path = bundle_dir / "metadata.json";
    auto j    = loadJson(path);

    if (!j.contains("model_files") || !j.at("model_files").is_object()) {
        throw std::runtime_error("llm_spec_loader: metadata.json missing 'model_files' object");
    }
    const auto& model_files = j.at("model_files");

    ParsedQAIRTMetadata out;
    out.model_id = j.value("model_id", std::string{});

    {
        const json* vp_obj = nullptr;
        if (j.contains("vision_preprocessing") && j.at("vision_preprocessing").is_object()) {
            vp_obj = &j.at("vision_preprocessing");
        } else if (j.contains("genie") && j.at("genie").is_object() && j.at("genie").contains("vision_preprocessing") &&
                   j.at("genie").at("vision_preprocessing").is_object()) {
            vp_obj = &j.at("genie").at("vision_preprocessing");
        }
        if (vp_obj) {
            ParsedVisionPreprocessing vp;
            parseVisionPreprocessing(*vp_obj, vp);
            out.vision_preprocessing = vp;
        }
    }

    size_t      total_shards = 0;
    std::string vision_encoder_key;

    // Per-shard representative graph entry. Any AR/CL variant of a given
    // shard exposes the same num_kv_heads / head_dim / hidden_size /
    // kv_layer_indices, so the first match wins.
    std::map<size_t, const json*> per_shard_entry;

    for (auto it = model_files.begin(); it != model_files.end(); ++it) {
        const std::string& key = it.key();
        GraphNameParts     parts;
        if (parseGraphName(key, parts)) {
            total_shards = std::max(total_shards, parts.total);
            per_shard_entry.try_emplace(parts.shard, &it.value());
            continue;
        }
        size_t shard = 0, total = 0;
        if (parsePartShardName(key, shard, total)) {
            total_shards           = std::max(total_shards, total);
            per_shard_entry[shard] = &it.value();
            continue;
        }
        if (key == "vision_encoder.bin") {
            vision_encoder_key = key;
            continue;
        }
        GENIEX_LOG_WARN("llm_spec_loader: ignoring unrecognised graph entry '{}'", key);
    }

    if (total_shards == 0) {
        throw std::runtime_error("llm_spec_loader: metadata.json contains no recognisable shard entries");
    }
    out.vision_encoder_graph = vision_encoder_key;

    out.shards.resize(total_shards);

    size_t max_past_key_idx   = 0;
    auto   absorb_hyperparams = [&](const ShardWiring& w) {
        if (out.hidden_size == 0 && w.in_state_shape.size() >= 3) {
            out.hidden_size = w.in_state_shape[w.in_state_shape.size() - 1];
        }
        if (out.hidden_size == 0 && w.out_state_shape.size() >= 3) {
            out.hidden_size = w.out_state_shape[w.out_state_shape.size() - 1];
        }
        if (!w.past_key_shape.empty()) {
            if (out.num_kv_heads == 0) out.num_kv_heads = w.past_key_shape[0];
            if (out.head_dim == 0 && w.past_key_shape.size() >= 3) out.head_dim = w.past_key_shape[2];
        }
        if (out.vocab_size == 0 && !w.logits_shape.empty()) {
            out.vocab_size = w.logits_shape.back();
        }
    };

    for (size_t s = 1; s <= total_shards; ++s) {
        auto it = per_shard_entry.find(s);
        if (it == per_shard_entry.end() || it->second == nullptr) {
            throw std::runtime_error("llm_spec_loader: could not locate graph entry for shard " + std::to_string(s));
        }
        auto w            = readShardWiring(*it->second, "shard " + std::to_string(s));
        out.shards[s - 1] = ShardSpec{};

        // Record shard 0's first-input name (raw JSON key) so the embedding
        // provider factory can decide between input_ids and inputs_embeds.
        if (s == 1) out.first_shard_input_hint = w.first_input_name;

        for (size_t idx : w.kv_layer_indices) max_past_key_idx = std::max(max_past_key_idx, idx);

        absorb_hyperparams(w);
    }

    // num_hidden_layers = highest past_key_<N> index seen + 1.
    if (max_past_key_idx > 0 || out.num_kv_heads > 0) {
        out.num_hidden_layers = max_past_key_idx + 1;
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// parseGenieConfig
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// Maps the rope-type strings to our RopeScaling variant.
RopeScaling parseRopeScaling(const json& rs) {
    if (!rs.is_object()) return StandardRope{};
    const std::string type = rs.value("rope-type", rs.value("type", std::string{"default"}));

    if (type == "llama3") {
        Llama3RopeScaling s;
        s.factor                           = rs.value("factor", 1.0f);
        s.low_freq_factor                  = rs.value("low-freq-factor", 1.0f);
        s.high_freq_factor                 = rs.value("high-freq-factor", 4.0f);
        s.original_max_position_embeddings = rs.value("original-max-position-embeddings", size_t{8192});
        return s;
    }
    if (type == "longrope") {
        LongRopeScaling s;
        if (rs.contains("long-factor"))
            for (const auto& v : rs.at("long-factor")) s.long_factor.push_back(v.get<float>());
        if (rs.contains("short-factor"))
            for (const auto& v : rs.at("short-factor")) s.short_factor.push_back(v.get<float>());
        s.original_max_position_embeddings = rs.value("original-max-position-embeddings", size_t{4096});
        return s;
    }
    if (type == "qwen2vl-mrope" || type == "qwen3vl-mrope") {
        MRopeScaling s;
        s.spatial_merge_size = rs.value("spatial-merge-size", 2);
        s.time_step          = rs.value("time-step", 50);
        if (rs.contains("mrope-section") && rs.at("mrope-section").is_array()) {
            for (const auto& v : rs.at("mrope-section")) s.mrope_section.push_back(v.get<int>());
        } else {
            s.mrope_section = {16, 24, 24};  // qwen2-vl default
        }
        return s;
    }
    if (type == "partial" || type == "proportional") {
        // "proportional" is Gemma3/4's name for partial-rotary RoPE: only the
        // front `partial-rotary-factor` of head dims are rotated. Its `factor`
        // is the post-scale (usually 1.0).
        PartialRopeScaling s;
        s.rope_fraction = rs.value("rope-fraction", rs.value("partial-rotary-factor", 1.0f));
        s.scale         = rs.value("scale", rs.value("factor", 1.0f));
        return s;
    }
    return StandardRope{};
}

// Reads an embedding block's `datatype` + `quant-param` (Genie's schema for a
// quantized LUT). Absent datatype, or "float32", leaves the spec unquantized.
QuantizedLutSpec parseLutQuant(const json& block) {
    QuantizedLutSpec q;
    if (auto v = getOpt<std::string>(block, "datatype")) q.datatype = *v;
    if (block.contains("quant-param") && block.at("quant-param").is_object()) {
        const auto& qp = block.at("quant-param");
        q.scale        = qp.value("scale", 1.0f);
        q.offset       = qp.value("offset", 0);
    }
    return q;
}

}  // namespace

ParsedGenieConfig parseGenieConfig(const std::filesystem::path& bundle_dir) {
    ParsedGenieConfig out;
    auto              path = bundle_dir / "genie_config.json";
    if (!std::filesystem::exists(path)) return out;
    try {
        auto j = loadJson(path);
        if (!j.contains("dialog") || !j.at("dialog").is_object()) return out;
        const auto& dialog = j.at("dialog");

        if (dialog.contains("type") && dialog.at("type").is_string()) {
            out.dialog_type = dialog.at("type").get<std::string>();
        }

        // dialog.context.{n-vocab is informational, bos-token, eos-token}.
        if (dialog.contains("context") && dialog.at("context").is_object()) {
            const auto& ctx = dialog.at("context");
            if (auto v = getOpt<int32_t>(ctx, "bos-token")) out.bos_token_id = *v;
            if (auto v = getOpt<int32_t>(ctx, "pad-token")) out.pad_token_id = *v;
            if (ctx.contains("eos-token") && !ctx.at("eos-token").is_null()) {
                const auto& eos = ctx.at("eos-token");
                if (eos.is_number_integer()) {
                    out.eos_token_ids.push_back(eos.get<int32_t>());
                } else if (eos.is_array()) {
                    for (const auto& e : eos) out.eos_token_ids.push_back(e.get<int32_t>());
                }
            }
        }

        // RoPE: prefer dialog.engine.model.positional-encoding (full schema);
        // fall back to dialog.engine.backend.QnnHtp.rope-theta for older
        // bundles that only carry the base.
        if (dialog.contains("engine") && dialog.at("engine").is_object()) {
            const auto& engine = dialog.at("engine");

            if (engine.contains("model") && engine.at("model").is_object() &&
                engine.at("model").contains("positional-encoding") &&
                engine.at("model").at("positional-encoding").is_object()) {
                const auto& pe = engine.at("model").at("positional-encoding");
                out.rope_theta = pe.value("rope-theta", 10000.0f);
                if (pe.contains("rope-scaling") && pe.at("rope-scaling").is_object()) {
                    out.rope_scaling = parseRopeScaling(pe.at("rope-scaling"));
                }
            } else if (engine.contains("backend") && engine.at("backend").is_object() &&
                       engine.at("backend").contains("QnnHtp") && engine.at("backend").at("QnnHtp").is_object()) {
                const auto& htp = engine.at("backend").at("QnnHtp");
                if (htp.contains("rope-theta")) out.rope_theta = htp.at("rope-theta").get<float>();
            }

            // Gemma3/4 local (sliding-window) RoPE: dialog.engine.model.
            // local-positional-encoding. Separate theta + (optional) scaling.
            if (engine.contains("model") && engine.at("model").is_object() &&
                engine.at("model").contains("local-positional-encoding") &&
                engine.at("model").at("local-positional-encoding").is_object()) {
                const auto& lpe                       = engine.at("model").at("local-positional-encoding");
                out.local_positional_encoding_present = true;
                out.local_rope_theta                  = lpe.value("rope-theta", 10000.0f);
                if (lpe.contains("rope-scaling") && lpe.at("rope-scaling").is_object()) {
                    out.local_rope_scaling = parseRopeScaling(lpe.at("rope-scaling"));
                }
            }
        }

        // dialog.embedding.lut-path — VLM/external-embedding bundles.
        if (dialog.contains("embedding") && dialog.at("embedding").is_object()) {
            const auto& emb = dialog.at("embedding");
            if (auto v = getOpt<std::string>(emb, "lut-path")) out.embedding_lut_path = *v;
            out.embedding_quant = parseLutQuant(emb);
        }

        // dialog.perlayer-embedding — Gemma3/4 per-layer embedding stream.
        if (dialog.contains("perlayer-embedding") && dialog.at("perlayer-embedding").is_object()) {
            const auto& ple = dialog.at("perlayer-embedding");
            if (auto v = getOpt<std::string>(ple, "lut-path")) out.perlayer_embedding_lut_path = *v;
            out.perlayer_embedding_size  = ple.value("size", size_t{0});
            out.perlayer_embedding_quant = parseLutQuant(ple);
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("llm_spec_loader: failed to parse genie_config.json: {}", e.what());
    }
    return out;
}

ParsedSamplerConfig parseGenieSamplerConfig(const std::filesystem::path& bundle_dir) {
    ParsedSamplerConfig out;
    auto                path = bundle_dir / "genie_config.json";
    if (!std::filesystem::exists(path)) return out;
    try {
        auto j = loadJson(path);
        if (!j.contains("dialog") || !j.at("dialog").is_object()) return out;
        const auto& dialog = j.at("dialog");
        if (!dialog.contains("sampler") || !dialog.at("sampler").is_object()) return out;
        const auto& s = dialog.at("sampler");

        if (auto v = getOpt<uint32_t>(s, "seed")) out.seed = *v;
        if (auto v = getOpt<float>(s, "temp")) out.temperature = *v;
        if (auto v = getOpt<int32_t>(s, "top-k")) out.top_k = *v;
        if (auto v = getOpt<float>(s, "top-p")) out.top_p = *v;

        if (s.contains("token-penalty") && s.at("token-penalty").is_object()) {
            const auto& tp = s.at("token-penalty");
            if (auto v = getOpt<float>(tp, "repetition-penalty")) out.repetition_penalty = *v;
            if (auto v = getOpt<float>(tp, "presence-penalty")) out.presence_penalty = *v;
            if (auto v = getOpt<float>(tp, "frequency-penalty")) out.frequency_penalty = *v;
            if (auto v = getOpt<int32_t>(tp, "penalize-last-n")) out.penalty_last_n = *v;
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("llm_spec_loader: failed to parse dialog.sampler: {}", e.what());
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildSpecSkeleton
// ─────────────────────────────────────────────────────────────────────────────
LLMSpec buildSpecSkeleton(const ParsedGenieConfig& gc) {
    LLMSpec spec;
    spec.state_blocks  = {makeKVStateBlock()};
    spec.eos_token_ids = gc.eos_token_ids;
    spec.bos_token_id  = gc.bos_token_id;
    return spec;
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider factories
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<InputProvider> makeRoPEProvider(
    size_t head_dim, const ParsedGenieConfig& gc, std::string cos_name, std::string sin_name) {
    return std::visit(
        [&](const auto& s) -> std::unique_ptr<InputProvider> {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, Llama3RopeScaling>) {
                // llama3 frequency scaling materially changes the RoPE table
                // (low-freq dims divided by factor, mid band interpolated). The
                // compiled graph consumes position_ids_cos/sin as live inputs,
                // so we must reproduce Genie's scaled table to match its logits.
                GENIEX_LOG_INFO(
                    "llm_spec_loader: rope_scaling=llama3 (factor={}); using Llama3 RoPE provider", s.factor);
                return std::make_unique<Llama3RoPEInputProvider>(head_dim,
                    gc.rope_theta,
                    s.factor,
                    s.low_freq_factor,
                    s.high_freq_factor,
                    static_cast<int>(s.original_max_position_embeddings),
                    cos_name,
                    sin_name);
            } else if constexpr (std::is_same_v<T, LongRopeScaling>) {
                const size_t orig = s.original_max_position_embeddings ? s.original_max_position_embeddings : 4096;
                return std::make_unique<LongRoPEInputProvider>(head_dim,
                    gc.rope_theta,
                    s.long_factor,
                    /*max_position_embeddings=*/131072,
                    static_cast<int>(orig),
                    cos_name,
                    sin_name);
            } else if constexpr (std::is_same_v<T, PartialRopeScaling>) {
                // Gemma3/4's global RoPE ships two on-disk layouts. The classic
                // export names the pair position_ids_cos/sin and stores the
                // compact rope_dim/2-wide table; the newer export (QAIRT 2.45+)
                // names it position_ids_global_cos/sin and stores the full
                // head_dim/2-wide zero-padded rotate_half table. Detect by name.
                const bool full_width = cos_name.find("_global") != std::string::npos;
                return std::make_unique<PartialRoPEInputProvider>(
                    head_dim, gc.rope_theta, s.rope_fraction, s.scale, cos_name, sin_name, full_width);
            } else if constexpr (std::is_same_v<T, MRopeScaling>) {
                // Caller (VLM family) wires a dedicated MRoPEInputProvider with
                // the full mrope_section; for the LLM dispatch this branch is
                // unreachable. Falling back here keeps the function total.
                GENIEX_LOG_INFO("llm_spec_loader: rope_scaling=mrope (mrope_section={}); using standard RoPE provider",
                    s.mrope_section.size());
                return std::make_unique<RoPEInputProvider>(head_dim, gc.rope_theta, cos_name, sin_name);
            } else {
                return std::make_unique<RoPEInputProvider>(head_dim, gc.rope_theta, cos_name, sin_name);
            }
        },
        gc.rope_scaling);
}

std::unique_ptr<InputProvider> makeEmbeddingProvider(
    const std::string& first_shard_input, const ParsedGenieConfig& gc) {
    if (first_shard_input == "input_ids") {
        int32_t pad = gc.pad_token_id;
        if (pad < 0) pad = gc.eos_token_ids.empty() ? 0 : gc.eos_token_ids.front();
        return std::make_unique<TokenIdInputProvider>("input_ids", pad);
    }
    if (first_shard_input == "input_embeds" || first_shard_input == "inputs_embeds") {
        auto p = std::make_unique<EmbeddingInputProvider>(first_shard_input);
        // Quantized main LUT (Gemma4 and other large-vocab bundles): mmap +
        // per-row conversion instead of a dequantized in-RAM table.
        if (gc.embedding_quant.quantized()) p->setQuantization(gc.embedding_quant);
        return p;
    }
    throw std::runtime_error("llm_spec_loader: unrecognised first-shard input '" + first_shard_input +
                             "' — expected 'input_ids', 'input_embeds', or 'inputs_embeds'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Bundle helpers
// ─────────────────────────────────────────────────────────────────────────────
std::filesystem::path bundleDirOf(const ModelConfig& model_cfg) {
    if (model_cfg.model_paths.empty()) {
        throw std::runtime_error("llm_spec_loader: model_cfg.model_paths is empty");
    }
    return std::filesystem::path(model_cfg.model_paths.front()).parent_path();
}

ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir) {
    ModelConfig cfg;

    auto tok = bundle_dir / "tokenizer.json";
    if (!std::filesystem::exists(tok)) {
        throw std::runtime_error("llm_spec_loader: tokenizer.json not found in " + bundle_dir.string());
    }
    cfg.tokenizer_path = tok.string();

    auto htp = bundle_dir / "htp_backend_ext_config.json";
    if (std::filesystem::exists(htp)) {
        cfg.htp_config_path = htp.string();
        cfg.num_cores       = parseHtpCoreCount(htp);
    }

    auto genie_path = bundle_dir / "genie_config.json";
    if (std::filesystem::exists(genie_path)) {
        try {
            auto gj = loadJson(genie_path);
            if (gj.contains("dialog") && gj.at("dialog").contains("engine") &&
                gj.at("dialog").at("engine").contains("model") &&
                gj.at("dialog").at("engine").at("model").contains("binary") &&
                gj.at("dialog").at("engine").at("model").at("binary").contains("ctx-bins")) {
                for (const auto& b : gj.at("dialog").at("engine").at("model").at("binary").at("ctx-bins")) {
                    cfg.model_paths.push_back((bundle_dir / b.get<std::string>()).string());
                }
            }
        } catch (const std::exception& e) {
            GENIEX_LOG_WARN("llm_spec_loader: failed to read genie_config.json: {}", e.what());
        }
    }

    if (cfg.model_paths.empty()) {
        std::vector<std::string> bins;
        for (const auto& entry : std::filesystem::directory_iterator(bundle_dir)) {
            if (entry.path().extension() == ".bin") bins.push_back(entry.path().string());
        }
        std::sort(bins.begin(), bins.end());
        cfg.model_paths = std::move(bins);
    }

    if (cfg.model_paths.empty()) {
        throw std::runtime_error("llm_spec_loader: no .bin shards found in " + bundle_dir.string());
    }
    return cfg;
}

uint32_t parseHtpCoreCount(const std::filesystem::path& htp_config_path) {
    if (!std::filesystem::exists(htp_config_path)) return 0;

    json j;
    try {
        j = loadJson(htp_config_path);
    } catch (const std::exception& e) {
        // Malformed JSON is not fatal here — QnnHtpNetRunExtensions will surface
        // its own error when it parses the same file at backend init.
        GENIEX_LOG_WARN("llm_spec_loader: could not parse {}: {}", htp_config_path.string(), e.what());
        return 0;
    }

    if (!j.contains("devices") || !j.at("devices").is_array()) return 0;

    uint32_t max_cores = 0;
    for (const auto& device : j.at("devices")) {
        if (!device.contains("cores") || !device.at("cores").is_array()) continue;
        max_cores = std::max<uint32_t>(max_cores, static_cast<uint32_t>(device.at("cores").size()));
    }
    return max_cores;
}

}  // namespace geniex

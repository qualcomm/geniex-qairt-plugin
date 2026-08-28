// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "PerfProfile.hpp"  // for geniex::PerfProfile
#include "QnnLog.h"
#include "QnnTypes.h"
#include "geniex-proc/tokenizer.h"  // for Tokenizer
#include "geniex-proc/types.h"      // for GENIEX_DEFAULT_SEED

namespace geniex {

// QNN backend settings shared across all models.
//
// The three path fields are optional. Leave them as std::nullopt (the default)
// to have geniex_core auto-detect the correct HTP runtime folder based on the
// device's HTP architecture version (see runtime_resolver.h). Set them
// explicitly to override the auto-detected paths.
struct QnnRuntimeConfig {
    // Path to QnnHtp.dll / libQnnHtp.so.
    // std::nullopt = auto-detect from htp-files/ next to geniex_core.
    std::optional<std::string> backend_path;

    // Path to QnnSystem.dll / libQnnSystem.so.
    // std::nullopt = auto-detect (same folder as backend_path).
    std::optional<std::string> system_lib_path;

    // Path to QnnHtpNetRunExtensions.dll / libQnnHtpNetRunExtensions.so.
    // std::nullopt = auto-detect (same folder as backend_path).
    std::optional<std::string> extensions_path;

    QnnLog_Level_t log_level = QNN_LOG_LEVEL_ERROR;
    bool           debug     = false;
};

// Per-model configuration: everything needed to load and run a QNN graph model.
struct ModelConfig {
    std::vector<std::string>   model_paths;      // .bin shards in order (required)
    std::string                tokenizer_path;   // tokenizer.json on disk (required)
    std::string                htp_config_path;  // HTP JSON config (empty = QNN defaults)
    std::optional<std::string> embedding_path;   // CPU-side embedding table; nullopt if embeddings live in the graph
    // tokenizer_config.json (chat template). nullopt = discover next to model_paths[0].
    std::optional<std::string> tokenizer_config_path;
    // Forecast-prefix KV-cache file used by SSD variants. nullopt for non-SSD models.
    std::optional<std::string> forecast_prefix_path;
    PerfProfile                perf_profile = PerfProfile::BURST;

    // Load-time HTP power knobs from htp_backend_ext_config.json
    // `devices[].cores[]`, all in microseconds; 0 = leave the backend default.
    // Only options the QAIRT docs mark "Used by qnn-net-run" appear here --
    // offline-preparation keys such as weight_sharing_enabled and num_cores are
    // baked into the context binary and cannot be set when loading one.
    uint32_t rpc_control_latency_us   = 0;
    uint32_t rpc_polling_time_us      = 0;
    uint32_t hmx_timeout_us           = 0;
    uint32_t adaptive_polling_time_us = 0;

    // Decode KV-overlap workers. 0 = serial decode; cpu_mask pins workers (0 = no pin);
    // poll busy-spins for jobs.
    unsigned n_decode_workers = 1;
    uint64_t decode_cpu_mask  = 0;
    bool     decode_poll      = false;

    // HTP (NSP) cores to request per graph via QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES.
    // 0 = auto: derived from the htp_backend_ext_config.json `devices[].cores` list
    // (by modelConfigFromDirectory, or at init when only htp_config_path is set).
    // 1 = force single core (backend default). Values above the device-reported
    // core count are clamped with a warning at init.
    uint32_t num_cores = 0;
};

// Configuration for a VLM
struct VLMConfig {
    ModelConfig llm_config;
    ModelConfig vision_config;
};

// Generation-time parameters passed to LLMModel / VLMModel.
//
// `enable_sampling == false` → greedy argmax fast path (skips the sampler
// chain entirely). Otherwise the geniex-proc chain is driven from these
// fields; `temperature <= 0` still degenerates to greedy at the temp sampler.
struct GenerationConfig {
    int32_t max_tokens = 512;

    // Opt-in ring-buffer context eviction. When a prefill chunk or decode step would
    // exceed the max context length, discards the oldest tokens above
    // `sliding_window_n_keep` instead of throwing ContextLengthExceededError (mirrors
    // llama.cpp's context-shift heuristic; see LLMModel::computeSlideDiscard). Unlike
    // llama.cpp, the surviving tail is re-prefilled rather than renumbered in place --
    // QAIRT's compiled graphs cache post-RoPE K/V with no facility to re-rotate cached
    // history, so relocating KV bytes as-is would leave survivors' RoPE rotation at an
    // out-of-distribution position (see LLMModel::slideWindowEvict).
    //
    // TODO: `sliding_window_n_keep` only anchors a fixed token count today. llama.cpp
    // keeps the whole system prompt in-window (n_keep sized to the system prompt's token
    // count); consider the same here so eviction never discards it.
    bool    sliding_window        = false;
    int32_t sliding_window_n_keep = 4;

    // Sampling (geniex-proc). Zero on top_k/top_p/min_p/penalties is
    // "disabled" inside the chain (matches geniex-proc semantics).
    bool     enable_sampling    = false;
    float    temperature        = 1.0f;
    float    top_p              = 1.0f;
    float    min_p              = 0.0f;
    int32_t  top_k              = 0;
    float    repetition_penalty = 1.0f;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
    int32_t  penalty_last_n     = 64;
    uint32_t seed               = GENIEX_DEFAULT_SEED;

    // Optional GBNF grammar; needs `tokenizer` to actually take effect.
    std::string grammar_str;
    std::string grammar_root = "root";

    // Non-owning. Injected by the pipeline (which owns the tokenizer) so the
    // model can build a Grammar at sampler-init time.
    Tokenizer* tokenizer = nullptr;
};

// Static description of a single graph tensor, populated from GraphInfo_t.
struct TensorSpec {
    std::string    name;
    Qnn_DataType_t dtype = QNN_DATATYPE_FLOAT_32;
    // Graph role: APP_WRITE (input), APP_READ (output), NATIVE, STATIC, etc.
    // Lets callers infer I/O structure from tensor metadata alone.
    Qnn_TensorType_t      type = QNN_TENSOR_TYPE_UNDEFINED;
    std::vector<uint32_t> shape;
    float                 quant_scale  = 1.0f;
    int32_t               quant_offset = 0;
    // Per-channel (axis) quantization: one (scale, offset) per channel.
    // Empty when the tensor uses scalar quant or none.
    std::vector<std::pair<float, int32_t>> axis_quant;
    // True if any dimension may vary at runtime.
    bool has_dynamic_dims = false;

    size_t elementSize() const {
        switch (dtype) {
            case QNN_DATATYPE_FLOAT_32:
            case QNN_DATATYPE_INT_32:
            case QNN_DATATYPE_UINT_32:
            case QNN_DATATYPE_SFIXED_POINT_32:
            case QNN_DATATYPE_UFIXED_POINT_32:
                return 4;
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_INT_16:
            case QNN_DATATYPE_UINT_16:
            case QNN_DATATYPE_SFIXED_POINT_16:
            case QNN_DATATYPE_UFIXED_POINT_16:
                return 2;
            case QNN_DATATYPE_INT_8:
            case QNN_DATATYPE_UINT_8:
            case QNN_DATATYPE_SFIXED_POINT_8:
            case QNN_DATATYPE_UFIXED_POINT_8:
            case QNN_DATATYPE_BOOL_8:
                return 1;
            case QNN_DATATYPE_INT_64:
            case QNN_DATATYPE_UINT_64:
            case QNN_DATATYPE_FLOAT_64:
                return 8;
            default:
                return 0;
        }
    }

    size_t elementCount() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    size_t byteCount() const { return elementSize() * elementCount(); }
};

// Wires one graph's output tensor to another graph's input tensor.
struct Connection {
    int         src_graph_idx;
    std::string src_tensor_name;
    int         dst_graph_idx;
    std::string dst_tensor_name;
};

// Modality-native input for VisionEncoder::encode().
// Decoupled from VLMInput so VisionEncoder can be used outside a VLM context.
struct PixelData {
    std::vector<float>                  pixel_values;    // flat [total_patches * C * H * W]
    std::vector<std::array<int32_t, 3>> image_grid_thw;  // [{T, H, W}] per image

    // Patch-budget encoders (Gemma4, SigLIP2) pad every image to a fixed patch
    // count instead of reporting a grid, and carry per-patch (x, y) ids with
    // (-1, -1) marking padding. Flat [n_images * max_patches * 2].
    // Empty for grid-based encoders.
    std::vector<int32_t> image_position_ids;

    // Soft (vision) tokens each image contributes after spatial pooling — the
    // number of image-token slots it occupies in the prompt. Empty when unused.
    std::vector<int32_t> num_soft_tokens_per_image;
};

}  // namespace geniex

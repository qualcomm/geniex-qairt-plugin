// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//==============================================================================
//
//  Qwen3-4B xtensor inference example
//
//  Low-level xtensor-based inference using direct Graph access.
//  Uses 2-shard model with CPU-side embedding loaded from .npy via xtensor.
//  Does NOT use LLMModel::generate() — all tensor management is done manually
//  with XTensorLLMUtils and Graph::write()/read()/execute().
//
//==============================================================================

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "llm/llm_model.h"
#include "llm/xtensor_utils.h"
#include "qwen3/qwen3.h"
#include "types.h"
#include "xtensor-all.hpp"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD  mode = 0;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(hOut, &mode)) SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

// This example writes every graph tensor by hand, so it must suppress the
// base-class embedding / RoPE providers that createInputProviders() would
// otherwise wire up during initialize().
namespace {
class ManualLLMModel : public geniex::LLMModel {
   public:
    using geniex::LLMModel::LLMModel;

   protected:
    void createInputProviders() override {}
};
}  // namespace

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    int32_t max_tokens = 512;
    bool    verbose    = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>   Max tokens to generate (default 512)\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--verbose")
            args.verbose = true;
        else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    return true;
}

// ── Chat template ─────────────────────────────────────────────────────────────

static std::string applyTemplate(const std::string& user_text) {
    return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto root      = std::filesystem::current_path();
    const auto htp_dir   = root / "third-party" / "windows";
    const auto model_dir = root / "modelfiles" / "qwen3_4b_xtensor";

    geniex::QnnRuntimeConfig runtime_cfg;
    runtime_cfg.backend_path    = (htp_dir / "QnnHtp.dll").string();
    runtime_cfg.system_lib_path = (htp_dir / "QnnSystem.dll").string();
    runtime_cfg.extensions_path = (htp_dir / "QnnHtpNetRunExtensions.dll").string();
    runtime_cfg.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

#ifdef _WIN32
    SetDllDirectoryA(htp_dir.string().c_str());
#endif

    geniex::ModelConfig model_cfg;
    model_cfg.model_paths = {
        (model_dir / "weight_sharing_model_1_of_2.serialized.bin").string(),
        (model_dir / "weight_sharing_model_2_of_2.serialized.bin").string(),
    };
    model_cfg.tokenizer_path = (model_dir / "tokenizer.json").string();
    // embedding_path left empty — we load the table ourselves via xt::load_npy.
    model_cfg.max_tokens = args.max_tokens;

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    // Initialize model (LLMModel handles QNN setup and graph ordering).
    // No input providers added — all tensor management is done manually, so a
    // ManualLLMModel suppresses the base createInputProviders().
    std::cout << "\033[1;36mLoading model...\033[0m\n";
    auto           gc = geniex::parseGenieConfig(model_dir);
    ManualLLMModel model(geniex::buildSpecSkeleton(gc), gc);
    try {
        if (!model.initialize(runtime_cfg, model_cfg)) {
            std::cerr << "Failed to initialize model.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Model initialization error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded.\033[0m\n";

    // Load embedding table via xtensor and tokenizer.
    const std::string embedding_path  = (model_dir / "embed_tokens.npy").string();
    xt::xarray<float> embedding_table = xt::load_npy<float>(embedding_path);
    auto              tokenizer       = geniex::Tokenizer::from_file(model_cfg.tokenizer_path);

    // Architecture constants.
    constexpr std::size_t SEQ_LEN_ARN  = 128;
    constexpr std::size_t KV_LEN_ARN   = 4096 - SEQ_LEN_ARN;
    constexpr std::size_t SEQ_LEN_AR1  = 1;
    constexpr std::size_t KV_LEN_AR1   = 4096 - SEQ_LEN_AR1;
    constexpr std::size_t HIDDEN_SIZE  = 2560;
    constexpr std::size_t NUM_HEADS    = 32;
    constexpr std::size_t NUM_KV_HEADS = 8;
    constexpr std::size_t HEAD_DIM     = 128;
    constexpr std::size_t VOCAB_SIZE   = 151936;
    constexpr float       ROPE_THETA   = 1000000.0f;
    constexpr int32_t     EOS_TOKEN    = 151645;

    // Create xtensor LLM utilities for each graph.
    // Graph ordering after LLMModel::initialize():
    //   graph(0) = ARN shard 0 (layers  0–17)
    //   graph(1) = ARN shard 1 (layers 18–35)
    //   graph(2) = AR1 shard 0 (layers  0–17)
    //   graph(3) = AR1 shard 1 (layers 18–35)
    using XTUtils = geniex::xt_utils::XTensorLLMUtils;

    XTUtils arn_utils_0(SEQ_LEN_ARN,
        KV_LEN_ARN,
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        0,
        17,
        VOCAB_SIZE,
        "input_embeds",
        "_Add_88_Add_output_0",
        ROPE_THETA,
        EOS_TOKEN);
    XTUtils arn_utils_1(SEQ_LEN_ARN,
        KV_LEN_ARN,
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        18,
        35,
        VOCAB_SIZE,
        "_Add_88_Add_output_0",
        "logits",
        ROPE_THETA,
        EOS_TOKEN);
    XTUtils ar1_utils_0(SEQ_LEN_AR1,
        KV_LEN_AR1,
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        0,
        17,
        VOCAB_SIZE,
        "input_embeds",
        "_Add_88_Add_output_0",
        ROPE_THETA,
        EOS_TOKEN);
    XTUtils ar1_utils_1(SEQ_LEN_AR1,
        KV_LEN_AR1,
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        18,
        35,
        VOCAB_SIZE,
        "_Add_88_Add_output_0",
        "logits",
        ROPE_THETA,
        EOS_TOKEN);

    arn_utils_0.initializeTensorMappings(model.graph(0));
    arn_utils_1.initializeTensorMappings(model.graph(1));
    ar1_utils_0.initializeTensorMappings(model.graph(2));
    ar1_utils_1.initializeTensorMappings(model.graph(3));

    std::cout << "\033[1;32mReady.\033[0m\n\n";

    // Chat loop.
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;

        const std::string          prompt_text = applyTemplate(input);
        auto                       encoded     = tokenizer->encode(prompt_text);
        const std::vector<int32_t> input_ids(encoded.begin(), encoded.end());

        // Per-turn state.
        std::size_t                                        n_past = 0;
        std::vector<int32_t>                               generated_tokens;
        std::map<std::string, std::pair<double, uint16_t>> time_log;

        // ── Initialize input/output data vectors with dummy data ─────────────
        std::vector<std::vector<float>> arn_0_input_data, arn_0_output_data;
        std::vector<std::vector<float>> arn_1_input_data, arn_1_output_data;
        std::vector<std::vector<float>> ar1_0_input_data, ar1_0_output_data;
        std::vector<std::vector<float>> ar1_1_input_data, ar1_1_output_data;

        {
            std::vector<int32_t> dummy_ids        = {EOS_TOKEN};
            auto                 dummy_arn_embeds = arn_utils_0.tokensToEmbedding(dummy_ids, embedding_table);
            auto                 d0               = arn_utils_0.prepare_inputs(dummy_arn_embeds, 0, SEQ_LEN_ARN);
            for (const auto& name : arn_utils_0.input_tensor_names) arn_0_input_data.push_back(d0[name]);

            auto d1 = arn_utils_1.prepare_inputs(dummy_arn_embeds, 0, SEQ_LEN_ARN);
            for (const auto& name : arn_utils_1.input_tensor_names) arn_1_input_data.push_back(d1[name]);

            auto dummy_ar1_embeds = ar1_utils_0.tokenToEmbedding(EOS_TOKEN, embedding_table);
            auto d2               = ar1_utils_0.prepare_inputs(dummy_ar1_embeds, 0, 1);
            for (const auto& name : ar1_utils_0.input_tensor_names) ar1_0_input_data.push_back(d2[name]);

            auto d3 = ar1_utils_1.prepare_inputs(dummy_ar1_embeds, 0, 1);
            for (const auto& name : ar1_utils_1.input_tensor_names) ar1_1_input_data.push_back(d3[name]);
        }

        const auto                                     t_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_first_token;
        bool                                           got_first_token = false;

        // ── Prefill (ARN, seq_len=128) ───────────────────────────────────────
        std::size_t curr_len         = input_ids.size();
        std::size_t tokens_processed = 0;
        std::size_t last_chunk_size  = 0;

        try {
            while (tokens_processed < curr_len) {
                std::size_t tokens_in_chunk = std::min(SEQ_LEN_ARN, curr_len - tokens_processed);
                last_chunk_size             = tokens_in_chunk;

                std::vector<int32_t> chunk_tokens(
                    input_ids.begin() + tokens_processed, input_ids.begin() + tokens_processed + tokens_in_chunk);

                auto input_embeds = arn_utils_0.tokensToEmbedding(chunk_tokens, embedding_table);
                auto pos          = arn_utils_0.get_position_ids(n_past, SEQ_LEN_ARN);
                auto [cos, sin]   = arn_utils_0.get_cos_sin(pos);
                auto attn_mask    = arn_utils_0.get_attention_mask(n_past, tokens_in_chunk);

                std::vector<float> embeds_flat(input_embeds.begin(), input_embeds.end());
                std::vector<float> cos_flat(cos.begin(), cos.end());
                std::vector<float> sin_flat(sin.begin(), sin.end());
                std::vector<float> mask_flat(attn_mask.begin(), attn_mask.end());

                // ── ARN shard 0 ──────────────────────────────────────────────
                arn_0_input_data[arn_utils_0.input_tensor_order[arn_utils_0.in_states_name_]] = embeds_flat;
                arn_0_input_data[arn_utils_0.input_tensor_order["position_ids_cos"]]          = cos_flat;
                arn_0_input_data[arn_utils_0.input_tensor_order["position_ids_sin"]]          = sin_flat;
                arn_0_input_data[arn_utils_0.input_tensor_order["attention_mask"]]            = mask_flat;

                arn_utils_0.writeInputs(model.graph(0), arn_0_input_data);
                model.graph(0).execute(time_log);
                arn_utils_0.readOutputs(model.graph(0), arn_0_output_data);

                // ── ARN shard 1 (receives hidden state from shard 0) ─────────
                arn_1_input_data[arn_utils_1.input_tensor_order[arn_utils_1.in_states_name_]] =
                    arn_0_output_data[arn_utils_0.output_tensor_order[arn_utils_0.out_states_name_]];
                arn_1_input_data[arn_utils_1.input_tensor_order["position_ids_cos"]] = cos_flat;
                arn_1_input_data[arn_utils_1.input_tensor_order["position_ids_sin"]] = sin_flat;
                arn_1_input_data[arn_utils_1.input_tensor_order["attention_mask"]]   = mask_flat;

                arn_utils_1.writeInputs(model.graph(1), arn_1_input_data);
                model.graph(1).execute(time_log);
                arn_utils_1.readOutputs(model.graph(1), arn_1_output_data);

                // Update KV caches.
                n_past += tokens_in_chunk;
                arn_utils_0.update_kv_cache(arn_0_input_data,
                    arn_0_output_data,
                    static_cast<int>(n_past - tokens_in_chunk),
                    static_cast<int>(tokens_in_chunk),
                    SEQ_LEN_ARN);
                arn_utils_1.update_kv_cache(arn_1_input_data,
                    arn_1_output_data,
                    static_cast<int>(n_past - tokens_in_chunk),
                    static_cast<int>(tokens_in_chunk),
                    SEQ_LEN_ARN);
                tokens_processed += tokens_in_chunk;
            }

            // ── First token from ARN logits ──────────────────────────────────
            auto logits_view =
                xt::view(xt::adapt(arn_1_output_data[arn_utils_1.output_tensor_order[arn_utils_1.out_states_name_]],
                             std::vector<std::size_t>{1, SEQ_LEN_ARN, VOCAB_SIZE}),
                    0,
                    last_chunk_size - 1,
                    xt::all());
            int32_t next_token = static_cast<int32_t>(xt::argmax(logits_view)());

            t_first_token   = std::chrono::high_resolution_clock::now();
            got_first_token = true;
            std::cout << "\033[33m" << tokenizer->decode_token(next_token) << std::flush;
            generated_tokens.push_back(next_token);

            // ── Transfer KV cache from ARN to AR1 ────────────────────────────
            ar1_utils_0.transfer_kv_cache(ar1_0_input_data, arn_0_input_data, static_cast<int>(n_past));
            ar1_utils_1.transfer_kv_cache(ar1_1_input_data, arn_1_input_data, static_cast<int>(n_past));

            // ── Decode loop (AR1, seq_len=1) ─────────────────────────────────
            for (int32_t i = 0; i < args.max_tokens && next_token != EOS_TOKEN && next_token != 151643; ++i) {
                auto input_embeds = ar1_utils_0.tokenToEmbedding(next_token, embedding_table);
                auto pos          = ar1_utils_0.get_position_ids(n_past, 1);
                auto [cos, sin]   = ar1_utils_0.get_cos_sin(pos);
                auto attn_mask    = ar1_utils_0.get_attention_mask(n_past, 1);

                std::vector<float> embeds_flat(input_embeds.begin(), input_embeds.end());
                std::vector<float> cos_flat(cos.begin(), cos.end());
                std::vector<float> sin_flat(sin.begin(), sin.end());
                std::vector<float> mask_flat(attn_mask.begin(), attn_mask.end());

                // ── AR1 shard 0 ──────────────────────────────────────────────
                ar1_0_input_data[ar1_utils_0.input_tensor_order[ar1_utils_0.in_states_name_]] = embeds_flat;
                ar1_0_input_data[ar1_utils_0.input_tensor_order["position_ids_cos"]]          = cos_flat;
                ar1_0_input_data[ar1_utils_0.input_tensor_order["position_ids_sin"]]          = sin_flat;
                ar1_0_input_data[ar1_utils_0.input_tensor_order["attention_mask"]]            = mask_flat;

                ar1_utils_0.writeInputs(model.graph(2), ar1_0_input_data);
                model.graph(2).execute(time_log);
                ar1_utils_0.readOutputs(model.graph(2), ar1_0_output_data);

                // ── AR1 shard 1 (receives hidden state from shard 0) ─────────
                ar1_1_input_data[ar1_utils_1.input_tensor_order[ar1_utils_1.in_states_name_]] =
                    ar1_0_output_data[ar1_utils_0.output_tensor_order[ar1_utils_0.out_states_name_]];
                ar1_1_input_data[ar1_utils_1.input_tensor_order["position_ids_cos"]] = cos_flat;
                ar1_1_input_data[ar1_utils_1.input_tensor_order["position_ids_sin"]] = sin_flat;
                ar1_1_input_data[ar1_utils_1.input_tensor_order["attention_mask"]]   = mask_flat;

                ar1_utils_1.writeInputs(model.graph(3), ar1_1_input_data);
                model.graph(3).execute(time_log);
                n_past += 1;
                ar1_utils_1.readOutputs(model.graph(3), ar1_1_output_data);

                // Update KV caches.
                ar1_utils_0.update_kv_cache(ar1_0_input_data, ar1_0_output_data, static_cast<int>(n_past - 1), 1, 1);
                ar1_utils_1.update_kv_cache(ar1_1_input_data, ar1_1_output_data, static_cast<int>(n_past - 1), 1, 1);

                // Sample next token.
                auto logits_ar1 =
                    xt::view(xt::adapt(ar1_1_output_data[ar1_utils_1.output_tensor_order[ar1_utils_1.out_states_name_]],
                                 std::vector<std::size_t>{1, 1, VOCAB_SIZE}),
                        0,
                        0,
                        xt::all());
                next_token = static_cast<int32_t>(xt::argmax(logits_ar1)());

                if (next_token == EOS_TOKEN || next_token == 151643) break;
                std::cout << tokenizer->decode_token(next_token) << std::flush;
                generated_tokens.push_back(next_token);
            }
        } catch (const std::exception& e) {
            std::cout << "\033[0m\n";
            std::cerr << "Generation error: " << e.what() << "\n";
            std::cerr.flush();
            continue;
        }
        std::cout << "\033[0m\n";

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (got_first_token) {
            const double ttft_ms    = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
            const double decode_ms  = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
            const size_t decode_tok = generated_tokens.size() > 1 ? generated_tokens.size() - 1 : 0;
            const double tps        = decode_ms > 0.0 ? decode_tok / (decode_ms / 1000.0) : 0.0;

            if (args.verbose) {
                std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                          << "Generated tokens : " << generated_tokens.size() << "\n"
                          << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms << " ms\n"
                          << "Decode time      : " << std::fixed << std::setprecision(1) << decode_ms << " ms\n"
                          << "Decode speed     : " << std::fixed << std::setprecision(2) << tps << " tokens/s\n"
                          << "===================\n\n";
            } else {
                std::cout << "TTFT: " << std::fixed << std::setprecision(1) << ttft_ms << " ms"
                          << "  |  " << std::setprecision(2) << tps << " tokens/s\n\n";
            }
        }
    }

    return 0;
}
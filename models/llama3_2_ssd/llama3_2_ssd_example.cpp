// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause


#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ssd_model.h"
#include "tokenizers_cpp.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "types.h"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD mode = 0;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(hOut, &mode))
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

struct Args {
    int32_t max_tokens = 1024;
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
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--max-tokens") args.max_tokens = std::stoi(next());
        else if (a == "--verbose")    args.verbose    = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

static std::string applyTemplate(const std::string& user_text, bool first_turn) {
    std::string prompt;
    if (first_turn) {
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                 "You are a helpful AI assistant<|eot_id|>";
    }
    prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
            + user_text
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto root = std::filesystem::current_path();
    const auto htp_dir = root / "third-party" / "windows";
    const auto model_dir = root / "modelfiles" / "llama_v3_2_3b_instruct_ssd-genie-w4a16-qualcomm_snapdragon_x_elite";

    geniex::QnnRuntimeConfig runtime_cfg;

    runtime_cfg.backend_path    = (htp_dir / "QnnHtp.dll").string();
    runtime_cfg.system_lib_path = (htp_dir / "QnnSystem.dll").string();
    runtime_cfg.extensions_path = (htp_dir / "QnnHtpNetRunExtensions.dll").string();

#ifdef _WIN32
    SetDllDirectoryA(htp_dir.string().c_str());
#endif

    geniex::ModelConfig model_cfg;
    model_cfg.model_paths = {
        (model_dir / "llama_v3_2_3b_instruct_ssd_part_1_of_3.bin").string(),
        (model_dir / "llama_v3_2_3b_instruct_ssd_part_2_of_3.bin").string(),
        (model_dir / "llama_v3_2_3b_instruct_ssd_part_3_of_3.bin").string(),
    };
    model_cfg.tokenizer_path  = (model_dir / "tokenizer.json").string();
    // No embedding_path needed – embedding runs on-device in shard 0.
    model_cfg.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    std::cout << "\033[1;36mLoading Llama-3.2-3B-Instruct-SSD...\033[0m\n";
    geniex::SSDModel model = geniex::llama3_2_3b_ssd::makeModel(
        (model_dir / "forecast-prefix" / "kv-cache.primary.qnn-htp").string());

    try {
        if (!model.initialize(runtime_cfg, model_cfg)) {
            std::cerr << "Failed to initialize model.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Model initialization error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded (SSD: branches=[3,2], forecast_prefix=16).\033[0m\n\n";

    auto tokenizer = tokenizers::Tokenizer::FromJSON(model_cfg.tokenizer_path);

    bool first_turn = true;
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;

        const std::string prompt_text = applyTemplate(input, first_turn);
        first_turn = false;

        auto encoded = tokenizer->Encode(prompt_text);
        const std::vector<int32_t> prompt_tokens(encoded.begin(), encoded.end());

        const auto t_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_first_token;
        bool got_first_token = false;

        std::cout << "\033[33m";
        std::vector<int32_t> output_tokens;
        try {
            output_tokens = model.generate(
                prompt_tokens,
                gen_cfg,
                [&](int32_t tok) {
                    if (!got_first_token) {
                        t_first_token   = std::chrono::high_resolution_clock::now();
                        got_first_token = true;
                    }
                    std::cout << tokenizer->Decode({tok}) << std::flush;
                    return true;
                });
        } catch (const std::exception& e) {
            std::cout << "\033[0m\n";
            std::cerr << "Generation error: " << e.what() << "\n";
            std::cerr.flush();
            model.resetKVCache();
            first_turn = true;
            continue;
        }
        std::cout << "\033[0m\n";

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (got_first_token) {
            const double ttft_ms    = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
            const double decode_ms  = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
            const size_t decode_tok = output_tokens.size() > 1 ? output_tokens.size() - 1 : 0;
            const double tps        = decode_ms > 0.0 ? decode_tok / (decode_ms / 1000.0) : 0.0;

            if (args.verbose) {
                std::cout << "\033[1;36m=== Performance (SSD) ===\033[0m\n"
                          << "Generated tokens : " << output_tokens.size() << "\n"
                          << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms  << " ms\n"
                          << "Decode time      : " << std::fixed << std::setprecision(1) << decode_ms << " ms\n"
                          << "Decode speed     : " << std::fixed << std::setprecision(2) << tps << " tokens/s\n"
                          << "=========================\n\n";
            } else {
                std::cout << "TTFT: " << std::fixed << std::setprecision(1) << ttft_ms << " ms"
                          << "  |  " << std::setprecision(2) << tps << " tokens/s\n\n";
            }
        }
    }

    return 0;
}

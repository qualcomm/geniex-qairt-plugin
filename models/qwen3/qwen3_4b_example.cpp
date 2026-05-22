// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "llm/llm_model.h"
#include "qwen3/qwen3.h"
#include "types.h"

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

struct Args {
    int32_t max_tokens      = 512;
    bool    verbose         = false;
    bool    enable_thinking = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>   Max tokens to generate (default 512)\n"
              << "  --thinking         Enable thinking mode\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--thinking")
            args.enable_thinking = true;
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

static std::string applyTemplate(const std::string& user_text) {
    return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir = std::filesystem::current_path() / "modelfiles" / "qwen3_4b";

    // All QNN runtime paths are left as std::nullopt → auto-detected from
    // htp-files/ installed alongside geniex_core.
    geniex::QnnRuntimeConfig runtime_cfg;

    geniex::ModelConfig model_cfg;
    model_cfg.model_paths = {
        (model_dir / "qwen3_4b_w4a16_part_1_of_4.bin").string(),
        (model_dir / "qwen3_4b_w4a16_part_2_of_4.bin").string(),
        (model_dir / "qwen3_4b_w4a16_part_3_of_4.bin").string(),
        (model_dir / "qwen3_4b_w4a16_part_4_of_4.bin").string(),
    };
    model_cfg.tokenizer_path = (model_dir / "tokenizer.json").string();
    // No embedding_path needed – embedding runs on-device in shard 0.
    model_cfg.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens    = args.max_tokens;
    gen_cfg.thinking_mode = args.enable_thinking;

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    std::cout << "\033[1;36mLoading model...\033[0m\n";
    geniex::LLMModel model = geniex::qwen3::makeModel(model_cfg);
    try {
        if (!model.initialize(runtime_cfg, model_cfg)) {
            std::cerr << "Failed to initialize model.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Model initialization error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    auto tokenizer = geniex::Tokenizer::from_file(model_cfg.tokenizer_path);

    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;

        const std::string prompt_text = applyTemplate(input);

        auto                       encoded = tokenizer->encode(prompt_text);
        const std::vector<int32_t> prompt_tokens(encoded.begin(), encoded.end());

        const auto                                     t_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_first_token;
        bool                                           got_first_token = false;

        std::cout << "\033[33m";
        std::vector<int32_t> output_tokens;
        try {
            output_tokens = model.generate(prompt_tokens, gen_cfg, [&](int32_t tok) {
                if (!got_first_token) {
                    t_first_token   = std::chrono::high_resolution_clock::now();
                    got_first_token = true;
                }
                std::cout << tokenizer->decode_token(tok) << std::flush;
                return true;
            });
        } catch (const std::exception& e) {
            std::cout << "\033[0m\n";
            std::cerr << "Generation error: " << e.what() << "\n";
            std::cerr.flush();
            model.resetKVCache();
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
                std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                          << "Generated tokens : " << output_tokens.size() << "\n"
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

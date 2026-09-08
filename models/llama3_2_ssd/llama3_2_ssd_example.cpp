// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "llama3_2_ssd/llama3_2_ssd.h"
#include "llm/llm_spec_loader.h"
#include "ssd_model.h"
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
    std::string model_dir;
    // Pre-templated prompt, tokenized verbatim, run once non-interactively. Lets a
    // run be compared token-for-token against another runtime on the same input.
    std::string raw_prompt_file;
    int32_t     max_tokens = 1024;
    bool        verbose    = false;
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
        if (a == "--model-dir")
            args.model_dir = next();
        else if (a == "--raw-prompt-file")
            args.raw_prompt_file = next();
        else if (a == "--max-tokens")
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

static std::string applyTemplate(const std::string& user_text, bool first_turn) {
    std::string prompt;
    if (first_turn) {
        prompt =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful AI assistant<|eot_id|>";
    }
    prompt += "<|start_header_id|>user<|end_header_id|>\n\n" + user_text +
              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto root      = std::filesystem::current_path();
    const auto htp_dir   = root / "third-party" / "windows";
    const auto model_dir = args.model_dir.empty() ? (root / "modelfiles" / "llama_v3_2_3b_instruct_ssd")
                                                  : std::filesystem::path(args.model_dir);

    geniex::QnnRuntimeConfig runtime_cfg;

    runtime_cfg.backend_path    = (htp_dir / "QnnHtp.dll").string();
    runtime_cfg.system_lib_path = (htp_dir / "QnnSystem.dll").string();
    runtime_cfg.extensions_path = (htp_dir / "QnnHtpNetRunExtensions.dll").string();

#ifdef _WIN32
    SetDllDirectoryA(htp_dir.string().c_str());
#endif

    // Resolve everything from the bundle (genie_config.json ctx-bins, tokenizer,
    // HTP extensions) rather than hardcoding the legacy part filenames, so newer
    // exports -- which name their shards partN_of_4.bin -- load unchanged.
    geniex::ModelConfig model_cfg = geniex::modelConfigFromDirectory(model_dir);

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
    if (!model_cfg.forecast_prefix_path) {
        model_cfg.forecast_prefix_path = (model_dir / "forecast-prefix" / "kv-cache.primary.qnn-htp").string();
    }
    geniex::SSDModel model = geniex::llama3_2_3b_ssd::makeModel(model_cfg);

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

    auto tokenizer = geniex::Tokenizer::from_file(model_cfg.tokenizer_path);

    bool first_turn = true;
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string prompt_text;
        if (!args.raw_prompt_file.empty()) {
            std::ifstream f(args.raw_prompt_file, std::ios::binary);
            if (!f) {
                std::cerr << "Cannot open raw prompt file: " << args.raw_prompt_file << "\n";
                return 1;
            }
            std::stringstream ss;
            ss << f.rdbuf();
            prompt_text = ss.str();
        } else {
            std::string input;
            if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
            prompt_text = applyTemplate(input, first_turn);
            first_turn  = false;
        }

        const std::vector<int32_t> prompt_tokens = tokenizer->encode(prompt_text);

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
                          << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms << " ms\n"
                          << "Decode time      : " << std::fixed << std::setprecision(1) << decode_ms << " ms\n"
                          << "Decode speed     : " << std::fixed << std::setprecision(2) << tps << " tokens/s\n"
                          << "=========================\n\n";
            } else {
                std::cout << "TTFT: " << std::fixed << std::setprecision(1) << ttft_ms << " ms"
                          << "  |  " << std::setprecision(2) << tps << " tokens/s\n\n";
            }
        }

        if (!args.raw_prompt_file.empty()) break;  // single-shot mode
    }

    return 0;
}

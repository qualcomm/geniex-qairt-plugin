// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "pipeline/auto_model.h"
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

namespace fs = std::filesystem;

struct Args {
    fs::path    model_dir;
    std::string prompt;  // non-empty → single-shot, non-interactive
    int32_t     max_tokens      = 512;
    bool        verbose         = false;
    bool        enable_thinking = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model-dir <path> QAIRT bundle directory (default: ./modelfiles/qwen3_1_7b)\n"
              << "  --prompt <text>    Run once on this prompt and exit (non-interactive)\n"
              << "  --max-tokens <n>   Max tokens to generate (default 512)\n"
              << "  --thinking         Enable thinking mode\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model-dir")
            args.model_dir = next();
        else if (a == "--prompt")
            args.prompt = next();
        else if (a == "--max-tokens")
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
    if (args.model_dir.empty()) args.model_dir = fs::current_path() / "modelfiles" / "qwen3_1_7b";
    return true;
}

// Runs one turn: render via the bundled Jinja template, stream the reply.
static void runTurn(geniex::LLMPipeline& pipe, const std::string& user_text, const Args& args) {
    geniex::ApplyChatTemplateOptions opts;
    opts.enable_thinking = args.enable_thinking;

    std::string prompt;
    try {
        prompt = pipe.applyChatTemplate({{geniex::Role::User, user_text}}, opts);
    } catch (const std::exception& e) {
        std::cerr << "Chat-template error: " << e.what() << "\n";
        return;
    }

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    std::cout << "\033[33m";
    const auto result = pipe.generate(prompt, gen_cfg, [](const char* piece) {
        std::cout << piece << std::flush;
        return true;
    });
    std::cout << "\033[0m\n";

    if (args.verbose) {
        std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                  << "Generated tokens : " << result.generated_tokens << "\n"
                  << "TTFT             : " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms\n"
                  << "Decode time      : " << std::fixed << std::setprecision(1) << result.decode_ms << " ms\n"
                  << "Decode speed     : " << std::fixed << std::setprecision(2) << result.tokens_per_second
                  << " tokens/s\n"
                  << "Stop reason      : " << result.stop_reason << "\n"
                  << "===================\n\n";
    } else {
        std::cout << "TTFT: " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms"
                  << "  |  " << std::setprecision(2) << result.tokens_per_second << " tokens/s\n\n";
    }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    geniex::QnnRuntimeConfig runtime_cfg;

    // Derive model_paths (in ctx-bins order), tokenizer_path, and
    // htp_config_path straight from the bundle's genie_config.json rather
    // than hardcoding shard filenames, since bundles name their shards
    // differently (this qwen3_1_7b bundle ships "partN_of_4.bin").
    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(args.model_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read bundle: " << e.what() << "\n";
        return 1;
    }
    // tokenizer_config_path left unset → discovered next to the bundle.

    std::cout << "\033[1;36mLoading model from " << args.model_dir.string() << "...\033[0m\n";
    auto pipe_opt = geniex::auto_model::makePipeline(runtime_cfg, model_cfg, {/*prepend_bos=*/true});
    if (!pipe_opt) {
        std::cerr << "Failed to create pipeline. See logs for details.\n";
        return 1;
    }
    auto& pipe = *pipe_opt;
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    if (!args.prompt.empty()) {
        runTurn(pipe, args.prompt, args);
        return 0;
    }

    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        runTurn(pipe, input, args);
        pipe.reset();
    }
    return 0;
}

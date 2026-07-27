// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gemma4/gemma4.h"
#include "types.h"

// Which Gemma4 variant this executable defaults to. E2B and E4B differ only in
// dimensions (hidden 1536 vs 2560, 35 vs 42 layers, per-layer stream 8960 vs
// 10752, 1 vs 2 KV heads) — every one of those is read from the bundle's
// genie_config.json or inferred from the loaded graphs, so the two targets share
// this source verbatim. Set by CMake; see models/gemma4/CMakeLists.txt.
#ifndef GEMMA4_VARIANT
#define GEMMA4_VARIANT "gemma4_e2b"
#endif
#ifndef GEMMA4_VARIANT_LABEL
#define GEMMA4_VARIANT_LABEL "Gemma4-E2B"
#endif

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
    fs::path                 model_dir;
    std::string              prompt;  // non-empty → single-shot, non-interactive
    std::vector<std::string> turns;   // --turn (repeatable) → scripted multi-round
    int32_t                  max_tokens = 256;
    bool                     verbose    = false;
    bool                     chat       = false;  // apply chat template (instruct models)
    bool                     greedy     = false;  // force argmax, ignore bundle sampler
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model-dir <path> QAIRT bundle directory (default: ./modelfiles/" GEMMA4_VARIANT ")\n"
              << "  --prompt <text>    Run once on this prompt and exit (non-interactive)\n"
              << "  --turn <text>      One user turn of a scripted conversation; repeat to\n"
              << "                     script multiple rounds (implies --chat). KV is kept\n"
              << "                     across turns, so each round only prefills its own text.\n"
              << "  --max-tokens <n>   Max tokens to generate (default 256)\n"
              << "  --chat             Apply the chat template (instruct-tuned checkpoints)\n"
              << "  --greedy           Force greedy argmax decoding (ignore the bundle sampler)\n"
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
        else if (a == "--turn")
            args.turns.push_back(next());
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--chat")
            args.chat = true;
        else if (a == "--greedy")
            args.greedy = true;
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
    if (args.model_dir.empty()) args.model_dir = fs::current_path() / "modelfiles" / GEMMA4_VARIANT;
    return true;
}

// Gemma4 turn framing. The chat template renders the FIRST turn (it also emits
// <bos>, which the pipeline would otherwise prepend itself). Continuation turns
// must not repeat <bos> or replay the history: the KV cache already holds it, so
// we only feed the delta -- close the model's turn, then open the next
// user/model pair. Genie has to re-send the whole transcript every round because
// its "basic" dialog is stateless; keeping KV is why round 2 here prefills ~20
// tokens instead of ~336.
static std::string continuationPrompt(const std::string& user_text) {
    return "<turn|>\n<|turn>user\n" + user_text + "<turn|>\n<|turn>model\n";
}

// Builds the per-call GenerationConfig from the bundle's genie_config.json
// sampler block (temp / top-k / top-p / seed), so the example decodes exactly
// like Genie does. Without this the config defaults to enable_sampling=false,
// i.e. greedy argmax, which on an instruct model degenerates into verbatim
// repetition ("...Washington, D.C.\n\n..." forever). `--greedy` opts back out.
static geniex::GenerationConfig buildGenConfig(const fs::path& model_dir, const Args& args) {
    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    if (args.greedy) return gen_cfg;  // enable_sampling stays false → argmax

    geniex::ParsedSamplerConfig s;
    try {
        s = geniex::parseGenieSamplerConfig(model_dir);
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("gemma4: could not read sampler config ({}); falling back to greedy", e.what());
        return gen_cfg;
    }

    // Any sampler field present in the bundle turns sampling on. Defaults mirror
    // Gemma4's shipped genie_config (temp 0.8 / top-k 40 / top-p 0.95, seed 42).
    const bool has_any = s.temperature || s.top_k || s.top_p || s.seed;
    if (!has_any) return gen_cfg;  // no sampler block → keep greedy

    gen_cfg.enable_sampling = true;
    gen_cfg.temperature     = s.temperature.value_or(0.8f);
    gen_cfg.top_k           = s.top_k.value_or(40);
    gen_cfg.top_p           = s.top_p.value_or(0.95f);
    if (s.seed) gen_cfg.seed = *s.seed;
    if (s.repetition_penalty) gen_cfg.repetition_penalty = *s.repetition_penalty;
    if (s.presence_penalty) gen_cfg.presence_penalty = *s.presence_penalty;
    if (s.frequency_penalty) gen_cfg.frequency_penalty = *s.frequency_penalty;
    if (s.penalty_last_n) gen_cfg.penalty_last_n = *s.penalty_last_n;
    return gen_cfg;
}

static void runTurn(geniex::LLMPipeline& pipe, const std::string& user_text, const Args& args,
    const geniex::GenerationConfig& gen_cfg, bool first_turn = true) {
    std::string prompt = user_text;
    if (args.chat) {
        try {
            prompt = first_turn ? pipe.applyChatTemplate({{geniex::Role::User, user_text}}, {})
                                : continuationPrompt(user_text);
        } catch (const std::exception& e) {
            std::cerr << "Chat-template error: " << e.what() << "\n";
            return;
        }
    }

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

    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(args.model_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read bundle: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\033[1;36mLoading " GEMMA4_VARIANT_LABEL " from " << args.model_dir.string() << "...\033[0m\n";
    auto pipe_opt = geniex::gemma4::makePipeline(runtime_cfg, model_cfg);
    if (!pipe_opt) {
        std::cerr << "Failed to create pipeline. See logs for details.\n";
        return 1;
    }
    auto& pipe = *pipe_opt;
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    const geniex::GenerationConfig gen_cfg = buildGenConfig(args.model_dir, args);

    if (!args.turns.empty()) {
        // Scripted conversation: one KV cache carried across every round.
        Args turn_args = args;
        turn_args.chat = true;
        for (size_t i = 0; i < args.turns.size(); ++i) {
            std::cout << "\033[1;35m=== Round " << (i + 1) << " — user: " << args.turns[i] << "\033[0m\n";
            runTurn(pipe, args.turns[i], turn_args, gen_cfg, /*first_turn=*/i == 0);
        }
        return 0;
    }

    if (!args.prompt.empty()) {
        runTurn(pipe, args.prompt, args, gen_cfg);
        return 0;
    }

    // Interactive: also a conversation, so KV is kept and only the first turn
    // goes through the chat template.
    bool first = true;
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit, 'reset' to clear history): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        if (input == "reset") {
            pipe.reset();
            first = true;
            std::cout << "\033[1;35m[history cleared]\033[0m\n";
            continue;
        }
        runTurn(pipe, input, args, gen_cfg, first);
        first = false;
    }
    return 0;
}

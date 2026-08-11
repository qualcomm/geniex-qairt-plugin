// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Standalone EAGLE (eaglet) speculative-decoding example for Qwen3-4B.
// Loads the two-engine bundle (target + draft), applies the Qwen3 chat
// template, and streams generated tokens. Intended to run on-device:
//
//   ./eaglet_example --model-dir /data/local/tmp/qwen3_eaglet/model \
//       --prompt "..." --max-tokens 128 [--thinking] [--verbose]

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "qwen3_eaglet.h"
#include "types.h"

namespace fs = std::filesystem;

struct Args {
    std::string model_dir;
    std::string prompt;
    std::string raw_prompt_file;  // tokenize file verbatim, no chat template
    int32_t     max_tokens = 128;
    bool        thinking   = false;
    bool        verbose    = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model-dir <dir> [OPTIONS]\n"
              << "  --model-dir <dir>  Bundle directory (genie_config.json + ctx-bins)\n"
              << "  --prompt <text>    Prompt; if omitted, reads prompt.txt in the bundle\n"
              << "  --max-tokens <n>   Max tokens to generate (default 128)\n"
              << "  --thinking         Enable Qwen3 <think> reasoning block\n"
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
        else if (a == "--raw-prompt-file")
            args.raw_prompt_file = next();
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--thinking")
            args.thinking = true;
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
    if (args.model_dir.empty()) {
        printUsage(argv[0]);
        return false;
    }
    return true;
}

// Minimal Qwen3 chat template (matches the tokenizer's chat_template). The
// assistant turn optionally opens an empty <think> block to suppress reasoning.
static std::string applyQwen3Template(const std::string& user_text, bool thinking) {
    std::string prompt = "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    if (!thinking) prompt += "<think>\n\n</think>\n\n";
    return prompt;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const fs::path model_dir(args.model_dir);

    // Discover the genie_config.json (any *.json whose dialog.type == eaglet is
    // resolved inside makeModel; here we just need model_paths + tokenizer).
    std::string cfg_json;
    for (auto& e : fs::directory_iterator(model_dir)) {
        if (e.path().extension() == ".json") {
            std::ifstream probe(e.path());
            std::string   content((std::istreambuf_iterator<char>(probe)), std::istreambuf_iterator<char>());
            if (content.find("\"eaglet\"") != std::string::npos) {
                cfg_json = e.path().string();
                break;
            }
        }
    }
    if (cfg_json.empty()) {
        std::cerr << "No eaglet genie_config.json found in " << model_dir << "\n";
        return 1;
    }

    // Resolve the target ctx-bins from the config's target engine.
    geniex::qwen3_eaglet::json root;
    {
        std::ifstream f(cfg_json);
        root = geniex::qwen3_eaglet::json::parse(f);
    }
    geniex::ModelConfig model_cfg;
    for (const auto& eng : root["dialog"]["engine"]) {
        if (eng.value("role", "") != "target") continue;
        for (const auto& b : eng["model"]["binary"]["ctx-bins"])
            model_cfg.model_paths.push_back((model_dir / b.get<std::string>()).string());
    }
    if (model_cfg.model_paths.empty()) {
        std::cerr << "No target engine ctx-bins in config.\n";
        return 1;
    }
    model_cfg.tokenizer_path = (model_dir / root["dialog"]["tokenizer"].value("path", "tokenizer.json")).string();
    model_cfg.embedding_path =
        (model_dir / root["dialog"]["embedding"].value("lut-path", "quantized_embedding_table.bin")).string();
    model_cfg.htp_config_path =
        (model_dir / root["dialog"]["engine"][0]["backend"].value("extensions", "htp_backend_ext_config_mc.json"))
            .string();

    geniex::QnnRuntimeConfig runtime_cfg;  // paths auto-resolve from geniex_core's htp-files/

    std::cout << "Loading Qwen3-4B eaglet (target + draft)...\n";
    std::unique_ptr<geniex::EagleModel> model;
    try {
        model = geniex::qwen3_eaglet::makeModel(runtime_cfg, model_cfg);
    } catch (const std::exception& e) {
        std::cerr << "Model load error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Model loaded.\n\n";

    auto tokenizer = geniex::Tokenizer::from_file(model_cfg.tokenizer_path);

    // --raw-prompt-file feeds an already-formatted prompt (system + chat markers
    // baked in) verbatim, with no chat template applied. --prompt / prompt.txt go
    // through the Qwen3 chat template instead.
    std::vector<int32_t> prompt_tokens;
    if (!args.raw_prompt_file.empty()) {
        std::ifstream pf(args.raw_prompt_file, std::ios::binary);
        if (!pf) {
            std::cerr << "Cannot open raw prompt file: " << args.raw_prompt_file << "\n";
            return 1;
        }
        std::stringstream ss;
        ss << pf.rdbuf();
        prompt_tokens = tokenizer->encode(ss.str());
    } else {
        std::string user_text = args.prompt;
        if (user_text.empty()) {
            std::ifstream pf(model_dir / "prompt.txt");
            if (pf) {
                std::stringstream ss;
                ss << pf.rdbuf();
                user_text = ss.str();
                while (!user_text.empty() && (user_text.back() == '\n' || user_text.back() == '\r'))
                    user_text.pop_back();
            }
        }
        if (user_text.empty()) {
            std::cerr << "No prompt provided (use --prompt or a prompt.txt in the bundle).\n";
            return 1;
        }
        const std::string prompt_text = applyQwen3Template(user_text, args.thinking);
        prompt_tokens                 = tokenizer->encode(prompt_text);
    }

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    const auto                                     t_start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t_first;
    bool                                           got_first = false;

    std::vector<int32_t> out;
    try {
        out = model->generate(prompt_tokens, gen_cfg, [&](int32_t tok) {
            if (!got_first) {
                t_first   = std::chrono::high_resolution_clock::now();
                got_first = true;
            }
            std::cout << tokenizer->decode_token(tok) << std::flush;
            return true;
        });
    } catch (const std::exception& e) {
        std::cerr << "\nGeneration error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\n";

    if (args.verbose && got_first) {
        const auto   t_end     = std::chrono::high_resolution_clock::now();
        const double ttft_ms   = std::chrono::duration<double, std::milli>(t_first - t_start).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first).count();
        const size_t dtok      = out.size();
        const double tps       = decode_ms > 0.0 ? dtok / (decode_ms / 1000.0) : 0.0;
        const auto&  stats     = model->lastStats();
        std::cout << "\n=== Performance (eaglet) ===\n"
                  << "Generated tokens : " << out.size() << "\n"
                  << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms << " ms\n"
                  << "Decode time      : " << std::fixed << std::setprecision(1) << decode_ms << " ms\n"
                  << "Decode speed     : " << std::fixed << std::setprecision(2) << tps << " tokens/s\n"
                  << "Verify rounds    : " << stats.iterations << "\n"
                  << "Tokens/round     : " << std::fixed << std::setprecision(2) << stats.meanAcceptedTokensPerRound()
                  << "\n";
        std::cout << "============================\n";
    }
    return 0;
}

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
              << "  --model-dir <dir>  Bundle directory (metadata.json + ctx-bins)\n"
              << "  --prompt <text>    Bare user turn, rendered through the chat template\n"
              << "  --raw-prompt-file  Pre-formatted prompt, tokenized verbatim (no template)\n"
              << "  --max-tokens <n>   Max tokens to generate (default 128)\n"
              << "  --thinking         Enable Qwen3 <think> reasoning block\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n"
              << "\nWith neither --prompt nor --raw-prompt-file, the bundle's prompt.txt is\n"
              << "read verbatim (it already carries the chat markers).\n";
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

// Minimal Qwen3 chat template, used only when the bundle ships no
// tokenizer_config.json chat_template (e.g. a stripped export). The preferred
// path renders through the tokenizer's own Jinja template so this example
// cannot drift from the one qwen3_4b_example uses.
static std::string applyQwen3TemplateFallback(const std::string& user_text, bool thinking) {
    std::string prompt = "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    if (!thinking) prompt += "<think>\n\n</think>\n\n";
    return prompt;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const fs::path model_dir(args.model_dir);

    // Resolve the target engine's ctx-bins from metadata.json's geniex.engine
    // block (the multi-engine target/draft split has no generic
    // modelConfigFromDirectory equivalent, so this example resolves it here).
    const auto    meta_path = model_dir / "metadata.json";
    std::ifstream meta_f(meta_path);
    if (!meta_f) {
        std::cerr << "No metadata.json found in " << model_dir << "\n";
        return 1;
    }
    geniex::qwen3_eaglet::json root = geniex::qwen3_eaglet::json::parse(meta_f);
    if (!root.contains("geniex") || root["geniex"].value("dialog_type", "") != "eaglet") {
        std::cerr << "metadata.json's geniex.dialog_type is not \"eaglet\" in " << model_dir << "\n";
        return 1;
    }
    const auto& gx = root.at("geniex");

    geniex::ModelConfig model_cfg;
    for (const auto& eng : gx.at("engine")) {
        if (eng.value("role", "") != "target") continue;
        for (const auto& b : eng["model"]["binary"]["ctx-bins"])
            model_cfg.model_paths.push_back((model_dir / b.get<std::string>()).string());
    }
    if (model_cfg.model_paths.empty()) {
        std::cerr << "No target engine ctx-bins in metadata.json's geniex.engine.\n";
        return 1;
    }
    model_cfg.tokenizer_path =
        (model_dir / gx.value("tokenizer", geniex::qwen3_eaglet::json::object()).value("path", "tokenizer.json"))
            .string();
    // The chat template lives in tokenizer_config.json; load it when present so
    // --prompt renders through the bundle's own Jinja template.
    {
        const auto tok_cfg = model_dir / "tokenizer_config.json";
        if (fs::exists(tok_cfg)) model_cfg.tokenizer_config_path = tok_cfg.string();
    }
    model_cfg.embedding_path = (model_dir / gx.value("embedding", geniex::qwen3_eaglet::json::object())
                                                .value("lut-path", "quantized_embedding_table.bin"))
                                   .string();
    model_cfg.htp_config_path = (model_dir / gx.at("engine")[0]
                                                 .value("backend", geniex::qwen3_eaglet::json::object())
                                                 .value("extensions", "htp_backend_ext_config_mc.json"))
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

    auto tokenizer =
        geniex::Tokenizer::from_file(model_cfg.tokenizer_path, model_cfg.tokenizer_config_path.value_or(std::string{}));

    // --raw-prompt-file feeds an already-formatted prompt (chat markers baked in)
    // verbatim. With no flag the bundle's prompt.txt is treated the same way --
    // the repo's *_prompt.txt files ship fully templated, so re-applying the chat
    // template would double-wrap them and silently change what the model sees.
    // Only a bare --prompt (a raw user turn) is rendered through the template.
    std::vector<int32_t> prompt_tokens;
    auto                 read_file_verbatim = [](const fs::path& p, std::string& out) -> bool {
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        return true;
    };

    if (!args.raw_prompt_file.empty()) {
        std::string text;
        if (!read_file_verbatim(args.raw_prompt_file, text)) {
            std::cerr << "Cannot open raw prompt file: " << args.raw_prompt_file << "\n";
            return 1;
        }
        prompt_tokens = tokenizer->encode(text);
    } else if (!args.prompt.empty()) {
        // Prefer the bundle's own Jinja chat template; fall back to the minimal
        // literal only when the tokenizer_config.json carries none.
        std::string prompt_text;
        if (tokenizer->has_chat_template()) {
            geniex::ApplyChatTemplateOptions opts;
            opts.enable_thinking = args.thinking;
            try {
                prompt_text = tokenizer->apply_chat_template({{geniex::Role::User, args.prompt}}, opts);
            } catch (const std::exception& e) {
                std::cerr << "Chat-template error: " << e.what() << "\n";
                return 1;
            }
        } else {
            prompt_text = applyQwen3TemplateFallback(args.prompt, args.thinking);
        }
        prompt_tokens = tokenizer->encode(prompt_text);
    } else {
        std::string text;
        if (!read_file_verbatim(model_dir / "prompt.txt", text) || text.empty()) {
            std::cerr << "No prompt provided (use --prompt, --raw-prompt-file, or a prompt.txt in the bundle).\n";
            return 1;
        }
        prompt_tokens = tokenizer->encode(text);
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

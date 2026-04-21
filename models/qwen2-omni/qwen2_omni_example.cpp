#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "qwen2_omni.h"
#include "vlm/vlm_types.h"
#include "types.h"
#include "geniex-proc/qwen2vl.h"
#include "geniex-proc/tokenizer.h"

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

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    int32_t     max_tokens      = 512;
    bool        verbose         = false;
    bool        enable_thinking = false;
    std::string image_path;
    std::string audio_path;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --image <path>     Image file for the first turn\n"
              << "  --audio <path>     Audio file for the first turn\n"
              << "  --max-tokens <n>   Max tokens to generate (default 512)\n"
              << "  --thinking         Enable thinking mode\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--image")      args.image_path      = next();
        else if (a == "--audio")      args.audio_path      = next();
        else if (a == "--max-tokens") args.max_tokens      = std::stoi(next());
        else if (a == "--thinking")   args.enable_thinking = true;
        else if (a == "--verbose")    args.verbose         = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

// ── BatchFeatures → geniex types ─────────────────────────────────────────────

static geniex::PixelData toPixelData(const geniex::BatchFeatures& bf) {
    geniex::PixelData pd;
    if (bf.image_grid_thw.dimension() == 0 || bf.image_grid_thw.shape()[0] == 0) return pd;

    pd.pixel_values.assign(bf.pixel_values.cbegin(), bf.pixel_values.cend());

    const size_t n = bf.image_grid_thw.shape()[0];
    pd.image_grid_thw.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pd.image_grid_thw[i] = {
            static_cast<int32_t>(bf.image_grid_thw(i, 0)),
            static_cast<int32_t>(bf.image_grid_thw(i, 1)),
            static_cast<int32_t>(bf.image_grid_thw(i, 2)),
        };
    }
    return pd;
}

// static geniex::AudioData toAudioData(const geniex::BatchFeatures& bf) {
//     geniex::AudioData ad;
//     if (bf.audio_features.dimension() < 3 || bf.audio_features.shape()[2] == 0) return ad;

//     // audio_features shape: [1, num_mel_bins, T_padded]
//     // audio_attention_mask shape: [1, T_padded]
//     // AudioData.audio_features: mel-major flat [num_mel_bins * num_frames],
//     //   indexed as data[m * num_frames + t].
//     // Padding is stripped: only copy the valid frames.

//     const int32_t num_mel  = static_cast<int32_t>(bf.audio_features.shape()[1]);
//     const int32_t T_padded = static_cast<int32_t>(bf.audio_features.shape()[2]);

//     int32_t valid_frames = 0;
//     for (int32_t t = 0; t < T_padded; ++t)
//         valid_frames += bf.audio_attention_mask(0, t);

//     ad.num_mel_bins = num_mel;
//     ad.num_frames   = valid_frames;
//     ad.audio_features.resize(static_cast<size_t>(num_mel) * valid_frames);
//     for (int32_t m = 0; m < num_mel; ++m)
//         for (int32_t t = 0; t < valid_frames; ++t)
//             ad.audio_features[m * valid_frames + t] = bf.audio_features(0, m, t);

//     ad.audio_attention_mask.assign(valid_frames, 1);
//     return ad;
// }

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir = std::filesystem::current_path() / "modelfiles" / "qwen2-omni";

    // All QNN runtime paths are left as std::nullopt → auto-detected from
    // htp-files/ installed alongside geniex_core.
    geniex::QnnRuntimeConfig runtime_cfg;

    // ── Model configs ─────────────────────────────────────────────────────────

    geniex::qwen2_omni::Qwen2OmniConfig config;

    config.llm_config.model_paths = {
        (model_dir / "llm" / "ar128-ar1-cl4096" / "weight_sharing_model_1_of_2.serialized.bin").string(),
        (model_dir / "llm" / "ar128-ar1-cl4096" / "weight_sharing_model_2_of_2.serialized.bin").string(),
    };
    config.llm_config.tokenizer_path  = (model_dir / "tokenizer.json").string();
    config.llm_config.embedding_path  = (model_dir / "embed_tokens.npy").string();

    config.vision_config.model_paths = {
        (model_dir / "vit" / "patch_embed.serialized.bin").string(),
        (model_dir / "vit" / "vit.serialized.bin").string(),
    };

    // Audio encoder disabled — audio processing not yet supported.
    // config.audio_config.model_paths = {
    //     (model_dir / "audio_encoder" / "audio_encoder_helper0.serialized.bin").string(),
    //     (model_dir / "audio_encoder" / "audio_encoder.serialized.bin").string(),
    //     (model_dir / "audio_encoder" / "audio_encoder_helper1.serialized.bin").string(),
    // };

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

    // ── Load model ────────────────────────────────────────────────────────────

    std::cout << "\033[1;36mLoading model...\033[0m\n";
    auto model = geniex::qwen2_omni::makeModel(runtime_cfg, config);
    if (!model) {
        std::cerr << "Failed to initialize model.\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    // ── Load tokenizer and processor ──────────────────────────────────────────

    auto processor = geniex::qwen2vl::Qwen2VLProcessor::create(config.llm_config.tokenizer_path);

    // ── Chat loop ─────────────────────────────────────────────────────────────
    //
    // Incremental KV cache strategy: only the tokens for the *current turn* are
    // fed to generate() each round. Previous turns are already in the KV cache.
    //
    // Turn 1:  process([system, user_msg], add_generation_prompt=true)
    //          → full chat template + image preprocessing via processor
    //
    // Turn N+: close the previous assistant turn (<|im_end|>\n) then open the
    //          new user turn, tokenized directly — no image preprocessing needed.
    //          The prefix fed to generate() is:
    //            <|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n

    bool first_turn = true;

    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;

        const bool saved_first_turn = first_turn;

        // ── Preprocess + Generate (all inside try so errors are reported) ──────

        const auto t_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_first_token;
        bool got_first_token = false;

        std::cout << "\033[33m";
        std::vector<int32_t> output_tokens;
        try {
            std::vector<int32_t> prompt_tokens;
            geniex::VLMInput vlm_input;

            if (first_turn) {
                // Full chat template + image preprocessing for the first turn.
                geniex::ChatMessage system_msg{"system", "You are a helpful assistant."};
                geniex::ChatMessage user_msg{"user", input};
                if (!args.image_path.empty())
                    user_msg.mm_content_paths.push_back(args.image_path);

                geniex::BatchFeatures bf = processor->process({{system_msg, user_msg}, /*add_generation_prompt=*/true});
                vlm_input.pixel_data = toPixelData(bf);
                prompt_tokens.assign(bf.input_ids.cbegin(), bf.input_ids.cend());
                first_turn = false;
            } else {
                // Incremental turn: close the previous assistant turn, open the new user turn.
                // The KV cache already holds all previous context.
                const std::string turn_text =
                    "<|im_end|>\n"
                    "<|im_start|>user\n" + input + "<|im_end|>\n"
                    "<|im_start|>assistant\n";
                prompt_tokens = processor->tokenizer().encode(turn_text, /*add_special_tokens=*/false);
            }

            output_tokens = model->generate(
                prompt_tokens,
                vlm_input,
                gen_cfg,
                [&](int32_t tok) {
                    if (!got_first_token) {
                        t_first_token   = std::chrono::high_resolution_clock::now();
                        got_first_token = true;
                    }
                    std::cout << processor->tokenizer().decode({tok}) << std::flush;
                });
        } catch (const std::exception& e) {
            std::cout << "\033[0m\n" << std::flush;
            std::cerr << "Error: " << e.what() << "\n" << std::flush;
            model->resetKVCache();
            first_turn = saved_first_turn;
            continue;
        } catch (...) {
            std::cout << "\033[0m\n" << std::flush;
            std::cerr << "Error: unknown exception\n" << std::flush;
            model->resetKVCache();
            first_turn = saved_first_turn;
            continue;
        }
        std::cout << "\033[0m\n";

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (got_first_token) {
            const double ttft_ms   = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
            const double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
            const size_t decode_tok = output_tokens.size() > 1 ? output_tokens.size() - 1 : 0;
            const double tps       = decode_ms > 0.0 ? decode_tok / (decode_ms / 1000.0) : 0.0;

            if (args.verbose) {
                std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                          << "Generated tokens : " << output_tokens.size() << "\n"
                          << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms  << " ms\n"
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

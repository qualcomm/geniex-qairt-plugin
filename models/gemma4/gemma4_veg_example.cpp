// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// gemma4_veg — run ONLY the Gemma4 Visual Embedding Generator (VEG) context
// binary on the HTP, no decoder involved.
//
//   image -> Gemma4Processor -> pixel_values [1,2520,768] + image_position_ids [1,2520,2]
//         -> Gemma4VisionEncoder (VEG graph on v73 HTP) -> vision_embedding [1,256,1536]
//
// This is the vision half of gemma4_vlm_example.cpp, split out so the VEG can be
// exercised on-device without the multi-GB decoder bundle. It prints the graph's
// declared I/O, runs one forward pass, reports summary statistics of the output
// embedding (and its timing), and can dump the tensors / compare against a
// reference vision_embedding.bin for cosine similarity.
//
//   gemma4_veg --veg-dir <dir with the .serialized.bin> --image <path>
//              [--tokenizer <tokenizer.json>]   (optional; preprocessing is tokenizer-free)
//              [--dump-dir <dir>]               (writes pixel_values/position_ids/vision_embedding)
//              [--ref <vision_embedding.bin>]   (cosine vs a golden fp32 embedding)
//              [--iters <n>]                    (repeat the forward pass for timing)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gemma4/gemma4_vision.h"
#include "geniex-proc/gemma4.h"
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

namespace {

struct Args {
    fs::path veg_dir;  // dir holding the VEG context binary (.bin)
    fs::path image;
    fs::path tokenizer;  // optional; process_images() does not need it
    fs::path dump_dir;   // optional
    fs::path ref;        // optional golden vision_embedding.bin (fp32)
    int      iters = 1;  // repeat forward passes (timing)
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --veg-dir <path>    Directory containing the VEG .serialized.bin (required)\n"
              << "  --image <path>      Input image (required)\n"
              << "  --tokenizer <path>  tokenizer.json (optional; preprocessing is tokenizer-free)\n"
              << "  --dump-dir <path>   Write pixel_values/image_position_ids/vision_embedding there\n"
              << "  --ref <path>        Golden fp32 vision_embedding.bin to cosine-compare against\n"
              << "  --iters <n>         Repeat the forward pass n times for timing (default 1)\n";
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (s == "--veg-dir")
            a.veg_dir = next();
        else if (s == "--image")
            a.image = next();
        else if (s == "--tokenizer")
            a.tokenizer = next();
        else if (s == "--dump-dir")
            a.dump_dir = next();
        else if (s == "--ref")
            a.ref = next();
        else if (s == "--iters")
            a.iters = std::max(1, std::stoi(next()));
        else if (s == "--help" || s == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            return false;
        }
    }
    if (a.veg_dir.empty()) {
        std::cerr << "--veg-dir is required\n";
        return false;
    }
    if (a.image.empty()) {
        std::cerr << "--image is required\n";
        return false;
    }
    return true;
}

template <typename T>
void writeBin(const fs::path& p, const T* data, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(T)));
}

std::vector<float> readFloatBin(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    const auto n = static_cast<size_t>(f.tellg()) / sizeof(float);
    f.seekg(0);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return v;
}

// The VEG bundle is just a context binary; build a minimal ModelConfig for it.
geniex::ModelConfig vegConfig(const fs::path& veg_dir) {
    geniex::ModelConfig cfg;
    for (const auto& e : fs::directory_iterator(veg_dir)) {
        if (e.is_regular_file() && e.path().extension() == ".bin") {
            cfg.model_paths.push_back(e.path().string());
        }
    }
    if (cfg.model_paths.empty()) {
        throw std::runtime_error("no .bin context binary found in " + veg_dir.string());
    }
    std::sort(cfg.model_paths.begin(), cfg.model_paths.end());
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    // ── 1. Preprocess the image ───────────────────────────────────────────────
    // process_images() is tokenizer-free (see gemma4_prep_check), so an empty
    // tokenizer path is fine unless the caller supplied one.
    geniex::gemma4::Gemma4Config                     proc_cfg;
    std::unique_ptr<geniex::gemma4::Gemma4Processor> proc;
    try {
        proc = geniex::gemma4::Gemma4Processor::create(args.tokenizer.string(), /*tokenizer_config_path=*/"", proc_cfg);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create processor: " << e.what() << "\n";
        return 1;
    }

    geniex::BatchFeatures feats;
    try {
        feats = proc->process_images({args.image.string()});
    } catch (const std::exception& e) {
        std::cerr << "Image preprocessing failed: " << e.what() << "\n";
        return 1;
    }
    const int n_soft = feats.num_soft_tokens_per_image[0];
    std::cout << "\033[1;36mimage\033[0m " << args.image.string() << " -> " << n_soft << " soft tokens\n";

    std::vector<float>   pixel_values(feats.pixel_values.begin(), feats.pixel_values.end());
    std::vector<int32_t> position_ids(feats.image_position_ids.begin(), feats.image_position_ids.end());
    std::cout << "pixel_values " << pixel_values.size() << " floats, image_position_ids " << position_ids.size()
              << " ints\n";

    // ── 2. Load the VEG context binary ────────────────────────────────────────
    geniex::QnnRuntimeConfig            runtime_cfg;
    geniex::gemma4::Gemma4VisionEncoder veg;
    if (!veg.initialize(runtime_cfg, vegConfig(args.veg_dir))) {
        std::cerr << "Failed to initialize the VEG. See logs.\n";
        return 1;
    }
    if (static_cast<int>(veg.numSoftTokens()) != n_soft) {
        std::cerr << "VEG emits " << veg.numSoftTokens() << " soft tokens but the image implies " << n_soft
                  << ". The processor's force_square_size must match the exported geometry.\n";
        return 1;
    }

    // ── 3. Run the vision encoder ─────────────────────────────────────────────
    std::vector<float> vision_embeds;
    double             total_ms = 0.0, best_ms = 1e30;
    for (int it = 0; it < args.iters; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            vision_embeds = veg.encode(pixel_values, position_ids);
        } catch (const std::exception& e) {
            std::cerr << "VEG execution failed: " << e.what() << "\n";
            return 1;
        }
        const auto   t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        best_ms = std::min(best_ms, ms);
    }
    std::cout << "\033[1;32mvision_embedding\033[0m [" << veg.numSoftTokens() << "," << veg.hiddenSize()
              << "] = " << vision_embeds.size() << " floats\n";

    // ── 4. Summary statistics of the output ───────────────────────────────────
    double mn = 1e30, mx = -1e30, sum = 0.0, sumsq = 0.0;
    size_t n_nan = 0;
    for (float v : vision_embeds) {
        if (std::isnan(v) || std::isinf(v)) {
            ++n_nan;
            continue;
        }
        mn = std::min(mn, static_cast<double>(v));
        mx = std::max(mx, static_cast<double>(v));
        sum += v;
        sumsq += static_cast<double>(v) * v;
    }
    const double n    = static_cast<double>(vision_embeds.size());
    const double mean = sum / n;
    const double var  = sumsq / n - mean * mean;
    std::cout << std::fixed << std::setprecision(6) << "\033[1;36m=== vision_embedding stats ===\033[0m\n"
              << "  min " << mn << "  max " << mx << "  mean " << mean << "  std " << std::sqrt(std::max(0.0, var))
              << "  nan/inf " << n_nan << "\n"
              << "  first 8: ";
    for (size_t i = 0; i < 8 && i < vision_embeds.size(); ++i) std::cout << vision_embeds[i] << " ";
    std::cout << "\n";

    // ── 5. Timing ─────────────────────────────────────────────────────────────
    std::cout << "\033[1;36m=== timing ===\033[0m\n"
              << "  iters " << args.iters << "  best " << std::setprecision(2) << best_ms << " ms  avg "
              << (total_ms / args.iters) << " ms\n";

    // ── 6. Optional cosine vs a golden fp32 embedding ─────────────────────────
    if (!args.ref.empty()) {
        try {
            const auto ref = readFloatBin(args.ref);
            if (ref.size() != vision_embeds.size()) {
                std::cerr << "ref size " << ref.size() << " != output size " << vision_embeds.size() << "\n";
            } else {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    dot += static_cast<double>(ref[i]) * vision_embeds[i];
                    na += static_cast<double>(ref[i]) * ref[i];
                    nb += static_cast<double>(vision_embeds[i]) * vision_embeds[i];
                }
                const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
                std::cout << "\033[1;32m=== cosine vs ref ===\033[0m " << std::setprecision(6) << cos << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "ref compare failed: " << e.what() << "\n";
        }
    }

    // ── 7. Optional dump ──────────────────────────────────────────────────────
    if (!args.dump_dir.empty()) {
        fs::create_directories(args.dump_dir);
        writeBin(args.dump_dir / "pixel_values.bin", pixel_values.data(), pixel_values.size());
        writeBin(args.dump_dir / "image_position_ids.bin", position_ids.data(), position_ids.size());
        writeBin(args.dump_dir / "vision_embedding.bin", vision_embeds.data(), vision_embeds.size());
        std::cout << "dumped tensors to " << args.dump_dir.string() << "\n";
    }

    return 0;
}

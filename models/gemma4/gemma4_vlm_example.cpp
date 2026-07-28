// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 VLM (image + text) inference.
//
// Pipeline:
//   image -> Gemma4Processor            -> pixel_values [1,2520,768] + position ids
//         -> VEG graph                  -> vision_embedding [1,256,1536]
//         -> splice into inputs_embeds at the 256 image-token positions
//         -> Gemma4 decoder prefill/decode (unchanged from the text path)
//
// The prompt is built exactly as Gemma4Processor.replace_image_token() does:
//   <bos><|turn>user\n<|image>{<|image|> x 256}<image|>{PROMPT}<turn|>\n<|turn>model\n
//
// --veg-only stops after the vision half: it runs ONLY the Visual Embedding
// Generator on the HTP, with no decoder, so the VEG can be exercised on-device
// without the multi-GB decoder bundle. Preprocessing is tokenizer-free in that
// mode, so --model-dir is not required. Combined with --stats / --ref / --iters
// this is the vision bring-up path:
//
//   gemma4_vlm --veg-only --veg-dir <dir> --image <path> --stats
//              [--ref vision_embedding.bin]   (cosine vs a golden fp32 embedding)
//              [--iters <n>]                  (repeat the forward pass for timing)
//
// --stats and --ref also work in full VLM mode, so an end-to-end run can check
// the embedding numerically while still generating text.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gemma4/gemma4.h"
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
    fs::path    model_dir;  // LLM decoder bundle; not needed with --veg-only
    fs::path    veg_dir;    // dir holding the VEG context binary
    fs::path    image;
    std::string prompt     = "describe this image";
    int32_t     max_tokens = 200;
    bool        verbose    = false;
    fs::path    dump_dir;  // optional: write vision_embedding for comparison
    // Vision-side diagnostics.
    bool     veg_only = false;  // stop after the VEG; no decoder
    bool     stats    = false;  // summary statistics of vision_embedding
    fs::path ref;               // optional golden vision_embedding.bin (fp32)
    int      iters = 1;         // repeat the VEG forward pass (timing)
    // Sampling. Defaults to greedy so runs are comparable against the Genie
    // reference, which also samples at temp 0 / top-k 1.
    float    temperature = 0.0f;
    int32_t  top_k       = 1;
    float    top_p       = 1.0f;
    uint32_t seed        = 42;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model-dir <path>  Gemma4 decoder bundle (default ./modelfiles/gemma4_e2b;\n"
              << "                      unused with --veg-only)\n"
              << "  --veg-dir <path>    Directory containing the VEG context binary\n"
              << "  --image <path>      Input image\n"
              << "  --prompt <text>     Text prompt (default \"describe this image\")\n"
              << "  --max-tokens <n>    Default 200\n"
              << "  --dump-dir <path>   Write vision_embedding.bin / pixel_values.bin there\n"
              << "  --temp <f>          Sampling temperature (default 0 = greedy)\n"
              << "  --top-k <n>         Default 1\n"
              << "  --top-p <f>         Default 1.0\n"
              << "  --seed <n>          Default 42\n"
              << "  --verbose\n"
              << "\n Vision-only / diagnostics:\n"
              << "  --veg-only          Run just the VEG and exit; no decoder is loaded\n"
              << "  --stats             Print min/max/mean/std + nan/inf of vision_embedding\n"
              << "  --ref <path>        Golden fp32 vision_embedding.bin to cosine-compare against\n"
              << "  --iters <n>         Repeat the VEG forward pass n times for timing\n"
              << "                      (default 1; requires --veg-only above 1)\n";
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (s == "--model-dir")
            a.model_dir = next();
        else if (s == "--veg-dir")
            a.veg_dir = next();
        else if (s == "--image")
            a.image = next();
        else if (s == "--prompt")
            a.prompt = next();
        else if (s == "--max-tokens")
            a.max_tokens = std::stoi(next());
        else if (s == "--dump-dir")
            a.dump_dir = next();
        else if (s == "--veg-only")
            a.veg_only = true;
        else if (s == "--stats")
            a.stats = true;
        else if (s == "--ref")
            a.ref = next();
        else if (s == "--iters")
            a.iters = std::max(1, std::stoi(next()));
        else if (s == "--temp")
            a.temperature = std::stof(next());
        else if (s == "--top-k")
            a.top_k = std::stoi(next());
        else if (s == "--top-p")
            a.top_p = std::stof(next());
        else if (s == "--seed")
            a.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (s == "--verbose")
            a.verbose = true;
        else if (s == "--help" || s == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            return false;
        }
    }
    // --veg-only never touches the decoder, so it needs no bundle. Defaulting
    // model_dir anyway would make the tokenizer paths below point at a bundle
    // that may not be on disk.
    if (a.model_dir.empty() && !a.veg_only) a.model_dir = fs::current_path() / "modelfiles" / "gemma4_e2b";
    if (a.image.empty()) {
        std::cerr << "--image is required\n";
        return false;
    }
    if (a.veg_dir.empty()) {
        std::cerr << "--veg-dir is required\n";
        return false;
    }
    // Re-running the encoder would discard all but the last embedding, and the
    // decoder path needs exactly one, so only time it when nothing follows.
    if (a.iters > 1 && !a.veg_only) {
        std::cerr << "--iters above 1 requires --veg-only\n";
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

void printEmbeddingStats(const std::vector<float>& v) {
    double mn = 1e30, mx = -1e30, sum = 0.0, sumsq = 0.0;
    size_t n_nan = 0;
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) {
            ++n_nan;
            continue;
        }
        mn = std::min(mn, static_cast<double>(x));
        mx = std::max(mx, static_cast<double>(x));
        sum += x;
        sumsq += static_cast<double>(x) * x;
    }
    const double n    = static_cast<double>(v.size());
    const double mean = sum / n;
    const double var  = sumsq / n - mean * mean;
    std::cout << std::fixed << std::setprecision(6) << "\033[1;36m=== vision_embedding stats ===\033[0m\n"
              << "  min " << mn << "  max " << mx << "  mean " << mean << "  std " << std::sqrt(std::max(0.0, var))
              << "  nan/inf " << n_nan << "\n"
              << "  first 8: ";
    for (size_t i = 0; i < 8 && i < v.size(); ++i) std::cout << v[i] << " ";
    std::cout << "\n" << std::defaultfloat;
}

// Cosine similarity against a golden fp32 embedding dumped from the reference
// implementation. Reports rather than throws: a mismatch is a diagnostic, not a
// reason to abandon a decode that would otherwise run.
void compareWithRef(const fs::path& ref_path, const std::vector<float>& out) {
    try {
        const auto ref = readFloatBin(ref_path);
        if (ref.size() != out.size()) {
            std::cerr << "ref size " << ref.size() << " != output size " << out.size() << "\n";
            return;
        }
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            dot += static_cast<double>(ref[i]) * out[i];
            na += static_cast<double>(ref[i]) * ref[i];
            nb += static_cast<double>(out[i]) * out[i];
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        std::cout << "\033[1;32m=== cosine vs ref ===\033[0m " << std::fixed << std::setprecision(6) << cos << "\n"
                  << std::defaultfloat;
    } catch (const std::exception& e) {
        std::cerr << "ref compare failed: " << e.what() << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    geniex::QnnRuntimeConfig runtime_cfg;

    // ── 1. Preprocess the image ───────────────────────────────────────────────
    // process_images() is tokenizer-free (see gemma4_prep_check), so --veg-only
    // can pass empty tokenizer paths and skip the decoder bundle entirely.
    geniex::gemma4::Gemma4Config                     proc_cfg;
    std::unique_ptr<geniex::gemma4::Gemma4Processor> proc;
    try {
        proc = args.veg_only ? geniex::gemma4::Gemma4Processor::create("", /*tokenizer_config_path=*/"", proc_cfg)
                             : geniex::gemma4::Gemma4Processor::create((args.model_dir / "tokenizer.json").string(),
                                   (args.model_dir / "tokenizer_config.json").string(),
                                   proc_cfg);
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
    if (args.verbose || args.veg_only) {
        std::cout << "pixel_values " << pixel_values.size() << " floats, image_position_ids " << position_ids.size()
                  << " ints\n";
    }

    // ── 2. Run the vision encoder ─────────────────────────────────────────────
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
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total_ms += ms;
        best_ms = std::min(best_ms, ms);
    }
    std::cout << "\033[1;32mvision_embedding\033[0m [" << veg.numSoftTokens() << "," << veg.hiddenSize() << "]\n";

    if (args.stats) printEmbeddingStats(vision_embeds);
    if (args.iters > 1 || args.verbose) {
        std::cout << "\033[1;36m=== VEG timing ===\033[0m iters " << args.iters << "  best " << std::fixed
                  << std::setprecision(2) << best_ms << " ms  avg " << (total_ms / args.iters) << " ms\n"
                  << std::defaultfloat;
    }
    if (!args.ref.empty()) compareWithRef(args.ref, vision_embeds);

    if (!args.dump_dir.empty()) {
        fs::create_directories(args.dump_dir);
        writeBin(args.dump_dir / "vision_embedding.bin", vision_embeds.data(), vision_embeds.size());
        writeBin(args.dump_dir / "pixel_values.bin", pixel_values.data(), pixel_values.size());
        writeBin(args.dump_dir / "image_position_ids.bin", position_ids.data(), position_ids.size());
        std::cout << "dumped vision tensors to " << args.dump_dir.string() << "\n";
    }

    // Vision-only bring-up stops here: everything below needs the decoder bundle.
    if (args.veg_only) return 0;

    // ── 3. Build the prompt (image span + text) ───────────────────────────────
    const std::string marker  = proc->image_marker();
    const std::string content = marker + args.prompt;
    std::string       text;
    try {
        text = proc->apply_chat_template({{geniex::Role::User, content}}, /*add_generation_prompt=*/true);
    } catch (const std::exception& e) {
        std::cerr << "Chat-template error: " << e.what() << "\n";
        return 1;
    }
    const auto batch = proc->process(text, {args.image.string()});

    // Locate the contiguous image-token run; its start is where the vision rows go.
    const int32_t image_id = proc->tokenizer().encode(proc_cfg.image_token, false).front();
    const auto&   ids      = batch.input_ids;
    const auto    first    = std::find(ids.begin(), ids.end(), image_id);
    if (first == ids.end()) {
        std::cerr << "No image token in the rendered prompt.\n";
        return 1;
    }
    const size_t img_start = static_cast<size_t>(std::distance(ids.begin(), first));
    size_t       img_count = 0;
    while (img_start + img_count < ids.size() && ids[img_start + img_count] == image_id) ++img_count;
    if (static_cast<int>(img_count) != n_soft) {
        std::cerr << "prompt has " << img_count << " image tokens but the encoder produced " << n_soft << "\n";
        return 1;
    }
    std::cout << "prompt " << ids.size() << " tokens; image slots [" << img_start << ".." << (img_start + img_count - 1)
              << "]\n";

    if (!args.dump_dir.empty()) {
        writeBin(args.dump_dir / "input_ids.bin", ids.data(), ids.size());
    }

    // ── 4. Decoder ────────────────────────────────────────────────────────────
    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(args.model_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read decoder bundle: " << e.what() << "\n";
        return 1;
    }

    const auto bundle = geniex::bundleDirOf(model_cfg);
    auto       gc     = geniex::parseGenieConfig(bundle);
    const auto cfg    = geniex::gemma4::withEmbeddingPath(model_cfg, gc, bundle);

    auto model = geniex::gemma4::makeModel(cfg);
    if (!model.initialize(runtime_cfg, cfg)) {
        std::cerr << "Failed to initialize the decoder. See logs.\n";
        return 1;
    }
    model.setVisionEmbeddings(img_start, vision_embeds);

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;
    // temp 0 / top-k 1 is greedy; the sampler chain is only worth engaging above that.
    gen_cfg.enable_sampling = args.temperature > 0.0f || args.top_k != 1 || args.top_p < 1.0f;
    gen_cfg.temperature     = args.temperature;
    gen_cfg.top_k           = args.top_k;
    gen_cfg.top_p           = args.top_p;
    gen_cfg.seed            = args.seed;

    const auto t0        = std::chrono::steady_clock::now();
    auto       first_tok = t0;
    bool       got_first = false;

    std::cout << "\n\033[33m";
    const auto out_tokens = model.generate(ids, gen_cfg, [&](int32_t tok) {
        if (!got_first) {
            first_tok = std::chrono::steady_clock::now();
            got_first = true;
        }
        std::cout << proc->tokenizer().decode_token(tok) << std::flush;
        return true;
    });
    const auto t1         = std::chrono::steady_clock::now();
    std::cout << "\033[0m\n";

    if (args.verbose) {
        const double ttft_ms = std::chrono::duration<double, std::milli>(first_tok - t0).count();
        const double dec_ms  = std::chrono::duration<double, std::milli>(t1 - first_tok).count();
        const double tps     = out_tokens.size() > 1 ? (out_tokens.size() - 1) * 1000.0 / dec_ms : 0.0;
        std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                  << "Prompt tokens    : " << ids.size() << "\n"
                  << "Generated tokens : " << out_tokens.size() << "\n"
                  << "TTFT             : " << std::fixed << std::setprecision(1) << ttft_ms << " ms\n"
                  << "Decode speed     : " << std::setprecision(2) << tps << " tokens/s\n";
    }
    return 0;
}

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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    fs::path    model_dir;   // LLM decoder bundle
    fs::path    veg_dir;     // dir holding the VEG context binary
    fs::path    image;
    std::string prompt     = "describe this image";
    int32_t     max_tokens = 200;
    bool        verbose    = false;
    fs::path    dump_dir;    // optional: write vision_embedding for comparison
    // Sampling. Defaults to greedy so runs are comparable against the Genie
    // reference, which also samples at temp 0 / top-k 1.
    float    temperature = 0.0f;
    int32_t  top_k       = 1;
    float    top_p       = 1.0f;
    uint32_t seed        = 42;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model-dir <path>  Gemma4 decoder bundle (default ./modelfiles/gemma4_e2b)\n"
              << "  --veg-dir <path>    Directory containing the VEG context binary\n"
              << "  --image <path>      Input image\n"
              << "  --prompt <text>     Text prompt (default \"describe this image\")\n"
              << "  --max-tokens <n>    Default 200\n"
              << "  --dump-dir <path>   Write vision_embedding.bin / pixel_values.bin there\n"
              << "  --temp <f>          Sampling temperature (default 0 = greedy)\n"
              << "  --top-k <n>         Default 1\n"
              << "  --top-p <f>         Default 1.0\n"
              << "  --seed <n>          Default 42\n"
              << "  --verbose\n";
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (s == "--model-dir") a.model_dir = next();
        else if (s == "--veg-dir") a.veg_dir = next();
        else if (s == "--image") a.image = next();
        else if (s == "--prompt") a.prompt = next();
        else if (s == "--max-tokens") a.max_tokens = std::stoi(next());
        else if (s == "--dump-dir") a.dump_dir = next();
        else if (s == "--temp") a.temperature = std::stof(next());
        else if (s == "--top-k") a.top_k = std::stoi(next());
        else if (s == "--top-p") a.top_p = std::stof(next());
        else if (s == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (s == "--verbose") a.verbose = true;
        else if (s == "--help" || s == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << s << "\n"; return false; }
    }
    if (a.model_dir.empty()) a.model_dir = fs::current_path() / "modelfiles" / "gemma4_e2b";
    if (a.image.empty()) { std::cerr << "--image is required\n"; return false; }
    if (a.veg_dir.empty()) { std::cerr << "--veg-dir is required\n"; return false; }
    return true;
}

void writeBin(const fs::path& p, const void* data, size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
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

    geniex::QnnRuntimeConfig runtime_cfg;

    // ── 1. Preprocess the image ───────────────────────────────────────────────
    geniex::gemma4::Gemma4Config proc_cfg;
    auto proc = geniex::gemma4::Gemma4Processor::create(
        (args.model_dir / "tokenizer.json").string(), (args.model_dir / "tokenizer_config.json").string(), proc_cfg);

    const auto feats = proc->process_images({args.image.string()});
    const int  n_soft = feats.num_soft_tokens_per_image[0];
    std::cout << "\033[1;36mimage\033[0m " << args.image.string() << " -> " << n_soft << " soft tokens\n";

    std::vector<float>   pixel_values(feats.pixel_values.begin(), feats.pixel_values.end());
    std::vector<int32_t> position_ids(feats.image_position_ids.begin(), feats.image_position_ids.end());

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

    const auto vision_embeds = veg.encode(pixel_values, position_ids);
    std::cout << "\033[1;32mvision_embedding\033[0m [" << veg.numSoftTokens() << "," << veg.hiddenSize() << "]\n";

    if (!args.dump_dir.empty()) {
        fs::create_directories(args.dump_dir);
        writeBin(args.dump_dir / "vision_embedding.bin", vision_embeds.data(), vision_embeds.size() * sizeof(float));
        writeBin(args.dump_dir / "pixel_values.bin", pixel_values.data(), pixel_values.size() * sizeof(float));
        writeBin(args.dump_dir / "image_position_ids.bin", position_ids.data(), position_ids.size() * sizeof(int32_t));
        std::cout << "dumped vision tensors to " << args.dump_dir.string() << "\n";
    }

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
    std::cout << "prompt " << ids.size() << " tokens; image slots [" << img_start << ".."
              << (img_start + img_count - 1) << "]\n";

    if (!args.dump_dir.empty()) {
        fs::create_directories(args.dump_dir);
        writeBin(args.dump_dir / "input_ids.bin", ids.data(), ids.size() * sizeof(ids[0]));
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
    const auto t1 = std::chrono::steady_clock::now();
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

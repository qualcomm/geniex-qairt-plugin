// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// gemma4_prep_check — numerical check of the C++ Gemma4 vision preprocessing
// against the transformers Gemma4ImageProcessorPil reference.
//
// Generate the reference with driver/prep_vision.py, then:
//   gemma4_prep_check --image images.jpg --ref-dir <dir with pixel_values.bin,
//                                                   image_position_ids.bin>
// It also writes its own tensors so they can be fed to the VEG graph directly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "geniex-proc/gemma4.h"

namespace {

template <typename T>
std::vector<T> readBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<size_t>(f.tellg()) / sizeof(T);
    f.seekg(0);
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return v;
}

template <typename T>
void writeBin(const std::string& path, const T* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
    std::string image, ref_dir, out_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--image") image = next();
        else if (a == "--ref-dir") ref_dir = next();
        else if (a == "--out-dir") out_dir = next();
        else { std::cerr << "unknown arg " << a << "\n"; return 1; }
    }
    if (image.empty()) { std::cerr << "usage: --image <path> [--ref-dir <dir>] [--out-dir <dir>]\n"; return 1; }

    geniex::gemma4::Gemma4Config cfg;
    std::cout << "config: patch=" << cfg.patch_size << " pooling=" << cfg.pooling_kernel_size
              << " max_soft_tokens=" << cfg.max_soft_tokens << " -> max_patches=" << cfg.max_patches()
              << " force_square=" << cfg.force_square_size << "\n";

    // Image-only path needs no tokenizer file, but create() loads one; the
    // preprocessing itself is independent of it.
    auto proc = geniex::gemma4::Gemma4Processor::create(
        /*tokenizer_path=*/"", /*tokenizer_config_path=*/"", cfg);

    const auto feats = proc->process_images({image});
    const auto& pv   = feats.pixel_values;
    const auto& pid  = feats.image_position_ids;

    std::cout << "pixel_values       [" << pv.shape(0) << "," << pv.shape(1) << "," << pv.shape(2) << "]\n"
              << "image_position_ids [" << pid.shape(0) << "," << pid.shape(1) << "," << pid.shape(2) << "]\n"
              << "num_soft_tokens    " << feats.num_soft_tokens_per_image[0] << "\n";

    if (!out_dir.empty()) {
        writeBin(out_dir + "/pixel_values.bin", pv.data(), pv.size());
        writeBin(out_dir + "/image_position_ids.bin", pid.data(), pid.size());
        std::cout << "wrote tensors to " << out_dir << "\n";
    }

    if (ref_dir.empty()) return 0;

    const auto ref_pv  = readBin<float>(ref_dir + "/pixel_values.bin");
    const auto ref_pid = readBin<int32_t>(ref_dir + "/image_position_ids.bin");

    if (ref_pv.size() != pv.size() || ref_pid.size() != pid.size()) {
        std::cerr << "SIZE MISMATCH: pixel_values " << pv.size() << " vs ref " << ref_pv.size() << ", position_ids "
                  << pid.size() << " vs ref " << ref_pid.size() << "\n";
        return 2;
    }

    size_t pid_bad = 0;
    for (size_t i = 0; i < ref_pid.size(); ++i)
        if (ref_pid[i] != pid.data()[i]) ++pid_bad;

    double max_abs = 0.0, sum_abs = 0.0;
    size_t over_1_255 = 0;
    for (size_t i = 0; i < ref_pv.size(); ++i) {
        const double d = std::fabs(static_cast<double>(ref_pv[i]) - static_cast<double>(pv.data()[i]));
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        if (d > 1.0 / 255.0) ++over_1_255;
    }
    const double mean_abs = sum_abs / static_cast<double>(ref_pv.size());

    std::cout << std::fixed << std::setprecision(6) << "\n=== vs reference ===\n"
              << "image_position_ids mismatches : " << pid_bad << " / " << ref_pid.size() << "\n"
              << "pixel_values max abs diff     : " << max_abs << "  (" << max_abs * 255.0 << " /255)\n"
              << "pixel_values mean abs diff    : " << mean_abs << "  (" << mean_abs * 255.0 << " /255)\n"
              << "elements differing > 1/255    : " << over_1_255 << " / " << ref_pv.size() << "  ("
              << (100.0 * static_cast<double>(over_1_255) / static_cast<double>(ref_pv.size())) << "%)\n";

    // Position ids must be exact. Pixels go through a different resampler
    // (stb Catmull-Rom vs PIL BICUBIC), so a small difference is expected.
    return pid_bad == 0 ? 0 : 3;
}

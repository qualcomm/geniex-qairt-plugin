// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "phonon_model.h"
#include "utils/detail/json.hpp"

namespace {
void writeWav(const std::string& path, const geniex::SpeechResult& result) {
    if (result.audio.size() > (UINT32_MAX - 36) / 2) throw std::runtime_error("WAV exceeds RIFF size limit");
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create WAV: " + path);
    const auto u32 = [&](uint32_t n) { out.write(reinterpret_cast<const char*>(&n), 4); };
    const auto u16 = [&](uint16_t n) { out.write(reinterpret_cast<const char*>(&n), 2); };
    out.write("RIFF", 4);
    u32(36 + static_cast<uint32_t>(result.audio.size() * 2));
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(result.sample_rate);
    u32(result.sample_rate * 2);
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(static_cast<uint32_t>(result.audio.size() * 2));
    for (float sample : result.audio) {
        const auto value = static_cast<int16_t>(std::clamp(sample, -1.0f, 1.0f) * 32767);
        u16(static_cast<uint16_t>(value));
    }
    if (!out) throw std::runtime_error("WAV write failed: " + path);
}
}  // namespace
int main(int argc, char** argv) {
    try {
        geniex::PhononOptions    options;
        geniex::QnnRuntimeConfig runtime;
        geniex::SpeechConfig     config;
        std::string              text, output = "phonon.wav";
        int                      repeats = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "--bundle")
                options.bundle_dir = value;
            else if (flag == "--soc-model")
                options.soc_model = value;
            else if (flag == "--lang")
                options.language = value;
            else if (flag == "--text-library")
                options.text_library = value;
            else if (flag == "--qairt")
                runtime.htp_dir = value;
            else if (flag == "--text")
                text = value;
            else if (flag == "--voice")
                config.voice = value;
            else if (flag == "--temperature")
                config.temperature = std::stof(value);
            else if (flag == "--out")
                output = value;
            else if (flag == "--repeats")
                repeats = std::stoi(value);
            else
                throw std::invalid_argument("Unknown option: " + flag);
        }
        if (text.empty() || repeats < 1 || repeats > 1000)
            throw std::invalid_argument("Provide text and 1..1000 repeats");
        auto model = std::make_unique<geniex::PhononModel>(options);
        if (!model->load(runtime)) throw std::runtime_error("Shared QAIRT model initialization failed");
        geniex::TTSPipeline pipeline(std::move(model));
        for (int i = 0; i < repeats; ++i) {
            size_t     callbacks = 0;
            const auto result    = pipeline.synthesize(text, config, [&](const float*, size_t, int) {
                ++callbacks;
                return true;
            });
            const auto file      = repeats == 1 ? output : output + "." + std::to_string(i) + ".wav";
            writeWav(file, result);
            const double seconds = static_cast<double>(result.samples) / result.sample_rate;
            qualla::json profiles;
            for (const auto& item : result.graphs)
                profiles[item.first] = {{"calls", item.second.calls}, {"accelerator_ms", item.second.accelerator_ms}};
            std::cout << qualla::json({{"run", i},
                                          {"first_pcm_ms", result.first_audio_ms},
                                          {"generation_ms", result.total_ms},
                                          {"audio_seconds", seconds},
                                          {"rtf", seconds > 0 ? result.total_ms / (1000 * seconds) : 0},
                                          {"chunks", result.chunks},
                                          {"callbacks", callbacks},
                                          {"graphs", profiles},
                                          {"wav", file}})
                             .dump()
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}

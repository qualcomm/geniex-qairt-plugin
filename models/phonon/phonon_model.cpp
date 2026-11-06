// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// Acoustic scheduling follows Gradium xn-ptts's MIT-licensed Phonon runner;
// see THIRD_PARTY_NOTICES.md. Graph execution uses GenieX Model/Graph only.

#include "phonon_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

#include "bundle.h"
#include "runtime.h"
#include "text_frontend.h"

namespace geniex {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::vector<uint16_t> readHalf(const std::filesystem::path& path, size_t count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || count > (1u << 28) || in.tellg() != static_cast<std::streamoff>(count * 2))
        throw std::runtime_error("Invalid Phonon half data: " + path.string());
    std::vector<uint16_t> data(count);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * 2)))
        throw std::runtime_error("Cannot read Phonon half data: " + path.string());
    return data;
}
int bounded(const qualla::json& value, const char* name, int maximum) {
    const int result = value.at(name).get<int>();
    if (result <= 0 || result > maximum) throw std::invalid_argument(std::string("Invalid Phonon ") + name);
    return result;
}
void requireSpec(const TensorSpec& spec, Qnn_DataType_t dtype, const std::vector<uint32_t>& shape) {
    if (spec.dtype != dtype || spec.shape != shape || spec.has_dynamic_dims)
        throw std::invalid_argument("Unexpected Phonon tensor layout: " + spec.name);
}
void copyOutput(Graph& graph, const std::string& output, const std::string& input) {
    if (graph.outputSpec(output).shape != graph.inputSpec(input).shape ||
        graph.outputSpec(output).dtype != graph.inputSpec(input).dtype)
        throw std::invalid_argument("Incompatible Phonon recurrent state: " + input);
    const void* source      = graph.outputPtr(output);
    void*       destination = graph.inputPtr(input);
    if (source != destination) std::memmove(destination, source, graph.inputSpec(input).byteCount());
}
}  // namespace

struct PhononModel::Impl {
    PhononOptions                                options;
    ModelConfig                                  model_config;
    std::unique_ptr<phonon_detail::TextFrontend> text;
    struct Voice {
        std::string           name;
        int                   length;
        std::vector<uint16_t> kv;
    };
    std::vector<Voice>    voices;
    std::vector<uint16_t> bos;
    Graph*                prefill     = nullptr;
    Graph*                step        = nullptr;
    int                   sample_rate = 0, slots = 0, prefill_tokens = 0, max_tokens = 0;
    int                   layers2 = 0, heads = 0, width = 0, min_frames = 0;
    double                frame_rate  = 0;
    float                 temperature = 0, eos_threshold = 0;
    size_t                states = 0;
    std::atomic<bool>     failed{false};
};

PhononModel::PhononModel(const PhononOptions& options) : impl_(std::make_unique<Impl>()) {
    auto& p   = *impl_;
    p.options = options;
    if (options.language.empty()) throw std::invalid_argument("Phonon requires a normalization language");
    const auto    root = std::filesystem::u8path(options.bundle_dir);
    std::ifstream file(root / "metadata.json");
    if (!file) throw std::invalid_argument("Phonon bundle has no metadata.json");
    const auto meta             = qualla::json::parse(file);
    p.model_config.model_paths  = {phonon_detail::bundleFile(
        root, phonon_detail::contextForSoc(meta.at("runtime").at("context_binaries"), options.soc_model))
             .string()};
    p.model_config.perf_profile = PerfProfile::BURST;
    const auto& gen             = meta.at("generation");
    p.sample_rate               = bounded(gen, "sample_rate", 192000);
    p.slots                     = bounded(gen, "cache_slots", 65536);
    p.prefill_tokens            = bounded(gen, "prefill_tokens", p.slots);
    p.max_tokens                = bounded(gen, "max_tokens_per_chunk", p.slots);
    p.frame_rate                = gen.at("frame_rate").get<double>();
    p.temperature               = gen.at("temperature").get<float>();
    p.eos_threshold             = gen.at("eos_threshold").get<float>();
    p.min_frames                = gen.at("min_frames_before_eos").get<int>();
    if (!std::isfinite(p.frame_rate) || p.frame_rate <= 0 || !std::isfinite(p.temperature) || p.temperature < 0 ||
        !std::isfinite(p.eos_threshold) || p.min_frames < 0 || p.min_frames >= p.slots)
        throw std::invalid_argument("Invalid Phonon generation parameters");
    const auto& layout = gen.at("layout");
    p.layers2          = 2 * bounded(layout, "layers", 128);
    p.heads            = bounded(layout, "heads", 128);
    p.width            = bounded(layout, "head_dim", 4096);
    size_t bos_count   = 1;
    for (int n : gen.at("bos").at("shape")) {
        if (n <= 0 || bos_count > (1u << 26) / static_cast<size_t>(n)) throw std::invalid_argument("Invalid BOS shape");
        bos_count *= static_cast<size_t>(n);
    }
    p.bos = readHalf(phonon_detail::bundleFile(root, gen.at("bos").at("file")), bos_count);
    for (const auto& voice : meta.at("voices")) {
        Impl::Voice v;
        v.name   = voice.at("name").get<std::string>();
        v.length = bounded(voice, "length", p.slots - p.prefill_tokens);
        if (v.name.empty() ||
            std::any_of(p.voices.begin(), p.voices.end(), [&](const auto& prior) { return prior.name == v.name; }))
            throw std::invalid_argument("Invalid or duplicate Phonon voice");
        v.kv = readHalf(
            phonon_detail::bundleFile(root, voice.at("file")), size_t(p.layers2) * p.heads * v.length * p.width);
        p.voices.push_back(std::move(v));
    }
    if (p.voices.empty()) throw std::invalid_argument("Phonon bundle has no voices");
    p.text = std::make_unique<phonon_detail::TextFrontend>(options.text_library,
        phonon_detail::bundleFile(root, meta.at("text").at("tokenizer")).string(),
        options.language);
}
PhononModel::~PhononModel() = default;
bool PhononModel::load(const QnnRuntimeConfig& runtime) {
    auto resolved = runtime;
    resolveHtpPaths(resolved);
    if (std::filesystem::u8path(*resolved.backend_path).filename() != kHtpBackendLib)
        throw std::invalid_argument("Phonon requires QNN HTP; no CPU/GPU fallback is supported");
    return Model::initialize(resolved, impl_->model_config);
}
bool                     PhononModel::isReady() const { return isInitialized() && !impl_->failed; }
std::vector<std::string> PhononModel::voices() const {
    std::vector<std::string> result;
    for (const auto& voice : impl_->voices) result.push_back(voice.name);
    return result;
}
bool PhononModel::onInitialized() {
    auto& p = *impl_;
    for (auto& graph : graphs_) {
        if (graph.name() == "prefill") p.prefill = &graph;
        if (graph.name() == "step") p.step = &graph;
    }
    if (!p.prefill || !p.step || graphs_.size() != 2)
        throw std::invalid_argument("Expected Phonon prefill and step graphs");
    const std::vector<uint32_t> cache  = {uint32_t(p.layers2), uint32_t(p.heads), uint32_t(p.slots), uint32_t(p.width)};
    const std::vector<uint32_t> prefix = {
        uint32_t(p.layers2), uint32_t(p.heads), uint32_t(p.prefill_tokens), uint32_t(p.width)};
    const std::vector<uint32_t> single = {uint32_t(p.layers2), uint32_t(p.heads), 1, uint32_t(p.width)};
    requireSpec(p.prefill->inputSpec("kv_cache"), QNN_DATATYPE_FLOAT_16, cache);
    requireSpec(p.step->inputSpec("kv_cache"), QNN_DATATYPE_FLOAT_16, cache);
    requireSpec(p.prefill->outputSpec("output_0"), QNN_DATATYPE_FLOAT_16, prefix);
    requireSpec(p.step->outputSpec("output_3"), QNN_DATATYPE_FLOAT_16, single);
    for (auto* graph : {p.prefill, p.step}) {
        if (graph->inputSpec("pos").dtype != QNN_DATATYPE_INT_32 || graph->inputSpec("pos").elementCount() != 1)
            throw std::invalid_argument("Invalid Phonon position input");
    }
    if (p.prefill->inputSpec("tokens").dtype != QNN_DATATYPE_INT_32 ||
        p.prefill->inputSpec("tokens").elementCount() != size_t(p.prefill_tokens) ||
        p.step->inputSpec("frame").dtype != QNN_DATATYPE_INT_32 || p.step->inputSpec("frame").elementCount() != 1 ||
        p.step->inputSpec("emb").dtype != QNN_DATATYPE_FLOAT_16 ||
        p.step->inputSpec("emb").elementCount() != p.bos.size() || p.step->outputSpec("output_1").elementCount() != 1)
        throw std::invalid_argument("Invalid Phonon graph inputs");
    if (p.step->outputSpec("output_2").shape != p.step->inputSpec("emb").shape ||
        p.step->outputSpec("output_2").dtype != p.step->inputSpec("emb").dtype)
        throw std::invalid_argument("Invalid Phonon embedding feedback");
    while (p.step->hasInput("mimi_states_" + std::to_string(p.states))) {
        const auto& in  = p.step->inputSpec("mimi_states_" + std::to_string(p.states));
        const auto& out = p.step->outputSpec("output_" + std::to_string(5 + p.states));
        if (in.shape != out.shape || in.dtype != out.dtype) throw std::invalid_argument("Invalid Phonon codec state");
        ++p.states;
    }
    if (!p.states) throw std::invalid_argument("Phonon graph has no codec state");
    if (p.options.verify_npu && !api_->enableExecutionProfiling())
        throw std::runtime_error("QNN basic profiling is unavailable");
    return true;
}

SpeechResult PhononModel::synthesize(const std::string& text, const SpeechConfig& config, const AudioCallback& on_audio,
    const StopCallback& should_stop) {
    auto& p = *impl_;
    if (!isReady()) throw std::runtime_error("Phonon model is not ready");
    std::string name = config.voice;
    if (name.empty()) {
        name = p.voices.front().name;
        for (const auto& voice : p.voices)
            if (voice.name == "default") name = voice.name;
    }
    const auto voice = std::find_if(p.voices.begin(), p.voices.end(), [&](const auto& v) { return v.name == name; });
    if (voice == p.voices.end()) throw std::invalid_argument("Unknown Phonon voice: " + name);
    const float temperature = config.temperature < 0 ? p.temperature : config.temperature;
    if (!std::isfinite(temperature) || temperature < 0) throw std::invalid_argument("Invalid Phonon temperature");
    const auto   start  = Clock::now();
    const auto   chunks = p.text->split(text, p.max_tokens, p.frame_rate);
    SpeechResult result;
    result.sample_rate = p.sample_rate;
    std::mt19937_64                 random(config.seed);
    std::normal_distribution<float> normal(0, std::sqrt(temperature));
    const auto                      before_prefill = api_->executionProfile("prefill");
    const auto                      before_step    = api_->executionProfile("step");
    auto                            stop           = [&] { return should_stop && should_stop(); };
    Graph&                          prefill        = *p.prefill;
    Graph&                          step           = *p.step;
    auto*                           kv             = static_cast<uint16_t*>(step.inputPtr("kv_cache"));
    const size_t                    row = p.width, plane = size_t(p.slots) * row;
    auto                            write_kv = [&](const void* source, int stride, int count, int at) {
        if (at < 0 || count < 0 || at > p.slots - count) throw std::runtime_error("Phonon KV bounds exceeded");
        const auto* block = static_cast<const uint16_t*>(source);
        for (int lh = 0; lh < p.layers2 * p.heads; ++lh)
            std::memmove(
                kv + lh * plane + size_t(at) * row, block + size_t(lh) * stride * row, size_t(count) * row * 2);
    };
    auto execute = [&](Graph& graph) {
        std::map<std::string, std::pair<double, uint16_t>> times;
        if (!graph.execute(times)) throw std::runtime_error("Phonon graph execution failed: " + graph.name());
    };
    try {
        for (const auto& chunk : chunks) {
            if (stop()) {
                result.cancelled = true;
                break;
            }
            const int n = static_cast<int>(chunk.tokens.size());
            if (!n) continue;
            const int pieces = (n + p.prefill_tokens - 1) / p.prefill_tokens;
            if (voice->length + pieces * p.prefill_tokens > p.slots)
                throw std::invalid_argument("Phonon text exceeds cache");
            const int max_frames = std::min(chunk.frame_budget, p.slots - voice->length - n);
            if (max_frames <= 0) throw std::invalid_argument("Phonon text leaves no audio cache");
            std::memset(kv, 0, step.inputSpec("kv_cache").byteCount());
            write_kv(voice->kv.data(), voice->length, voice->length, 0);
            int32_t              pos = voice->length;
            std::vector<int32_t> tokens(p.prefill_tokens);
            for (int first = 0; first < n; first += p.prefill_tokens) {
                if (stop()) {
                    result.cancelled = true;
                    break;
                }
                const int count = std::min(p.prefill_tokens, n - first);
                std::fill(tokens.begin(), tokens.end(), 0);
                std::copy_n(chunk.tokens.begin() + first, count, tokens.begin());
                prefill.write("tokens", tokens.data(), tokens.size());
                prefill.write("pos", &pos, 1);
                prefill.write("kv_cache", static_cast<const void*>(kv), prefill.inputSpec("kv_cache").byteCount());
                execute(prefill);
                write_kv(prefill.outputPtr("output_0"), p.prefill_tokens, count, pos);
                pos += count;
            }
            if (result.cancelled) break;
            step.write("emb", static_cast<const void*>(p.bos.data()), p.bos.size() * 2);
            for (size_t k = 0; k < p.states; ++k) {
                const auto state = "mimi_states_" + std::to_string(k);
                std::memset(step.inputPtr(state), 0, step.inputSpec(state).byteCount());
            }
            std::vector<float> noise(step.inputSpec("noise").elementCount());
            std::vector<float> pcm(step.outputSpec("output_0").elementCount());
            int                eos_frame = -1;
            for (int32_t frame = 0; frame < max_frames; ++frame) {
                if (stop()) {
                    result.cancelled = true;
                    break;
                }
                for (auto& sample : noise) sample = normal(random);
                step.write("noise", noise.data(), noise.size());
                step.write("pos", &pos, 1);
                step.write("frame", &frame, 1);
                execute(step);
                write_kv(step.outputPtr("output_3"), 1, 1, pos++);
                copyOutput(step, "output_2", "emb");
                for (size_t k = 0; k < p.states; ++k)
                    copyOutput(step, "output_" + std::to_string(5 + k), "mimi_states_" + std::to_string(k));
                float eos = 0;
                step.read("output_1", &eos, 1);
                if (!std::isfinite(eos)) throw std::runtime_error("Nonfinite Phonon EOS output");
                if (eos_frame < 0 && frame >= p.min_frames && eos > p.eos_threshold) eos_frame = frame;
                if (eos_frame >= 0 && frame >= eos_frame + chunk.frames_after_eos) break;
                step.read("output_0", pcm.data(), pcm.size());
                for (float sample : pcm)
                    if (!std::isfinite(sample)) throw std::runtime_error("Nonfinite Phonon audio");
                if (config.collect_audio) result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
                result.samples += pcm.size();
                if (result.first_audio_ms < 0) result.first_audio_ms = elapsed(start);
                if (on_audio && !on_audio(pcm.data(), pcm.size(), p.sample_rate)) {
                    result.cancelled = true;
                    break;
                }
            }
            ++result.chunks;
            if (result.cancelled) break;
        }
        const auto after_prefill = api_->executionProfile("prefill");
        const auto after_step    = api_->executionProfile("step");
        result.graphs["prefill"] = {after_prefill.calls - before_prefill.calls,
            (after_prefill.accelerator_us - before_prefill.accelerator_us) / 1000};
        result.graphs["step"]    = {
            after_step.calls - before_step.calls, (after_step.accelerator_us - before_step.accelerator_us) / 1000};
        if (p.options.verify_npu && !result.cancelled) {
            for (const auto& timing : result.graphs)
                if (!timing.second.calls || timing.second.accelerator_ms <= 0)
                    throw std::runtime_error("Missing positive NPU profiling for " + timing.first);
        }
        result.total_ms = elapsed(start);
        return result;
    } catch (...) {
        // A graph or callback failure invalidates the session; do not reuse an ambiguous cache.
        p.failed = true;
        throw;
    }
}
}  // namespace geniex

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace geniex::phonon_detail {
struct TextChunk {
    std::vector<int32_t> tokens;
    int                  frames_after_eos = 0;
    int                  frame_budget     = 0;
};
// C ABI published by Gradium qnn/text-ffi (ptts_text.h), pinned in README.
// No Gradium QNN runtime code is linked into this pipeline.
class TextFrontend {
    struct NativeChunk {
        const char*     text;
        const char*     prepared;
        const uint32_t* tokens;
        size_t          n_tokens;
        uint32_t        frames_after_eos, frame_budget, seq_budget;
    };
    struct NativeChunks {
        const NativeChunk* chunks;
        size_t             count;
    };
    void* library_                                                    = nullptr;
    void* handle_                                                     = nullptr;
    void* (*split_)(const void*, const char*, size_t, double, char**) = nullptr;
    void (*free_chunks_)(void*)                                       = nullptr;
    void (*free_string_)(char*)                                       = nullptr;
    void (*free_)(void*)                                              = nullptr;
    template <class T>
    T symbol(const char* name) {
#ifdef _WIN32
        const auto p = GetProcAddress(static_cast<HMODULE>(library_), name);
#else
        const auto p = dlsym(library_, name);
#endif
        if (!p) throw std::runtime_error(std::string("Missing text frontend symbol: ") + name);
        return reinterpret_cast<T>(p);
    }

   public:
    TextFrontend(const std::string& library, const std::string& tokenizer, const std::string& language) {
        const auto path = std::filesystem::u8path(library);
        if (!path.is_absolute() || !std::filesystem::is_regular_file(path))
            throw std::invalid_argument("Phonon text_library must name an absolute trusted native library");
#ifdef _WIN32
        library_ = LoadLibraryW(path.wstring().c_str());
#else
        library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library_) throw std::runtime_error("Cannot load Phonon text frontend: " + library);
        // Libraries stay resident: Rust TLS destructors may outlive a model.
        const auto create = symbol<void* (*)(const char*, const char*, char**)>("ptts_text_new");
        split_            = symbol<decltype(split_)>("ptts_text_split");
        free_chunks_      = symbol<decltype(free_chunks_)>("ptts_text_chunks_free");
        free_string_      = symbol<decltype(free_string_)>("ptts_text_string_free");
        free_             = symbol<decltype(free_)>("ptts_text_free");
        char* error       = nullptr;
        handle_           = create(tokenizer.c_str(), language.c_str(), &error);
        if (!handle_) {
            const std::string message = error ? error : "unknown error";
            if (error) free_string_(error);
            throw std::runtime_error("Phonon text frontend: " + message);
        }
    }
    ~TextFrontend() {
        if (handle_) free_(handle_);
    }
    TextFrontend(const TextFrontend&)                     = delete;
    TextFrontend&          operator=(const TextFrontend&) = delete;
    std::vector<TextChunk> split(const std::string& text, size_t max_tokens, double frame_rate) const {
        char* error  = nullptr;
        auto* chunks = static_cast<NativeChunks*>(split_(handle_, text.c_str(), max_tokens, frame_rate, &error));
        if (!chunks) {
            const std::string message = error ? error : "unknown error";
            if (error) free_string_(error);
            throw std::runtime_error("Phonon text splitting: " + message);
        }
        std::unique_ptr<void, decltype(free_chunks_)> guard(chunks, free_chunks_);
        std::vector<TextChunk>                        result;
        if (chunks->count > 4096 || (chunks->count && !chunks->chunks)) throw std::runtime_error("Invalid text chunks");
        for (size_t i = 0; i < chunks->count; ++i) {
            const auto& native = chunks->chunks[i];
            if (native.n_tokens > 65536 || !native.tokens || native.frames_after_eos > 4096 ||
                native.frame_budget > 65536)
                throw std::runtime_error("Invalid text chunk bounds");
            TextChunk chunk;
            chunk.tokens.assign(native.tokens, native.tokens + native.n_tokens);
            chunk.frames_after_eos = static_cast<int>(native.frames_after_eos) + 1;
            chunk.frame_budget     = static_cast<int>(native.frame_budget);
            result.push_back(std::move(chunk));
        }
        return result;
    }
};
}  // namespace geniex::phonon_detail

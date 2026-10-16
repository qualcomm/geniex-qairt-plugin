// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "runtime.h"

#include <mutex>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
extern "C" IMAGE_DOS_HEADER __ImageBase;
#elif defined(__ANDROID__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace geniex {

std::filesystem::path geniex_core_dir() {
#ifdef _WIN32
    // __ImageBase is a linker-generated symbol that refers to the base of
    // geniex_core.dll regardless of inlining.
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH);
    auto result = std::filesystem::path(path).parent_path();
    GENIEX_LOG_INFO("geniex_core_dir: resolved to {}", result.string());
    return result;
#elif defined(__ANDROID__) || defined(__linux__)
    // dladdr resolves the shared object for this function. Because this .cpp
    // is compiled only into libgeniex_core.so, the address always points to it.
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&geniex_core_dir), &info) && info.dli_fname) {
        auto result = std::filesystem::canonical(info.dli_fname).parent_path();
        GENIEX_LOG_INFO("geniex_core_dir: resolved to {}", result.string());
        return result;
    }
    throw std::runtime_error("geniex: cannot determine geniex_core library directory");
#else
    return std::filesystem::current_path();
#endif
}

#if defined(__linux__) && !defined(__ANDROID__)
void ensure_fastrpc_alias() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL)) return;

        std::error_code ec;
        const auto      alias = geniex_core_dir() / "libgeniex_cdsprpc_alias.so";
        if (!std::filesystem::exists(alias, ec)) return;

        if (dlopen(alias.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
            GENIEX_LOG_INFO("libcdsprpc.so not found; using bundled alias for libcdsprpc.so.1");
        } else {
            GENIEX_LOG_WARN("libcdsprpc.so not found and bundled alias failed to load: {}", dlerror());
        }
    });
}
#else
void ensure_fastrpc_alias() {}
#endif

}  // namespace geniex

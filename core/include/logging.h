// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// FMT_HEADER_ONLY and FMT_USE_CONSTEVAL are set globally via CMake compile
// definitions (GENIEX_COMPILE_DEFS). Guard against redefinition warnings.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#ifndef FMT_USE_CONSTEVAL
#define FMT_USE_CONSTEVAL 0
#endif
#include <cstdint>

#include "geniex_export.h"
#include "utils/fmt/core.h"

namespace geniex {

enum class LogLevel : uint32_t {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
};

// Signature for a user-installed log sink.
// message is null-terminated, fully formatted, with no trailing newline.
using LogCallback = void (*)(LogLevel level, const char* message);

// Global log sink. Defaults to a colorized stderr handler.
// Replace via geniex_set_log_callback() before calling any other API.
GENIEX_API extern LogCallback geniex_log_callback;

// Installs a custom log sink. Pass nullptr to restore the default handler.
GENIEX_API void geniex_set_log_callback(LogCallback cb);

// Null-safe pointer formatter for log arguments: prints "nullptr", the string
// value (char*), the address (void*), or the dereferenced value for other pointer types.
template <typename T>
inline auto lp(T arg) {
    if constexpr (std::is_pointer_v<T>) {
        if (arg == nullptr) {
            return fmt::format("nullptr");
        } else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_pointer_t<T>>, char>) {
            return fmt::format("{}", arg);
        } else if constexpr (std::is_same_v<std::remove_cv_t<T>, void*> ||
                             std::is_same_v<std::remove_cv_t<T>, const void*>) {
            return fmt::format("{}", static_cast<const void*>(arg));
        } else {
            return fmt::format("{}", *arg);
        }
    } else {
        return fmt::format("{}", arg);
    }
}

#ifdef GENIEX_DEBUG

#include <cstring>

template <typename... Args>
void geniex_log_internal(LogLevel level, const char* file, int32_t line, const char* func,
    fmt::format_string<Args...> fmt_str, Args&&... args) {
    if (geniex_log_callback == nullptr) return;
#ifdef PROJECT_SOURCE_DIR
    auto p        = std::strstr(file, PROJECT_SOURCE_DIR);
    auto filename = p ? p + std::strlen(PROJECT_SOURCE_DIR) + 1 : file;
#else
    auto filename = file;
#endif
    geniex_log_callback(level,
        fmt::format("[{}:{}:{}] {}", filename, line, func, fmt::format(fmt_str, lp(std::forward<Args>(args))...))
            .c_str());
}

#define GENIEX_LEVEL_LOG(level, ...) geniex::geniex_log_internal(level, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define GENIEX_LOG_TRACE(...) GENIEX_LEVEL_LOG(geniex::LogLevel::Trace, __VA_ARGS__)
#define GENIEX_LOG_DEBUG(...) GENIEX_LEVEL_LOG(geniex::LogLevel::Debug, __VA_ARGS__)
#define GENIEX_LOG_INFO(...) GENIEX_LEVEL_LOG(geniex::LogLevel::Info, __VA_ARGS__)
#define GENIEX_LOG_WARN(...) GENIEX_LEVEL_LOG(geniex::LogLevel::Warn, __VA_ARGS__)
#define GENIEX_LOG_ERROR(...) GENIEX_LEVEL_LOG(geniex::LogLevel::Error, __VA_ARGS__)

#else  // GENIEX_DEBUG

template <typename... Args>
inline void geniex_log_internal(LogLevel level, fmt::format_string<Args...> fmt_str, Args&&... args) {
    if (geniex_log_callback == nullptr) return;
    geniex_log_callback(level, fmt::format(fmt_str, lp(std::forward<Args>(args))...).c_str());
}

#define GENIEX_LOG_TRACE(...) geniex::geniex_log_internal(geniex::LogLevel::Trace, __VA_ARGS__)
#define GENIEX_LOG_DEBUG(...) geniex::geniex_log_internal(geniex::LogLevel::Debug, __VA_ARGS__)
#define GENIEX_LOG_INFO(...) geniex::geniex_log_internal(geniex::LogLevel::Info, __VA_ARGS__)
#define GENIEX_LOG_WARN(...) geniex::geniex_log_internal(geniex::LogLevel::Warn, __VA_ARGS__)
#define GENIEX_LOG_ERROR(...) geniex::geniex_log_internal(geniex::LogLevel::Error, __VA_ARGS__)

#endif  // GENIEX_DEBUG

}  // namespace geniex

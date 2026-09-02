// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// QNN HTP runtime support: device self-location, arch detection, and path resolution.
//
// Four concerns are unified here because they form a single pipeline:
//   1. geniex_core_dir()   — locate the directory containing geniex_core.dll/.so
//   2. detect_htp_arch()   — query the CDSP for the device's HTP arch version
//   3. selectHtpDir()      — choose WHICH runtime folder to load libraries from
//   4. resolveHtpPaths()   — fill nullopt QnnRuntimeConfig path fields from it
//
// Called automatically by Model::initialize() for any path field left as
// std::nullopt in QnnRuntimeConfig; set those fields to take explicit control.
//
// FastRPC constants are from the Qualcomm FastRPC public headers (BSD-3-Clause):
//   https://github.com/qualcomm/fastrpc

#include <cstdint>
#include <cstdlib>  // getenv, setenv / _putenv_s
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include "logging.h"
#include "types.h"

#ifdef _WIN32
#include <windows.h>
#include <winsvc.h>

#include <vector>
#elif defined(__ANDROID__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace geniex {

// Returns the directory that contains geniex_core.dll / libgeniex_core.so.
// Implemented in runtime.cpp (compiled exclusively into geniex_core) so that
// the address anchors (__ImageBase on Windows, dladdr on Linux/Android) always
// resolve to the geniex_core shared library — never to a consuming executable
// or a different DLL.
std::filesystem::path geniex_core_dir();

// Detects the HTP arch version of the current device by querying the CDSP
// via the FastRPC remote_handle_control API.
// The result is cached after the first call (thread-safe via std::call_once).
// Returns the arch version (e.g. 73, 75, 79, 81), or 0 on failure.
inline int detect_htp_arch() {
    static int            s_arch = -1;
    static std::once_flag s_flag;

    std::call_once(s_flag, []() {
        using remote_handle_control_t = int (*)(uint32_t, void*, uint32_t);
        remote_handle_control_t fn    = nullptr;

#ifdef _WIN32
        // Locate libcdsprpc.dll via the qcnspmcdm service in the Windows driver store.
        SC_HANDLE scm = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
        if (!scm) {
            GENIEX_LOG_WARN("HTP detect: cannot open SCManager ({})", GetLastError());
            s_arch = 0;
            return;
        }

        SC_HANDLE svc = OpenServiceW(scm, L"qcnspmcdm", SERVICE_QUERY_CONFIG);
        if (!svc) {
            GENIEX_LOG_WARN("HTP detect: qcnspmcdm service not found ({})", GetLastError());
            CloseServiceHandle(scm);
            s_arch = 0;
            return;
        }

        DWORD buf_size = 0;
        QueryServiceConfigW(svc, NULL, 0, &buf_size);
        std::vector<uint8_t> cfg_buf(buf_size);
        auto*                cfg = reinterpret_cast<LPQUERY_SERVICE_CONFIGW>(cfg_buf.data());
        if (!QueryServiceConfigW(svc, cfg, buf_size, &buf_size)) {
            GENIEX_LOG_WARN("HTP detect: QueryServiceConfigW failed ({})", GetLastError());
            CloseServiceHandle(svc);
            CloseServiceHandle(scm);
            s_arch = 0;
            return;
        }

        std::wstring drv_dir(cfg->lpBinaryPathName);
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);

        if (auto sep = drv_dir.find_last_of(L'\\'); sep != std::wstring::npos) drv_dir.resize(sep);

        // Resolve %SystemRoot% placeholder if present.
        const std::wstring placeholder = L"\\SystemRoot";
        if (drv_dir.compare(0, placeholder.size(), placeholder) == 0) {
            wchar_t windir[MAX_PATH];
            if (GetEnvironmentVariableW(L"windir", windir, MAX_PATH)) drv_dir.replace(0, placeholder.size(), windir);
        }

        std::wstring dll_path = drv_dir + L"\\libcdsprpc.dll";

        DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
        SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
        HMODULE lib = LoadLibraryW(dll_path.c_str());
        SetErrorMode(old_mode);

        if (!lib) {
            GENIEX_LOG_WARN("HTP detect: failed to load libcdsprpc.dll from driver store");
            s_arch = 0;
            return;
        }

        fn = reinterpret_cast<remote_handle_control_t>(GetProcAddress(lib, "remote_handle_control"));
        if (!fn) {
            GENIEX_LOG_WARN("HTP detect: remote_handle_control not found in libcdsprpc.dll");
            s_arch = 0;
            return;
        }

#else  // __ANDROID__ and __linux__
       // libcdsprpc.so is a system library on Android; on Linux the Qualcomm
       // FastRPC driver installs it; if absent, detection returns 0.
       // libcdsprpc is pulled in transitively by the QNN HTP stub anyway,
       // so we keep this mapping live for the process.
        void* lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            GENIEX_LOG_WARN("HTP detect: failed to load libcdsprpc.so: {}", dlerror());
            s_arch = 0;
            return;
        }

        fn = reinterpret_cast<remote_handle_control_t>(
            dlsym(lib, "remote_handle_control"));
        if (!fn) {
            GENIEX_LOG_WARN("HTP detect: remote_handle_control not found in libcdsprpc.so");
            s_arch = 0;
            return;
        }
#endif

        // FastRPC constants from the Qualcomm FastRPC public headers (BSD-3-Clause).
        constexpr uint32_t DSPRPC_GET_DSP_INFO = 2;
        constexpr uint32_t FASTRPC_ARCH_VER    = 6;
        constexpr uint32_t FASTRPC_CDSP_DOMAIN = 3;

        struct {
            uint32_t domain, attribute_ID, capability;
        } cap{};
        cap.domain       = FASTRPC_CDSP_DOMAIN;
        cap.attribute_ID = FASTRPC_ARCH_VER;

        int err = fn(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
        if (err != 0) {
            GENIEX_LOG_WARN("HTP detect: DSPRPC_GET_DSP_INFO failed (err={})", err);
            s_arch = 0;
            return;
        }

        switch (cap.capability & 0xff) {
            case 0x68:
                s_arch = 68;
                break;
            case 0x69:
                s_arch = 69;
                break;
            case 0x73:
                s_arch = 73;
                break;
            case 0x75:
                s_arch = 75;
                break;
            case 0x79:
                s_arch = 79;
                break;
            case 0x81:
                s_arch = 81;
                break;
            case 0x85:
                s_arch = 85;
                break;
            default:
                GENIEX_LOG_WARN("HTP detect: unknown arch capability 0x{:x}", cap.capability);
                s_arch = 0;
        }

        if (s_arch > 0) GENIEX_LOG_INFO("Detected HTP arch: v{}", s_arch);
        // `lib` intentionally leaked
    });

    return s_arch;
}

// The three libraries a runtime folder must provide, by platform.
#ifdef _WIN32
inline constexpr const char* kHtpBackendLib = "QnnHtp.dll";
inline constexpr const char* kHtpSystemLib  = "QnnSystem.dll";
inline constexpr const char* kHtpExtLib     = "QnnHtpNetRunExtensions.dll";
#else  // __ANDROID__ and __linux__
inline constexpr const char* kHtpBackendLib = "libQnnHtp.so";
inline constexpr const char* kHtpSystemLib  = "libQnnSystem.so";
inline constexpr const char* kHtpExtLib     = "libQnnHtpNetRunExtensions.so";
#endif

// Which knob produced the runtime folder, so a failed load can name the thing
// the caller actually set.
enum class HtpDirSource {
    ConfigField,  // QnnRuntimeConfig::htp_dir
    Environment,  // GENIEX_QNN_LIB
    Bundled,      // <geniex_core_dir>/htp-files
    CoreDirFlat,  // <geniex_core_dir> itself, for flattened deployments
};

struct HtpDirChoice {
    std::filesystem::path dir;
    HtpDirSource          source;
};

// Chooses which folder to load the HTP runtime from, highest precedence first:
//
//   1. cfg_dir              — QnnRuntimeConfig::htp_dir
//   2. env_value            — GENIEX_QNN_LIB
//   3. <core_dir>/htp-files — the bundled runtime, when bundled_dir_exists
//   4. <core_dir>           — flattened deployments that drop the runtime libs
//                             directly beside geniex_core (e.g. Android packaging)
//
// Pure by design -- the caller supplies `bundled_dir_exists` and validates the
// result -- so the precedence rules are unit testable (tests/core/runtime_test.cpp).
inline HtpDirChoice selectHtpDir(const std::optional<std::string>& cfg_dir, const char* env_value,
    const std::filesystem::path& core_dir, bool bundled_dir_exists) {
    if (cfg_dir.has_value() && !cfg_dir->empty()) {
        return {std::filesystem::path(*cfg_dir), HtpDirSource::ConfigField};
    }
    if (env_value != nullptr && env_value[0] != '\0') {
        return {std::filesystem::path(env_value), HtpDirSource::Environment};
    }
    if (bundled_dir_exists) {
        return {core_dir / "htp-files", HtpDirSource::Bundled};
    }
    return {core_dir, HtpDirSource::CoreDirFlat};
}

// True if `dir` ships Hexagon skel libraries (libQnnHtp<arch>Skel.so), as the
// bundled htp-files/ does. Host-library-only folders do not -- a stock QAIRT SDK
// keeps skels under lib/hexagon-v<arch>/unsigned/, separate from the host libs.
inline bool hasHexagonSkels(const std::filesystem::path& dir) {
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        if (it->path().filename().string().find("Skel.so") != std::string::npos) return true;
    }
    return false;
}

// Points the Hexagon FastRPC loader at `dir` so it can find the skel libraries
// that sit alongside the host libraries in a runtime folder. Only call this when
// `dir` actually holds skels; pointing FastRPC at a folder with none costs a
// failed DSP session, or an outright load failure on a stricter platform.
inline void setAdspLibraryPath(const std::filesystem::path& dir) {
    const std::string value = dir.string();
#ifdef _WIN32
    _putenv_s("ADSP_LIBRARY_PATH", value.c_str());
#else  // __ANDROID__ and __linux__
    setenv("ADSP_LIBRARY_PATH", value.c_str(), 1);
#endif
    GENIEX_LOG_DEBUG("ADSP_LIBRARY_PATH set to {}", value);
}

// Fills any std::nullopt path fields in `cfg` from the runtime folder chosen by
// selectHtpDir. Fields that already have a value are left unchanged; if all
// three are set this is a no-op. Throws std::runtime_error if the chosen folder
// is missing or does not hold a QNN runtime.
//
// Side effects: sets ADSP_LIBRARY_PATH so FastRPC finds the skels, and on
// Windows calls SetDllDirectoryA() so the loader finds transitive HTP DLL
// dependencies (e.g. QnnHtpV73Stub.dll).
inline void resolveHtpPaths(QnnRuntimeConfig& cfg) {
    if (cfg.backend_path.has_value() && cfg.system_lib_path.has_value() && cfg.extensions_path.has_value()) {
        return;
    }

    // Arch is logged for diagnostics; a runtime folder bundles all arch variants together.
    int arch = detect_htp_arch();
    if (arch > 0)
        GENIEX_LOG_INFO("HTP arch v{} detected.", arch);
    else
        GENIEX_LOG_WARN("HTP arch detection failed; continuing with platform folder.");

    // Names a runtime folder to load instead of the bundled one, so a caller can
    // run against another QAIRT version without rebuilding. Documented in README.md.
    constexpr const char* kEnvVar = "GENIEX_QNN_LIB";

    const auto core_dir = geniex_core_dir();
    const auto choice =
        selectHtpDir(cfg.htp_dir, std::getenv(kEnvVar), core_dir, std::filesystem::exists(core_dir / "htp-files"));
    const auto& runtime_dir = choice.dir;
    const bool  overridden  = choice.source == HtpDirSource::ConfigField || choice.source == HtpDirSource::Environment;

    // Named rather than numbered: this goes into user-facing text, where a rung
    // index would not tell the caller which knob to change.
    const char* source = "unknown";
    switch (choice.source) {
        case HtpDirSource::ConfigField:
            source = "QnnRuntimeConfig::htp_dir";
            break;
        case HtpDirSource::Environment:
            source = kEnvVar;
            break;
        case HtpDirSource::Bundled:
            source = "bundled htp-files/";
            break;
        case HtpDirSource::CoreDirFlat:
            source = "geniex_core directory (flat layout)";
            break;
    }

    if (!std::filesystem::is_directory(runtime_dir)) {
        throw std::runtime_error("geniex: HTP runtime folder not found: " + runtime_dir.string() + "\n  (from " +
                                 source + ")\nExpected a directory holding " + kHtpBackendLib +
                                 " and its arch stubs. Unset " + kEnvVar +
                                 " to fall back to the runtime bundled with geniex_core.");
    }

    if (!std::filesystem::exists(runtime_dir / kHtpBackendLib)) {
        std::string msg = "geniex: " + std::string(kHtpBackendLib) +
                          " not found in HTP runtime folder: " + runtime_dir.string() + "\n  (from " + source + ")";
        // A stock QAIRT SDK root is the likely mistake: it keeps host libraries
        // under lib/<target-triple>/ rather than flat. Say so instead of failing
        // with a bare "not found".
        if (std::filesystem::is_directory(runtime_dir / "lib")) {
            msg += "\nThis looks like a QAIRT SDK root. Point " + std::string(kEnvVar) +
                   " at the host library directory itself (the lib/<target-triple>/ folder holding " + kHtpBackendLib +
                   "), not the SDK root.";
        } else {
            msg += "\nExpected a flat directory holding " + std::string(kHtpBackendLib) +
                   " and its arch stubs, shaped like the bundled htp-files/.";
        }
        throw std::runtime_error(msg);
    }

    if (overridden)
        GENIEX_LOG_INFO("HTP runtime path: {} (overridden via {})", runtime_dir.string(), source);
    else
        GENIEX_LOG_INFO("HTP runtime path: {} (auto-resolved from {})", runtime_dir.string(), source);

    if (!cfg.backend_path.has_value()) cfg.backend_path = (runtime_dir / kHtpBackendLib).string();
    if (!cfg.system_lib_path.has_value()) cfg.system_lib_path = (runtime_dir / kHtpSystemLib).string();
    if (!cfg.extensions_path.has_value()) cfg.extensions_path = (runtime_dir / kHtpExtLib).string();

    if (hasHexagonSkels(runtime_dir)) {
        setAdspLibraryPath(runtime_dir);
    } else {
        GENIEX_LOG_DEBUG("No Hexagon skels in {}; leaving ADSP_LIBRARY_PATH as-is", runtime_dir.string());
    }

#ifdef _WIN32
    // Allow the loader to find transitive HTP DLL dependencies in the same folder.
    SetDllDirectoryA(runtime_dir.string().c_str());
#endif
}

}  // namespace geniex
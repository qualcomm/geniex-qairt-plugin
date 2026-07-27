// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// QNN HTP runtime support: device self-location, arch detection, and path resolution.
//
// Three concerns are unified here because they form a single pipeline:
//   1. geniex_core_dir()   — locate the directory containing geniex_core.dll/.so
//   2. detect_htp_arch()   — query the CDSP for the device's HTP arch version
//   3. resolveHtpPaths()   — fill nullopt QnnRuntimeConfig path fields from the
//                            htp-files/ folder installed next to geniex_core
//
// Called automatically by Model::initialize() for any path field left as
// std::nullopt in QnnRuntimeConfig. Callers who want explicit control simply
// set the path fields before calling initialize().
//
// Platform support:
//   Windows  — GetModuleHandleExW/GetModuleFileNameW for self-location;
//               SCManager + libcdsprpc.dll for arch detection;
//               SetDllDirectoryA() so the loader finds HTP DLL dependencies.
//   Android  — dladdr() for self-location; libcdsprpc.so for arch detection.
//   Linux    — dladdr() for self-location; libcdsprpc.so for arch detection
//               (requires Qualcomm FastRPC driver, e.g. on Snapdragon X Elite
//               Linux dev kits); falls back to v79 if detection fails.
//
// FastRPC constants are from the Qualcomm FastRPC public headers (BSD-3-Clause):
//   https://github.com/qualcomm/fastrpc

#include <cstdint>
#include <filesystem>
#include <mutex>
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

        if (s_arch > 0) GENIEX_LOG_DEBUG("Detected HTP arch: v{}", s_arch);
        // `lib` intentionally leaked
    });

    return s_arch;
}

// Fills any std::nullopt path fields in `cfg` using the platform HTP folder
// (<geniex_core_dir>/htp-files/). Throws std::runtime_error if the folder is
// absent. Fields that already have a value are left unchanged.
// Side effect on Windows: calls SetDllDirectoryA() so the loader can find
// transitive HTP DLL dependencies (e.g. QnnHtpV73Stub.dll).
inline void resolveHtpPaths(QnnRuntimeConfig& cfg) {
    if (cfg.backend_path.has_value() && cfg.system_lib_path.has_value() && cfg.extensions_path.has_value()) {
        return;
    }

    // Arch is logged for diagnostics; the htp-files/ folder bundles all arch variants together.
    int arch = detect_htp_arch();
    if (arch > 0)
        GENIEX_LOG_DEBUG("HTP arch v{} detected.", arch);
    else
        GENIEX_LOG_WARN("HTP arch detection failed; continuing with platform folder.");

    auto htp_dir = geniex_core_dir() / "htp-files";

    if (!std::filesystem::exists(htp_dir)) {
        throw std::runtime_error("geniex: HTP runtime folder not found: " + htp_dir.string() +
                                 "\nExpected htp-files/ to be installed alongside geniex_core. "
                                 "Set QnnRuntimeConfig path fields explicitly to override.");
    }

    GENIEX_LOG_DEBUG("Auto-resolved HTP runtime path: {}", htp_dir.string());

#ifdef _WIN32
    if (!cfg.backend_path.has_value()) cfg.backend_path = (htp_dir / "QnnHtp.dll").string();
    if (!cfg.system_lib_path.has_value()) cfg.system_lib_path = (htp_dir / "QnnSystem.dll").string();
    if (!cfg.extensions_path.has_value()) cfg.extensions_path = (htp_dir / "QnnHtpNetRunExtensions.dll").string();

    // Allow the loader to find transitive HTP DLL dependencies in the same folder.
    SetDllDirectoryA(htp_dir.string().c_str());
#else  // __ANDROID__ and __linux__
    if (!cfg.backend_path.has_value()) cfg.backend_path = (htp_dir / "libQnnHtp.so").string();
    if (!cfg.system_lib_path.has_value()) cfg.system_lib_path = (htp_dir / "libQnnSystem.so").string();
    if (!cfg.extensions_path.has_value()) cfg.extensions_path = (htp_dir / "libQnnHtpNetRunExtensions.so").string();
#endif
}

}  // namespace geniex
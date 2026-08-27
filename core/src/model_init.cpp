// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Device-only Model initialization: QNN backend load, HTP path resolution, and
// graph setup. Separated from model.cpp so that file's CPU-reachable code can
// be measured by the coverage surface without pulling in QNN/HTP dependencies.

#include <cstdarg>
#include <cstring>

#include "QnnConfig.hpp"
#include "llm/llm_spec_loader.h"  // parseHtpCoreCount
#include "logging.h"
#include "model.h"
#include "qnn-utils.hpp"
#include "runtime.h"

namespace geniex {

// Bridges QNN's va_list callback into geniex logging. Uses vsnprintf rather than
// a variadic wrapper because QNN delivers an already-formatted va_list. Prefixes
// "[QNN] " to distinguish QNN-internal messages from library-originated ones.
static constexpr size_t kQnnLogBufSize = 1024;

// Some QNN messages arrive at ERROR level but are not actionable:
//
//   * teardown artifacts — when the HTP context is freed, FastRPC may report that
//     a buffer it already reclaimed "failed to unmap" (error 0x3a /
//     AEE_ENOSUCHMAP);
//   * "[ERROR][SetBufName][SW|NSW] status=-1" — the HTP backend tries to attach a
//     debug name to each graph I/O buffer and the shipped context binaries carry
//     no name table, so it reports -1 once per tensor. The SDK's own qnn-net-run
//     prints the same lines against these binaries and still exits 0 with correct
//     output, so it is a property of the assets, not of this runtime.
//
// Both fire on every single run and only alarm users, so drop them before they
// reach the log sink. Matching stays narrow on purpose: a broad filter here would
// hide real QNN errors, which are the main diagnostic signal when a graph fails.
static bool isBenignQnnMessage(const char* msg) {
    static constexpr const char* kBenignSubstrings[] = {
        "fastrpc memory failed to unmap",
        "fastrpc memory unmap error reporting failed",
        "UnMapping buffer fd",
        "[SetBufName]",
    };
    for (const char* needle : kBenignSubstrings) {
        if (std::strstr(msg, needle) != nullptr) return true;
    }
    return false;
}

static void qnnLogCallback(const char* fmt, uint32_t level, uint64_t /*timestamp*/, va_list args) {
    char buf[kQnnLogBufSize];
    vsnprintf(buf, sizeof(buf), fmt, args);

    if (isBenignQnnMessage(buf)) return;

    // QNN numeric levels: 1=ERROR, 2=WARN, 3=INFO, 4=VERBOSE, 5=DEBUG.
    LogLevel mapped;
    switch (level) {
        case 1:
            mapped = LogLevel::Error;
            break;
        case 2:
            mapped = LogLevel::Warn;
            break;
        case 3:
            mapped = LogLevel::Info;
            break;
        case 4:
            mapped = LogLevel::Trace;
            break;
        default:
            mapped = LogLevel::Debug;
            break;
    }

    if (geniex_log_callback) {
        auto msg = fmt::format("[QNN] {}", buf);
        geniex_log_callback(mapped, msg.c_str());
    }
}

// Requests multicore execution when the config asks for more than one HTP core.
// Runs between context creation and the first execute — the only window in which
// QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES can still be applied to graphs retrieved
// from a prebuilt context binary. Never fails init: a device/driver without
// multicore support (QNN logs "Multicore support is unavailable") keeps running
// on a single core, just no longer silently.
void Model::applyHtpNumCores(const ModelConfig& model_cfg) {
    const uint32_t device_cores = api_->getHtpDeviceNumCores();
    if (device_cores > 0) {
        GENIEX_LOG_INFO("HTP device reports {} NSP core(s)", device_cores);
    } else {
        GENIEX_LOG_INFO("HTP device core count not reported by QNN platform info");
    }

    uint32_t requested = model_cfg.num_cores;
    if (requested == 0 && !model_cfg.htp_config_path.empty()) {
        // Callers that build ModelConfig by hand (example executables, embedders)
        // still get the bundle's core count; modelConfigFromDirectory pre-fills
        // num_cores, making this re-parse a no-op on that path.
        requested = parseHtpCoreCount(model_cfg.htp_config_path);
    }
    if (requested <= 1) {
        GENIEX_LOG_INFO(
            "HTP graphs will execute on 1 core (default; set num_cores or add "
            "htp_backend_ext_config.json `cores` entries to request more)");
        return;
    }

    if (device_cores > 0 && requested > device_cores) {
        GENIEX_LOG_WARN(
            "Requested {} HTP cores but the device exposes {}; clamping to {}", requested, device_cores, device_cores);
        requested = device_cores;
    }
    if (requested <= 1) {
        GENIEX_LOG_INFO("HTP graphs will execute on 1 core");
        return;
    }

    if (api_->setHtpNumCores(requested)) {
        GENIEX_LOG_INFO("HTP multicore enabled: graphs will execute on {} cores", requested);
    } else {
        GENIEX_LOG_WARN(
            "HTP multicore requested ({} cores) but the backend rejected "
            "QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES; continuing on a single core. "
            "Multicore may require context binaries generated with a multicore "
            "graph config and a SoC/driver exposing more than one NSP core.",
            requested);
    }
}

bool Model::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    if (initialized_) return true;
    model_cfg_ = model_cfg;

    QnnRuntimeConfig resolved_cfg = runtime_cfg;
    resolveHtpPaths(resolved_cfg);

    api_ = std::make_unique<QnnApi>();

    io_tensor_ = std::make_shared<IOTensor>(BufferAlloc::SHARED_BUFFER, api_->getQnnInterfaceVer());
    api_->setIOTensorBufferMgr(io_tensor_.get());

    // No backend-extensions library: QnnRuntimeConfig::extensions_path is accepted
    // for source compatibility but no longer loaded. Everything that library did is
    // applied through the public C API from the values parsed out of
    // htp_backend_ext_config.json (see parseHtpConfig).
    if (resolved_cfg.extensions_path.has_value() && !resolved_cfg.extensions_path->empty()) {
        GENIEX_LOG_INFO(
            "extensions_path is ignored; HTP config is applied via the QNN C API directly");
    }

    // Read the bundle's HTP knobs ourselves. Covers both modelConfigFromDirectory
    // bundles and hand-built configs (example executables) that only set the path.
    PerfProfile htp_perf_profile   = model_cfg.perf_profile;
    uint32_t    htp_rpc_latency_us  = model_cfg.rpc_control_latency_us;
    bool        htp_weight_sharing  = model_cfg.weight_sharing_enabled;
    if (!model_cfg.htp_config_path.empty()) {
        parseHtpConfig(
            model_cfg.htp_config_path, htp_perf_profile, htp_rpc_latency_us, htp_weight_sharing);
    }

    const bool ok = api_->initializeHtp(resolved_cfg.backend_path.value(),
        model_cfg.model_paths,
        htp_perf_profile,
        htp_rpc_latency_us,
        htp_weight_sharing,
        {},
        true,
        resolved_cfg.system_lib_path.value_or(""),
        resolved_cfg.debug,
        0,
        0,
        false,
        false,
        0,
        true,
        false,
        {},
        false,
        false,
        static_cast<uint32_t>(resolved_cfg.log_level),
        qnnLogCallback);

    if (!ok) {
        return false;
    }

    if (api_->perfVoteApplied()) {
        GENIEX_LOG_INFO("HTP power vote applied (perf_profile={}, rpc_control_latency={}us)",
            static_cast<int>(htp_perf_profile),
            htp_rpc_latency_us);
    } else {
        GENIEX_LOG_WARN(
            "HTP power vote was NOT applied; the NSP runs at the backend default power state");
    }

    applyHtpNumCores(model_cfg);

    auto quallaPerf = qualla::QnnUtils::qnnToQuallaPerformanceProfile(htp_perf_profile);
    api_->setPerfProfile(quallaPerf);

    qnn_wrapper_api::GraphInfo_t** graphs_info = api_->getGraphsInfo();
    const uint32_t                 count       = api_->getGraphsCount();

    graphs_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        qnn_wrapper_api::GraphInfo_t* g   = graphs_info[i];
        Qnn_ContextHandle_t           ctx = api_->getContexts(g);
        graphs_.emplace_back(g, api_.get(), io_tensor_.get());
        if (!graphs_.back().setup(ctx)) {
            return false;
        }
    }

    if (!onInitialized()) {
        return false;
    }

    initialized_ = true;
    return true;
}

}  // namespace geniex

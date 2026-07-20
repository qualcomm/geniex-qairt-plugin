// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Device-only Model initialization: QNN backend load, HTP path resolution, and
// graph setup. Separated from model.cpp so that file's CPU-reachable code can
// be measured by the coverage surface without pulling in QNN/HTP dependencies.

#include <cstdarg>
#include <cstring>

#include "QnnConfig.hpp"
#include "logging.h"
#include "model.h"
#include "qnn-utils.hpp"
#include "runtime.h"

namespace geniex {

// Bridges QNN's va_list callback into geniex logging. Uses vsnprintf rather than
// a variadic wrapper because QNN delivers an already-formatted va_list. Prefixes
// "[QNN] " to distinguish QNN-internal messages from library-originated ones.
static constexpr size_t kQnnLogBufSize = 1024;

// Some QNN messages are benign teardown artifacts, not actionable errors: when
// the HTP context is freed, FastRPC may report that a buffer it already reclaimed
// "failed to unmap" (error 0x3a / AEE_ENOSUCHMAP). These fire on every clean
// shutdown and only alarm users, so drop them before they reach the log sink.
static bool isBenignQnnMessage(const char* msg) {
    static constexpr const char* kBenignSubstrings[] = {
        "fastrpc memory failed to unmap",
        "fastrpc memory unmap error reporting failed",
        "UnMapping buffer fd",
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

bool Model::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    if (initialized_) return true;
    model_cfg_ = model_cfg;

    QnnRuntimeConfig resolved_cfg = runtime_cfg;
    resolveHtpPaths(resolved_cfg);

    api_ = std::make_unique<QnnApi>();

    io_tensor_ = std::make_shared<IOTensor>(BufferAlloc::SHARED_BUFFER, api_->getQnnInterfaceVer());
    api_->setIOTensorBufferMgr(io_tensor_.get());

    // extensions_path value_or("") preserves the convention where empty string disables the library.
    BackendExtensionsConfigs ext_cfg(resolved_cfg.extensions_path.value_or(""), model_cfg.htp_config_path);

    const bool ok = api_->initializeHtp(resolved_cfg.backend_path.value(),
        model_cfg.model_paths,
        ext_cfg,
        qnn::tools::netrun::PerfProfile::BURST,
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

    auto quallaPerf = qualla::QnnUtils::qnnToQuallaPerformanceProfile(model_cfg.perf_profile);
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

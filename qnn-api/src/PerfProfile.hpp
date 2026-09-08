// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

namespace geniex {

// HTP performance profile requested for a model.
//
// This used to be qnn::tools::netrun::PerfProfile, a type owned by the QAIRT
// SDK's qnn-net-run sample sources. Depending on that header pulled the whole
// IBackend C++ interface into our public API, and IBackend's vtable is
// reordered every QAIRT release (46 -> 49 -> 51 -> 54 slots across 2.36/2.45/
// 2.47/2.48), which made a plugin binary usable only with the exact SDK it was
// compiled against. The profile is now our own type and is applied through the
// public C API (QnnHtpPerfInfrastructure DCVS v3), so one build spans every
// QAIRT version the C interface negotiates.
//
// Member order is kept identical to the old netrun enum so existing config
// files and integer round-trips keep their meaning.
enum class PerfProfile {
    LOW_BALANCED,
    BALANCED,
    DEFAULT,
    HIGH_PERFORMANCE,
    SUSTAINED_HIGH_PERFORMANCE,
    BURST,
    EXTREME_POWER_SAVER,
    LOW_POWER_SAVER,
    POWER_SAVER,
    HIGH_POWER_SAVER,
    SYSTEM_SETTINGS,
    NO_USER_INPUT,
    CUSTOM,
    INVALID
};

// Load-time HTP power knobs, from htp_backend_ext_config.json devices[].cores[].
//
// Only keys the QAIRT docs annotate "Used by qnn-net-run" belong here -- those take
// effect when a context is loaded. Keys annotated "Used by
// qnn-context-binary-generator during offline preparation" (num_cores,
// weight_sharing_enabled, vtcm_mb, dlbc, the graph fusion/precision switches, ...)
// are baked into the context binary when it is generated and cannot be applied at
// load time, so they are deliberately absent.
// Schema: <qairt-sdk>/docs/QAIRT-Docs/QNN/general/htp/htp_backend.html
struct HtpPerfConfig {
    PerfProfile profile = PerfProfile::BURST;
    // Each duration is in microseconds; 0 = leave the backend default alone.
    uint32_t rpc_control_latency_us   = 0;  // doc default: unset
    uint32_t rpc_polling_time_us      = 0;  // doc default: 9999us for burst-family profiles
    uint32_t hmx_timeout_us           = 0;  // doc default: 300000us
    uint32_t adaptive_polling_time_us = 0;  // doc default: 0us
};

}  // namespace geniex

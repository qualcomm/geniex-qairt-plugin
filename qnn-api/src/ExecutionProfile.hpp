// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <set>
#include <vector>
#include "QnnHtpProfile.h"
#include "QnnInterface.h"

struct GraphExecutionProfile {
    uint64_t calls = 0;
    double accelerator_us = 0;
};
// Only accelerator events prove device execution; host/RPC time is not a substitute.
inline bool collectExecutionProfile(const QNN_INTERFACE_VER_TYPE& api, Qnn_ProfileHandle_t profile,
    GraphExecutionProfile& total) {
    if (!api.profileGetEvents || !api.profileGetEventData || !api.profileGetSubEvents) return false;
    const QnnProfile_EventId_t* events = nullptr;
    uint32_t count = 0;
    if (api.profileGetEvents(profile, &events, &count) != QNN_SUCCESS || (count && !events)) return false;
    std::vector<QnnProfile_EventId_t> pending;
    if (count) pending.assign(events, events + count);
    std::set<QnnProfile_EventId_t> visited;
    double accelerator_us = 0;
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (!visited.insert(id).second) continue;
        if (visited.size() > 4096) return false;
        QnnProfile_EventData_t data{};
        if (api.profileGetEventData(id, &data) != QNN_SUCCESS) return false;
        if (data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC)
            accelerator_us += static_cast<double>(data.value);
        const QnnProfile_EventId_t* children = nullptr;
        uint32_t n = 0;
        if (api.profileGetSubEvents(id, &children, &n) != QNN_SUCCESS || (n && !children)) return false;
        if (n) pending.insert(pending.end(), children, children + n);
    }
    total.calls++;
    total.accelerator_us += accelerator_us;
    return true;
}

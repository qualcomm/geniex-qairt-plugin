// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include "ExecutionProfile.hpp"
namespace {
int                    mode = 0;
QNN_INTERFACE_VER_TYPE profileApi() {
    QNN_INTERFACE_VER_TYPE api{};
    api.profileGetEvents = [](Qnn_ProfileHandle_t, const QnnProfile_EventId_t** ids, uint32_t* n) -> Qnn_ErrorHandle_t {
        static const QnnProfile_EventId_t root[] = {1, 1};
        if (mode == 1) return QNN_PROFILE_ERROR_INVALID_HANDLE;
        *ids = root;
        *n   = 2;
        return QNN_SUCCESS;
    };
    api.profileGetEventData = [](QnnProfile_EventId_t id, QnnProfile_EventData_t* data) -> Qnn_ErrorHandle_t {
        if (mode == 2) return QNN_PROFILE_ERROR_INVALID_HANDLE;
        data->type  = id == 1 ? QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC
                              : QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC;
        data->value = id == 1 ? 1000 : 17;
        return QNN_SUCCESS;
    };
    api.profileGetSubEvents =
        [](QnnProfile_EventId_t id, const QnnProfile_EventId_t** ids, uint32_t* n) -> Qnn_ErrorHandle_t {
        static const QnnProfile_EventId_t child[]  = {2};
        static const QnnProfile_EventId_t parent[] = {1};
        if (mode == 3) return QNN_PROFILE_ERROR_INVALID_HANDLE;
        *ids = id == 1 ? child : parent;
        *n   = id == 1 || mode == 4 ? 1 : 0;
        return QNN_SUCCESS;
    };
    return api;
}
TEST(ExecutionProfile, CountsOnlyAcceleratorEventsAndDeduplicatesCycles) {
    mode                      = 4;
    const auto            api = profileApi();
    GraphExecutionProfile total;
    EXPECT_TRUE(collectExecutionProfile(api, nullptr, total));
    EXPECT_EQ(total.calls, 1);
    EXPECT_DOUBLE_EQ(total.accelerator_us, 17);
    EXPECT_TRUE(collectExecutionProfile(api, nullptr, total));
    EXPECT_EQ(total.calls, 2);
    EXPECT_DOUBLE_EQ(total.accelerator_us, 34);
}
TEST(ExecutionProfile, FailsWithoutUpdatingTotalsOnVendorErrors) {
    auto api = profileApi();
    for (mode = 1; mode <= 3; ++mode) {
        GraphExecutionProfile total{7, 90};
        EXPECT_FALSE(collectExecutionProfile(api, nullptr, total));
        EXPECT_EQ(total.calls, 7);
        EXPECT_DOUBLE_EQ(total.accelerator_us, 90);
    }
    api.profileGetEvents = nullptr;
    GraphExecutionProfile total;
    EXPECT_FALSE(collectExecutionProfile(api, nullptr, total));
}
}  // namespace

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// ABI declarations for SDKs older than QAIRT 2.48.
#if __has_include("System/QnnSystemProfile.h")
#include "System/QnnSystemProfile.h"
#else

#include "QnnProfile.h"
#include "QnnTypes.h"

typedef void* QnnSystemProfile_SerializationTargetHandle_t;

typedef enum {
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE      = 0x01,
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_SerializationTargetType_t;

typedef struct {
  const char* fileName;
  const char* fileDirectory;
} QnnSystemProfile_SerializationTargetFile_t;

typedef struct {
  QnnSystemProfile_SerializationTargetType_t type;
  union {
    QnnSystemProfile_SerializationTargetFile_t file;
  };
} QnnSystemProfile_SerializationTarget_t;

typedef struct {
  const char* appName;
  const char* appVersion;
  const char* backendVersion;
} QnnSystemProfile_SerializationFileHeader_t;

typedef enum {
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_MAX_NUM_MESSAGES = 0,
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER = 1,
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_SerializationTargetConfigType_t;

typedef struct {
  QnnSystemProfile_SerializationTargetConfigType_t type;
  union {
    uint32_t maxNumMessages;
    QnnSystemProfile_SerializationFileHeader_t serializationHeader;
  };
} QnnSystemProfile_SerializationTargetConfig_t;

typedef enum {
  QNN_SYSTEM_PROFILE_VISIBILITY_PUBLIC  = 0,
  QNN_SYSTEM_PROFILE_VISIBILITY_PRIVATE = 1
} QnnSystemProfile_Visibility_t;

typedef enum {
  QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE = 0,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE = 1,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE = 2,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE_ASYNC = 3,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_CREATE_FROM_BINARY = 4,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_DEINIT = 5,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_CONTEXT_CREATE = 6,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_COMPOSE_GRAPHS = 7,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_EXECUTE_IPS = 8,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_GRAPH_COMPONENT = 9,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_BACKEND_LIB_LOAD = 10,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_APPLY_BINARY_SECTION = 11,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_FINALIZE = 12,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SIZE = 13,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY = 14,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SECTION_SIZE = 15,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE_TENSOR_UPDATES = 16,
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_UPDATE_BINARY_SECTION = 17
} QnnSystemProfile_MethodType_t;

typedef struct {
  uint64_t startTime;
  uint64_t stopTime;
  uint64_t startMem;
  uint64_t stopMem;
  QnnSystemProfile_MethodType_t methodType;
  QnnSystemProfile_Visibility_t visibility;
  const char* graphName;
} QnnSystemProfile_HeaderV1_t;

typedef enum {
  QNN_SYSTEM_PROFILE_EVENT_DATA = 0,
  QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA = 1,
  QNN_SYSTEM_PROFILE_EVENT_DATA_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_EventDataType_t;

typedef struct QnnSystemProfile_ProfileEventV1_t QnnSystemProfile_ProfileEventV1_t;

struct QnnSystemProfile_ProfileEventV1_t {
  QnnSystemProfile_EventDataType_t type;
  union {
    QnnProfile_EventData_t eventData;
    QnnProfile_ExtendedEventData_t extendedEventData;
  };
  QnnSystemProfile_ProfileEventV1_t* profileSubEventData;
  uint32_t numSubEvents;
};

typedef struct {
  QnnSystemProfile_HeaderV1_t header;
  QnnSystemProfile_ProfileEventV1_t* profilingEvents;
  uint32_t numProfilingEvents;
} QnnSystemProfile_ProfileDataV1_t;

typedef enum {
  QNN_SYSTEM_PROFILE_DATA_VERSION_1 = 0x01,
  QNN_SYSTEM_PROFILE_DATA_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_ProfileDataVersion_t;

typedef struct {
  QnnSystemProfile_ProfileDataVersion_t version;
  union {
    QnnSystemProfile_ProfileDataV1_t v1;
  };
} QnnSystemProfile_ProfileData_t;

#define QNN_SYSTEM_PROFILE_DATA_INIT                                                     \
  {                                                                                      \
    QNN_SYSTEM_PROFILE_DATA_VERSION_UNDEFINED,                                           \
    {                                                                                    \
      {0, 0, 0, 0, QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE,                                 \
       QNN_SYSTEM_PROFILE_VISIBILITY_PUBLIC, NULL, NULL, 0}                              \
    }                                                                                    \
  }

#endif

using QnnSystemProfile_createSerializationTargetFn_t = Qnn_ErrorHandle_t (*)(
    QnnSystemProfile_SerializationTarget_t,
    QnnSystemProfile_SerializationTargetConfig_t*,
    uint32_t,
    QnnSystemProfile_SerializationTargetHandle_t*);

using QnnSystemProfile_serializeEventDataFn_t = Qnn_ErrorHandle_t (*)(
    QnnSystemProfile_SerializationTargetHandle_t,
    const QnnSystemProfile_ProfileData_t**,
    uint32_t);

using QnnSystemProfile_freeSerializationTargetFn_t =
    Qnn_ErrorHandle_t (*)(QnnSystemProfile_SerializationTargetHandle_t);

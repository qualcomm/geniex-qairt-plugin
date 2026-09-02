// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Two different versions, both injected by CMake; the literals are fallbacks.

// QNN C API compiled against (from QAIRT SDK v2.36.1). The load-time floor:
// any runtime with C API >= this is accepted, i.e. SDK 2.36 and newer.
#ifndef GENIEX_QNN_API_VERSION
#define GENIEX_QNN_API_VERSION "2.27"
#endif

// Release of the runtime libs bundled under third-party/. A default, not a pin
// -- set GENIEX_QNN_LIB (or QnnRuntimeConfig::htp_dir) to load another.
#ifndef GENIEX_QAIRT_VERSION
#define GENIEX_QAIRT_VERSION "2.45"
#endif

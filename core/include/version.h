// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// QAIRT/QNN version this plugin's ABI is compiled against. No runtime libraries
// are bundled in this tree; the matching libs are supplied at load time via
// --qnn-lib / GENIEX_QNN_LIB. The value is injected by CMake from
// QAIRT_QNN_VERSION (see the top-level CMakeLists) so each per-version build
// reports its own ABI; the literal below is only a fallback for builds that do
// not define it.
#ifndef GENIEX_QAIRT_VERSION
#define GENIEX_QAIRT_VERSION "2.45"
#endif

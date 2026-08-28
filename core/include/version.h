// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Minimum QAIRT/QNN version this plugin's C API requires. This tree bundles a
// matching runtime under third-party/, copied to htp-files/ next to geniex_core
// at build and install time, so the default build needs no external SDK.
//
// This is a floor, not a pin: the C API negotiates at load time, so a newer
// runtime also works. The value is injected by CMake (see the top-level
// CMakeLists); the literal below is only a fallback for builds that do not
// define it.
#ifndef GENIEX_QAIRT_VERSION
#define GENIEX_QAIRT_VERSION "2.45"
#endif

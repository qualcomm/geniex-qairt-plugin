// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Two distinct versions. Both are injected by CMake (see the top-level
// CMakeLists); the literals below are only fallbacks for builds that do not
// define them.
//
// GENIEX_QNN_API_VERSION is the QNN C API this build compiles against, and the
// real floor on which runtimes will load: QnnApi.cpp accepts a provider when
// `QNN_API_VERSION_MINOR <= runtime minor`. C API 2.27 comes from QAIRT SDK
// v2.36.1, so any runtime from SDK 2.36 onward is accepted.
#ifndef GENIEX_QNN_API_VERSION
#define GENIEX_QNN_API_VERSION "2.27"
#endif

// GENIEX_QAIRT_VERSION is the release of the runtime libs this tree bundles
// under third-party/, copied to htp-files/ next to geniex_core at build and
// install time so the default build needs no external SDK.
//
// This is a default, not a pin: the C API negotiates at load time, so a newer
// runtime also works -- set GENIEX_QNN_LIB (or QnnRuntimeConfig::htp_dir) to
// load one instead of the bundled runtime.
#ifndef GENIEX_QAIRT_VERSION
#define GENIEX_QAIRT_VERSION "2.45"
#endif

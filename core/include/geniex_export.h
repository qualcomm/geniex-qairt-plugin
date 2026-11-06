// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// GENIEX_API  — public ABI of geniex_core.dll
// GENIEX_VLM_API — public ABI of geniex_vlm.dll
//
// CMake defines <target>_EXPORTS automatically when compiling each DLL.
// dllexport is used when building the owning library; dllimport when consuming it.
// On non-Windows platforms both macros expand to nothing.
#ifdef _WIN32
#ifdef geniex_core_EXPORTS
#define GENIEX_API __declspec(dllexport)
#else
#define GENIEX_API __declspec(dllimport)
#endif
#ifdef geniex_vlm_EXPORTS
#define GENIEX_VLM_API __declspec(dllexport)
#else
#define GENIEX_VLM_API __declspec(dllimport)
#endif
#ifdef geniex_tts_EXPORTS
#define GENIEX_TTS_API __declspec(dllexport)
#else
#define GENIEX_TTS_API __declspec(dllimport)
#endif
#else
#define GENIEX_API
#define GENIEX_VLM_API
#define GENIEX_TTS_API
#endif

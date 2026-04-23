//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Loop unrolling macros
#if defined(__clang__)
  #define PRAGMA_LOOP_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
  #define PRAGMA_LOOP_VECTORIZE _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
  #define PRAGMA_LOOP_VECTORIZE __pragma(loop(hint_parallel(4)))
#else
  #define PRAGMA_LOOP_VECTORIZE // Compiler does not support loop unrolling
#endif

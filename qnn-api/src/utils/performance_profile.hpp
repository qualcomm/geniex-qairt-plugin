//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_PERFORMANCE_PROFILE_HPP
#define QUALLA_PERFORMANCE_PROFILE_HPP

namespace qualla {

enum PerformanceProfile {
  PERFORMANCE_BURST                      = 10,
  PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE = 20,
  PERFORMANCE_HIGH_PERFORMANCE           = 30,
  PERFORMANCE_BALANCED                   = 40,
  PERFORMANCE_LOW_BALANCED               = 50,
  PERFORMANCE_HIGH_POWER_SAVER           = 60,
  PERFORMANCE_POWER_SAVER                = 70,
  PERFORMANCE_LOW_POWER_SAVER            = 80,
  PERFORMANCE_EXTREME_POWER_SAVER        = 90,
};

}  // namespace qualla

#endif  // QUALLA_PERFORMANCE_PROFILE_HPP

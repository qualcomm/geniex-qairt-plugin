//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <ostream>

#include "QnnHtpCommon.h"
#include "QnnHtpContext.h"
#include "QnnTypes.h"

std::ostream& operator<<(std::ostream& os, const QnnContext_Config_t& config);

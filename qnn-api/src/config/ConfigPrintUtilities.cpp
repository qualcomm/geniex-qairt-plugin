//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "ConfigPrintUtilities.hpp"

static inline std::string contextConfigOptionToString(QnnContext_ConfigOption_t option) {
  switch (option) {
    case QNN_CONTEXT_CONFIG_OPTION_CUSTOM:
      return "QNN_CONTEXT_CONFIG_OPTION_CUSTOM";
    case QNN_CONTEXT_CONFIG_OPTION_PRIORITY:
      return "QNN_CONTEXT_CONFIG_OPTION_PRIORITY";
    case QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING:
      return "QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING";
    case QNN_CONTEXT_CONFIG_OPTION_OEM_STRING:
      return "QNN_CONTEXT_CONFIG_OPTION_OEM_STRING";
    case QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH:
      return "QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH";
    case QNN_CONTEXT_CONFIG_ENABLE_GRAPHS:
      return "QNN_CONTEXT_CONFIG_ENABLE_GRAPHS";
    case QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT:
      return "QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT";
    case QNN_CONTEXT_CONFIG_PERSISTENT_BINARY:
      return "QNN_CONTEXT_CONFIG_PERSISTENT_BINARY";
    case QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY:
      return "QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY";
    case QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT:
      return "QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT";
    default:
      return "UNKNOWN";
  }
}

std::ostream& operator<<(std::ostream& os, const QnnContext_Config_t& config) {
  os << "Option: " << contextConfigOptionToString(config.option) << std::endl;
  switch (config.option) {
    default:
      break;
  }
  return os;
}

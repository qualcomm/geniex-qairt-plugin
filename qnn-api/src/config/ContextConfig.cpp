//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "ContextConfig.hpp"

void ContextConfig::serialize(qualla::ordered_json& json) const {
  const char* graphName = m_config.enableGraphs[0];
  switch (m_config.option) {
    case QNN_CONTEXT_CONFIG_OPTION_CUSTOM:
      json["option"]       = "QNN_CONTEXT_CONFIG_OPTION_CUSTOM";
      json["customConfig"] = qualla::ordered_json();  // Populated by subclasses
      break;
    case QNN_CONTEXT_CONFIG_OPTION_PRIORITY:
      json["option"]   = "QNN_CONTEXT_CONFIG_OPTION_PRIORITY";
      json["priority"] = m_config.priority;
      break;
    case QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING:
      json["option"]                        = "QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING";
      json["errorConfig"]["reportingLevel"] = m_config.errorConfig.reportingLevel;
      json["errorConfig"]["storageLimit"]   = m_config.errorConfig.storageLimit;
      break;
    case QNN_CONTEXT_CONFIG_OPTION_OEM_STRING:
      json["option"]    = "QNN_CONTEXT_CONFIG_OPTION_OEM_STRING";
      json["oemString"] = m_config.oemString;
      break;
    case QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH:
      json["option"]             = "QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH";
      json["asyncExeQueueDepth"] = m_config.asyncExeQueueDepth.depth;
      break;
    case QNN_CONTEXT_CONFIG_ENABLE_GRAPHS:
      json["option"] = "QNN_CONTEXT_CONFIG_ENABLE_GRAPHS";
      while (graphName != nullptr) {
        json["enableGraphs"].push_back(graphName++);
      }
      break;
    case QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT:
      json["option"]          = "QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT";
      json["memoryLimitHint"] = m_config.memoryLimitHint;
      break;
    case QNN_CONTEXT_CONFIG_PERSISTENT_BINARY:
      json["option"]             = "QNN_CONTEXT_CONFIG_PERSISTENT_BINARY";
      json["isPersistentBinary"] = m_config.isPersistentBinary;
      break;
    case QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY:
      json["option"]                  = "QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY";
      json["binaryCompatibilityType"] = m_config.binaryCompatibilityType;
      break;
    case QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT:
      json["option"]              = "QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT";
      json["isGraphInitDeferred"] = m_config.isGraphInitDeferred;
      break;
    default:
      json["option"] = "UNKNOWN";
      break;
  }
}

void ContextCustomHtpConfig::serialize(qualla::ordered_json& json) const {
  ContextConfig::serialize(json);
  qualla::ordered_json& customConfigJson = json["customConfig"];
  customConfigJson["backend"]            = "HTP";
  switch (m_customConfig.option) {
    case QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED:
      customConfigJson["option"] = "QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED";
      customConfigJson["weightSharingEnabled"] = m_customConfig.weightSharingEnabled;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS:
      customConfigJson["option"] = "QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS";
      customConfigJson["groupRegistration"]["firstGroupHandle"] =
          reinterpret_cast<uint64_t>(m_customConfig.groupRegistration.firstGroupHandle);
      customConfigJson["groupRegistration"]["maxSpillFillBuffer"] =
          m_customConfig.groupRegistration.maxSpillFillBuffer;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET:
      customConfigJson["option"] = "QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET";
      customConfigJson["fileReadMemoryBudgetInMb"] = m_customConfig.fileReadMemoryBudgetInMb;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED:
      customConfigJson["option"] = "QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED";
      customConfigJson["dspMemoryProfilingEnabled"] = m_customConfig.dspMemoryProfilingEnabled;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES:
      customConfigJson["option"]         = "QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES";
      customConfigJson["shareResources"] = m_customConfig.shareResources;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION:
      customConfigJson["option"]          = "QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION";
      customConfigJson["ioMemEstimation"] = m_customConfig.ioMemEstimation;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY:
      customConfigJson["option"]        = "QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY";
      customConfigJson["isPrepareOnly"] = m_customConfig.isPrepareOnly;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION:
      customConfigJson["option"]           = "QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION";
      customConfigJson["initAcceleration"] = m_customConfig.initAcceleration;
      break;
    case QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION:
      customConfigJson["option"] =
          "QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION";
      customConfigJson["skipValidationOnBinarySection"] =
          m_customConfig.skipValidationOnBinarySection;
      break;
    default:
      customConfigJson["option"] = "UNKNOWN";
      break;
  }
}

//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#if defined(LINUX_OE_HOST)
#include <fcntl.h>
#include <unistd.h>
#endif  // LINUX_OE_HOST

#include <chrono>
#include <sstream>
#if defined(__GNUC__) && !defined(__clang__)
#include <cstring>
#endif  // defined(__GNUC__) && !defined(__clang__)

#ifndef _WIN32
#include <sys/mman.h>
#endif  // _WIN32

#include "MmappedFile.hpp"
#include "QnnApi.hpp"
#include "dlwrap.hpp"
#ifdef SPILLFILL
#include "QnnHtpCommon.h"
#include "QnnHtpContext.h"
#endif  // SPILLFILL

LogCallback userCb = nullptr;
void registerUserCb(LogCallback l) { userCb = l; }

static std::vector<std::shared_ptr<mmapped::File>> mmappedFilesVec;

QnnApi::~QnnApi() {
  QNN_DEBUG("Freeing Graphs");
  if (true != freeGraphs()) {
    QNN_DEBUG("Could not free Graphs");
  }

  if (m_scorer && !freeGraphInfo(m_scorer)) {
    QNN_DEBUG("Could not free scorer graph");
  }

  // Free context if not already done
  if (m_isContextCreated) {
    QNN_DEBUG("Freeing Context");
    if (true != freeContext()) {
      QNN_DEBUG("Could not free context");
    }
  }

  if (m_profileBackendHandle) {
    QNN_DEBUG("Freeing profile handle");
    if (QNN_PROFILE_NO_ERROR != m_qnnInterface.profileFree(m_profileBackendHandle))
      QNN_ERROR("Could not free QNN HTP backend profile handle.");
  }

  QNN_DEBUG("Freeing Device");
  if (getDeviceStatus()) {
    if (true != freeDevice()) {
      QNN_ERROR("Device Free failure");
    }
  }

  // Terminate backend
  if (m_isBackendInitialized) {
    QNN_DEBUG("Terminating Backend");
    if (true != terminateBackend()) {
      QNN_DEBUG("Could not terminate backend");
    }
  }

  // For HTP backend, skip logging termination as the backend library
  // handles logger cleanup during its own shutdown to avoid registry conflicts
  // when multiple QnnApi instances share the same backend library
  if (m_backendId != QNN_BACKEND_ID_HTP) {
    QNN_DEBUG("Terminating Logging");
    if (m_isLogInitialized) {
      terminateLog();
    }
    m_isLogInitialized = false;
  }

  // Skip dlclose for HTP because it runs its own cleanup routines later.
  if (m_backendLibraryHandle && (m_backendId != QNN_BACKEND_ID_HTP)) {
    QNN_DEBUG("Closing Backend Lib Handle");
    dlclose(m_backendLibraryHandle);
  }

  if (m_libModelHandle) {
    QNN_DEBUG("Closing Model Lib Handle");
    dlclose(m_libModelHandle);
  }

  if (!m_contextBinBuffersToBeCleared.empty()) {
    if (m_mmapContextBins) {
      mmappedFilesVec.clear();
    } else {
      for (auto& [buffer, bufferSize] : m_contextBinBuffersToBeCleared) {
        QNN_DEBUG("Free context bin buffer %p of size %lu", buffer, bufferSize);
        delete[] buffer;
      }
    }
  }
}

bool QnnApi::getContextConfigs(QnnContext_Config_t*** configs,
                               uint32_t& contextConfigCount,
                               bool graphSwitching,
                               const std::vector<std::string>& execSelectGraphs,
                               bool loadSelectGraphs) {
  std::vector<QnnContext_Config_t*> contextConfigPtrsVec;
  const char** graphNames = nullptr;

  if (loadSelectGraphs && !execSelectGraphs.empty()) {
    graphNames = (const char**)malloc(sizeof(const char*) * (execSelectGraphs.size() + 1));
    for (size_t i = 0; i < execSelectGraphs.size(); ++i) {
      graphNames[i] = execSelectGraphs[i].c_str();
    }

    graphNames[execSelectGraphs.size()] = nullptr;  // NULL termination
    contextConfigPtrsVec.push_back((QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t)));
    contextConfigPtrsVec.back()->option =
        QnnContext_ConfigOption_t::QNN_CONTEXT_CONFIG_ENABLE_GRAPHS;
    contextConfigPtrsVec.back()->enableGraphs = graphNames;
  }

  if (graphSwitching) {
    contextConfigPtrsVec.push_back((QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t)));
    contextConfigPtrsVec.back()->option =
        QnnContext_ConfigOption_t::QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT;
    contextConfigPtrsVec.back()->memoryLimitHint = 1024;

    contextConfigPtrsVec.push_back((QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t)));
    contextConfigPtrsVec.back()->option =
        QnnContext_ConfigOption_t::QNN_CONTEXT_CONFIG_PERSISTENT_BINARY;
    contextConfigPtrsVec.back()->isPersistentBinary = 1;
  }

  contextConfigCount = contextConfigPtrsVec.size();

  QnnContext_Config_t** contextConfigPtrs =
      (QnnContext_Config_t**)malloc(contextConfigCount * sizeof(QnnContext_Config_t*));

  if (nullptr == contextConfigPtrs) {
    QNN_ERROR("Could not allocate memory for allContextConfigs");
    return false;
  }

  for (size_t i = 0; i < contextConfigCount; i++) {
    contextConfigPtrs[i] = contextConfigPtrsVec[i];
  }

  *configs = contextConfigPtrs;

  return true;
}

bool QnnApi::getContextConfigs(ConfigList<QnnContext_Config_t>& configList,
                               bool graphSwitching,
                               const std::vector<std::string>& execSelectGraphs,
                               bool loadSelectGraphs) {
  if (loadSelectGraphs && !execSelectGraphs.empty()) {
    configList.add(std::make_unique<ContextConfig>(ContextEnableGraphsConfig(execSelectGraphs)));
  }

  if (graphSwitching) {
    configList.add(std::make_unique<ContextConfig>(ContextMemoryLimitHintConfig(1024)));
    configList.add(std::make_unique<ContextConfig>(ContextPersistentBinaryConfig(true)));
  }

  return true;
}

bool QnnApi::mergeAllContextConfigs(QnnContext_Config_t*** allCustomContextConfigs,
                                    QnnContext_Config_t** customConfigs,
                                    QnnContext_Config_t**& contextConfigs,
                                    uint32_t customConfigCount,
                                    uint32_t contextConfigCount) {
  QnnContext_Config_t** allContextConfigs{nullptr};
  if (contextConfigCount + customConfigCount > 0) {
    allContextConfigs = (QnnContext_Config_t**)calloc((contextConfigCount + customConfigCount + 1),
                                                      sizeof(QnnContext_Config_t*));
    if (nullptr == allContextConfigs) {
      QNN_ERROR("Could not allocate memory for allContextConfigs");
      return false;
    }
    for (size_t cnt = 0; cnt < contextConfigCount; cnt++) {
      allContextConfigs[cnt] = contextConfigs[cnt];
    }
    if (contextConfigs) {
      free(contextConfigs);
      contextConfigs = nullptr;
    }
    for (size_t cnt = 0; cnt < customConfigCount; cnt++) {
      allContextConfigs[cnt + contextConfigCount] = customConfigs[cnt];
    }
  }
  *allCustomContextConfigs = allContextConfigs;

  return true;
}

bool QnnApi::freeContextConfigs(QnnContext_Config_t** contextConfigs, uint32_t contextConfigCount) {
  if (contextConfigs) {
    for (size_t i = 0; i < contextConfigCount; i++) {
      if (contextConfigs[i]->option == QNN_CONTEXT_CONFIG_ENABLE_GRAPHS) {
        free((const char**)contextConfigs[i]->enableGraphs);
      }
      free(contextConfigs[i]);
    }
    free(contextConfigs);
  }

  return true;
}

bool QnnApi::setGraphConfigsBeforeExecute(Qnn_GraphHandle_t graphHandle,
                                          QnnGraph_Config_t** graphConfigs,
                                          uint32_t configCount) {
  if (!graphConfigs || configCount == 0u) {
    QNN_ERROR("No graph configs to set");
    return false;
  }

  std::vector<const QnnGraph_Config_t*> graphConfigsPointers(configCount + 1, nullptr);
  for (size_t idx = 0u; idx < configCount; idx++) {
    graphConfigsPointers[idx] = graphConfigs[idx];
  }
  if (QNN_SUCCESS != m_qnnInterface.graphSetConfig(graphHandle, graphConfigsPointers.data())) {
    QNN_ERROR("Failed to set graph configs.");
    return false;
  }

  return true;
}

bool QnnApi::getQnnInterface(std::string backendPath) {
  QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};

  m_backendLibraryHandle = dlopen(backendPath.c_str(), RTLD_NOW);
  if (nullptr == m_backendLibraryHandle) {
    QNN_ERROR("Unable to load backend. dlerror(): %s", dlerror());
    return false;
  }

  // Get QNN Interface
  getInterfaceProviders =
      (QnnInterfaceGetProvidersFn_t)dlsym(m_backendLibraryHandle, "QnnInterface_getProviders");
  if (nullptr == getInterfaceProviders) {
    return false;
  }

  uint32_t numProviders{0};
  QnnInterface_t** interfaceProviders{nullptr};
  if (QNN_SUCCESS !=
      getInterfaceProviders((const QnnInterface_t***)&interfaceProviders, &numProviders)) {
    QNN_ERROR("Failed to get interface providers.");
    return false;
  }

  if (nullptr == interfaceProviders) {
    QNN_ERROR("Failed to get interface providers: null interface providers received.");
    return false;
  }
  if (0u == numProviders) {
    QNN_ERROR("Failed to get interface providers: 0 interface providers.");
    return false;
  }

  bool foundValidInterface{false};
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    const Qnn_ApiVersion_t& apiVersion = interfaceProviders[pIdx]->apiVersion;
    if ((QNN_API_VERSION_MAJOR == apiVersion.coreApiVersion.major) &&
        (QNN_API_VERSION_MINOR <= apiVersion.coreApiVersion.minor)) {
      foundValidInterface = true;
      m_qnnInterface      = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
      m_backendId         = interfaceProviders[pIdx]->backendId;
      break;
    }
  }

  if (!foundValidInterface) {
    QNN_ERROR("Unable to find a compatible QNN API interface.");
    QNN_ERROR("Expected API version %u.%u.%u or later",
              QNN_API_VERSION_MAJOR,
              QNN_API_VERSION_MINOR,
              QNN_API_VERSION_PATCH);
    std::stringstream availableVersions;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      const Qnn_ApiVersion_t& apiVersion = interfaceProviders[pIdx]->apiVersion;
      availableVersions << apiVersion.coreApiVersion.major << "." << apiVersion.coreApiVersion.minor
                        << "." << apiVersion.coreApiVersion.patch << ", ";
    }
    // Remove trailing comma
    availableVersions.seekp(-2, std::ios_base::cur);
    availableVersions << '\0';
    QNN_ERROR("Available API versions: %s", availableVersions.str().c_str());
    m_backendLibraryHandle = nullptr;
    return false;
  }

  return true;
}

bool QnnApi::getQnnSystemInterface(std::string systemLibraryPath) {
  QnnSystemInterfaceGetProvidersFn_t getSystemInterfaceProviders{nullptr};

  void* systemLibraryHandle = dlopen(systemLibraryPath.c_str(), RTLD_NOW);
  if (nullptr == systemLibraryHandle) {
    QNN_ERROR("Unable to load system library. dlerror(): %s", dlerror());
    return false;
  }

  // Get QNN System Interface
  getSystemInterfaceProviders = (QnnSystemInterfaceGetProvidersFn_t)dlsym(
      systemLibraryHandle, "QnnSystemInterface_getProviders");
  if (nullptr == getSystemInterfaceProviders) {
    return false;
  }

  uint32_t numProviders{0};
  QnnSystemInterface_t** systemInterfaceProviders{nullptr};
  if (QNN_SUCCESS != getSystemInterfaceProviders(
                         (const QnnSystemInterface_t***)&systemInterfaceProviders, &numProviders)) {
    QNN_ERROR("Failed to get system interface providers.");
    return false;
  }
  if (nullptr == systemInterfaceProviders) {
    QNN_ERROR(
        "Failed to get system interface providers: null system interface providers received.");
    return false;
  }
  if (0 == numProviders) {
    QNN_ERROR("Failed to get system interface providers: 0 system interface providers.");
    return false;
  }

  bool foundValidSystemInterface{false};
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    const Qnn_Version_t& systemApiVersion = systemInterfaceProviders[pIdx]->systemApiVersion;
    if (QNN_SYSTEM_API_VERSION_MAJOR == systemApiVersion.major &&
        QNN_SYSTEM_API_VERSION_MINOR <= systemApiVersion.minor) {
      foundValidSystemInterface = true;
      m_qnnSystemInterface      = systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!foundValidSystemInterface) {
    QNN_ERROR("Unable to find a valid system interface.");
    return false;
  }

  return true;
}

bool QnnApi::loadModel(std::string model_path) {
  const char* dlsym_error;

  dlerror();
  m_libModelHandle = dlopen(model_path.c_str(), RTLD_NOW);
  if (nullptr == m_libModelHandle) {
    QNN_ERROR("Unable to load model. dlerror(): %s", dlerror());
    return false;
  }

  // Currently model Prefix is fixed. If model was prepared with
  // custom prefix, we need to change this.
  std::string modelPrefix = "QnnModel";

  std::string modelPrepareFunc = modelPrefix + "_composeGraphs";
  m_composeGraphsFnHandle =
      (ComposeGraphsFnHandleType_t)dlsym(m_libModelHandle, modelPrepareFunc.c_str());
  dlsym_error = dlerror();
  if (dlsym_error || nullptr == m_composeGraphsFnHandle) {
    m_composeGraphsFnHandle           = nullptr;
    std::string genaiModelPrepareFunc = "QnnModel_GenAI_composeGraphs";
    m_genaiComposeGraphsFnHandle =
        (GenAIComposeGraphsFnHandleType_t)dlsym(m_libModelHandle, genaiModelPrepareFunc.c_str());
    dlsym_error = dlerror();
    if (dlsym_error || nullptr == m_genaiComposeGraphsFnHandle) {
      QNN_ERROR("Did not find QnnModel_composeGraph function: %s", dlsym_error);
      return false;
    }
  }

  std::string modelFreeFunc = modelPrefix + "_freeGraphsInfo";
  m_freeGraphInfoFnHandle =
      (FreeGraphInfoFnHandleType_t)dlsym(m_libModelHandle, modelFreeFunc.c_str());
  dlsym_error = dlerror();
  if (dlsym_error || nullptr == m_freeGraphInfoFnHandle) {
    QNN_ERROR("Did not find QnnModel_freeGraphsInfo function: %s", dlsym_error);
    return false;
  }

  return true;
}

void userLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
  uint32_t i = static_cast<uint32_t>(level);
  auto cb    = userCb;
  cb(fmt, i, timestamp, args);
}

bool QnnApi::initializeLogging(const QnnLog_Level_t& logLevel,
                               bool debug_qnn,
                               LogCallback userCallback) {
  // initialize logging in the backend
  if (nullptr != m_qnnInterface.logCreate) {
    QnnLog_Callback_t logCallback;
    if (userCallback) registerUserCb(userCallback);
    if (debug_qnn) {
      logCallback = userLogCallback;
    } else {
      logCallback = QnnApi::emptyLogCallback;
    }
    QNN_DEBUG("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
              logCallback,
              logLevel);
    if (QNN_SUCCESS != m_qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
      QNN_WARN("Unable to initialize logging in the backend.");
    }
    m_isLogInitialized = true;
  } else {
    QNN_WARN("Logging not available in the backend.");
    return true;
  }

  return true;
}

void QnnApi::terminateLog() {
  // Terminate logging in the backend
  if (nullptr != m_qnnInterface.logFree && nullptr != m_logHandle) {
    if (QNN_SUCCESS != m_qnnInterface.logFree(m_logHandle)) {
      QNN_WARN("Unable to terminate logging in the backend.");
    }
  }
}

bool QnnApi::initializeBackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                         qnn::tools::netrun::PerfProfile parsedPerfProfile,
                                         bool debug_qnn,
                                         QnnLog_Level_t qnnLogLevel) {
  if (backendExtensionsConfig.sharedLibraryPath.empty() &&
      backendExtensionsConfig.configFilePath.empty()) {
    // Backend extensions are not in use, return success.
    return true;
  }
  try {
    m_backendExtensions.reset(
        new BackendExtensions(backendExtensionsConfig,
                              m_backendLibraryHandle,
                              parsedPerfProfile,
                              debug_qnn,
                              debug_qnn ? userLogCallback : QnnApi::emptyLogCallback,
                              qnnLogLevel));
  } catch (const std::exception& e) {
    (void)e;
    QNN_ERROR(e.what.c_str());
    m_backendExtensions = nullptr;
    return false;
  }
  if (nullptr == m_backendExtensions) {
    QNN_ERROR("Unable to create backend extensions object.");
    return false;
  }

  return true;
}

// Initialize a QnnBackend.
bool QnnApi::initializeBackend() {
  if (nullptr == m_qnnInterface.backendCreate) {
    QNN_ERROR("BackendCreate API is not supported for this backend");
    return false;
  }

  QnnBackend_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeBackendInitialize(&customConfigs,
                                                                   &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeBackendInitialize()");
      return false;
    }
  }
  QnnBackend_Config_t** allBackendConfigs{nullptr};
  if ((m_backendConfigCount + customConfigCount) > 0) {
    allBackendConfigs = (QnnBackend_Config_t**)calloc(
        (m_backendConfigCount + customConfigCount + 1), sizeof(QnnBackend_Config_t*));
    if (nullptr == allBackendConfigs) {
      QNN_ERROR("Could not allocate memory for allBackendConfigs");
      return false;
    }
    for (size_t cnt = 0; cnt < m_backendConfigCount; cnt++) {
      allBackendConfigs[cnt] = m_backendConfigs[cnt];
    }
    for (size_t cnt = 0; cnt < customConfigCount; cnt++) {
      allBackendConfigs[cnt + m_backendConfigCount] = customConfigs[cnt];
    }
  }

  Qnn_ErrorHandle_t errCode = m_qnnInterface.backendCreate(
      m_logHandle, (const QnnBackend_Config_t**)allBackendConfigs, &m_backendHandle);
  if (QNN_SUCCESS != errCode) {
    QNN_ERROR("Could not initialize backend due to error = %llu",
              static_cast<unsigned long long>(errCode));
    if (allBackendConfigs) {
      free(allBackendConfigs);
    }
    return false;
  }
  QNN_DEBUG("Initialize Backend Returned Status = %llu", static_cast<unsigned long long>(errCode));

  m_isBackendInitialized = true;
  if (allBackendConfigs) {
    free(allBackendConfigs);
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterBackendInitialize()) {
      QNN_ERROR("Extensions Failure in afterBackendInitialize()");
      return false;
    }
  }

  return true;
}

// Terminate the backend after done.
bool QnnApi::terminateBackend() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeBackendTerminate()) {
      QNN_ERROR("Extensions Failure in beforeBackendTerminate()");
      return false;
    }
  }
  // Terminate backend
  if (m_isBackendInitialized && nullptr != m_qnnInterface.backendFree) {
    QNN_DEBUG("Freeing backend");
    if (QNN_BACKEND_NO_ERROR != m_qnnInterface.backendFree(m_backendHandle)) {
      QNN_ERROR("Could not free backend");
    }
  }
  m_isBackendInitialized = false;

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterBackendTerminate()) {
      QNN_ERROR("Extensions Failure in afterBackendTerminate()");
      return false;
    }
  }

  return true;
}

bool QnnApi::createDevice() {
  QnnDevice_Config_t** deviceConfigs{nullptr};
  uint32_t configCount{0};

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeCreateDevice(&deviceConfigs, &configCount)) {
      QNN_ERROR("Extensions Failure in beforeCreateDevice()");
      return false;
    }
  }
  std::vector<const QnnDevice_Config_t*> deviceConfigPointers(configCount + 1, nullptr);
  for (size_t idx = 0u; idx < configCount; idx++) {
    deviceConfigPointers[idx] = deviceConfigs[idx];
  }
  if (nullptr != m_qnnInterface.deviceCreate) {
    auto qnnStatus =
        m_qnnInterface.deviceCreate(m_logHandle, deviceConfigPointers.data(), &m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus) {
      if (QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE == qnnStatus) {
        QNN_WARN("Device feature unsupported");
      } else {
        QNN_ERROR("Failed to create device: %lu", (unsigned long)qnnStatus);
        return false;
      }
    }
  }
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterCreateDevice()) {
      QNN_ERROR("Extensions Failure in afterCreateDevice()");
      return false;
    }
  }
  return true;
}

bool QnnApi::freeDevice() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeFreeDevice()) {
      QNN_ERROR("Extensions Failure in beforeFreeDevice()");
      return false;
    }
  }
  if (nullptr != m_qnnInterface.deviceFree) {
    auto qnnStatus = m_qnnInterface.deviceFree(m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus) {
      if (QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE == qnnStatus) {
        QNN_WARN("Device feature unsupported");
      } else {
        QNN_ERROR("Failed to free device: %lu", (unsigned long)qnnStatus);
        return false;
      }
    }
  }
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterFreeDevice()) {
      QNN_ERROR("Extensions Failure in afterfreeDevice()");
      return false;
    }
  }
  return true;
}

// Create a Context in a backend.
bool QnnApi::createContext() {
  QnnContext_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeContextCreate(&customConfigs,
                                                               &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeContextCreate()");
      return false;
    }
  }

  QnnContext_Config_t** contextConfigs = nullptr;
  uint32_t contextConfigCount          = 0;
  if (true != getContextConfigs(&contextConfigs, contextConfigCount)) {
    QNN_ERROR("Couldn't populate context configs");
    return false;
  }

  QnnContext_Config_t** allContextConfigs{nullptr};
  if (true != mergeAllContextConfigs(&allContextConfigs,
                                     customConfigs,
                                     contextConfigs,
                                     customConfigCount,
                                     contextConfigCount)) {
    QNN_ERROR("Error merging custom and context configs");
    return false;
  }

  uint32_t totalConfigCount = customConfigCount + contextConfigCount;
  Qnn_ContextHandle_t contextHandle{nullptr};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnInterface.contextCreate(m_backendHandle,
                                   nullptr,
                                   (const QnnContext_Config_t**)allContextConfigs,
                                   &contextHandle)) {
    QNN_ERROR("Could not create context");
    if (true != freeContextConfigs(allContextConfigs, totalConfigCount)) {
      QNN_ERROR("Couldn't free allContextConfigs");
      return false;
    }
    allContextConfigs = nullptr;

    return false;
  }

  m_contextVec.push_back(contextHandle);
  m_isContextCreated = true;
  if (true != freeContextConfigs(allContextConfigs, totalConfigCount)) {
    QNN_ERROR("Couldn't free allContextConfigs");
    return false;
  }
  allContextConfigs = nullptr;

  if (true != freeContextConfigs(contextConfigs, contextConfigCount)) {
    QNN_ERROR("Couldn't free context configs");
    return false;
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterContextCreate()) {
      QNN_ERROR("Extensions Failure in afterContextCreate()");
      return false;
    }
  }

  return true;
}

bool QnnApi::freeCurrentGraph(std::string graphName) {
  auto graphInfo = m_graphsInfo[m_graphNameToIndex[graphName]];
  if (!freeGraphInfo(graphInfo)) {
    QNN_ERROR("Could not free graphInfo");
    return false;
  }
  m_graphsInfo[m_graphNameToIndex[graphName]] = nullptr;
  m_graphNameToIndex.erase(graphName);
  return true;
}

bool QnnApi::freeCurrentContext(std::string graphName) {
  auto contextHandle = m_contextVec[m_graphNameToContextIdx[graphName]];
  if (QNN_CONTEXT_NO_ERROR != m_qnnInterface.contextFree(contextHandle, nullptr)) {
    QNN_ERROR("Could not free contexeeet");
    return false;
  }
  m_contextVec[m_graphNameToContextIdx[graphName]] = (Qnn_ContextHandle_t) nullptr;
  m_graphNameToContextIdx.erase(graphName);
  return true;
}

// Free context after done.
bool QnnApi::freeContext() {
  // beforeContextFree/afterContextFree are only safe to call when the context
  // was created via createContext()+afterContextCreate(). For the binary-load
  // path (createFromBinaryHtp / async), QnnHtpNetRunExtensions crashes in
  // beforeContextFree() because the extension's internal context handle list
  // is not populated by afterCreateFromBinary(). Skip those hooks entirely.
  if (!m_contextCreatedFromBinary &&
      nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeContextFree()) {
      QNN_ERROR("Extensions Failure in beforeContextFree()");
      return false;
    }
  }
  for (const auto& context : m_contextVec) {
    if (context && (QNN_CONTEXT_NO_ERROR != m_qnnInterface.contextFree(context, nullptr))) {
      QNN_ERROR("Could not free context");
      return false;
    }
  }
  m_isContextCreated = false;

  if (!m_contextCreatedFromBinary &&
      nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterContextFree()) {
      QNN_ERROR("Extensions Failure in afterContextFree()");
      return false;
    }
  }

  return true;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in graphsInfo and graphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
bool QnnApi::composeGraphs(std::vector<GraphConfigs> graphConfigs) {
  qnn_wrapper_api::GraphConfigInfo_t** customConfigs{nullptr};
  uint32_t customConfigGraphsCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeComposeGraphs(&customConfigs,
                                                               &customConfigGraphsCount)) {
      QNN_ERROR("Extensions Failure in beforeComposeGraphs()");
      return false;
    }
  }

  std::map<std::string, std::vector<QnnGraph_Config_t*>> graphConfigsPointers;
  if (!graphConfigs.empty()) {
    for (auto const& inputGraphConfig : graphConfigs) {
      // Only reset the memory for this graph, if it has not previously been populated with
      // something
      if (graphConfigsPointers.find(inputGraphConfig.graphName) == graphConfigsPointers.end()) {
        graphConfigsPointers[inputGraphConfig.graphName] = std::vector<QnnGraph_Config_t*>();
        graphConfigsPointers[inputGraphConfig.graphName].reserve(s_graphConfigsReserveCount);
      }
      if (inputGraphConfig.priorityPresent) {
        QnnGraph_Config_t* newGraphConfig = (QnnGraph_Config_t*)malloc(sizeof(QnnGraph_Config_t));
        newGraphConfig->option            = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
        newGraphConfig->priority          = inputGraphConfig.priority;
        graphConfigsPointers[inputGraphConfig.graphName].push_back(newGraphConfig);
      }
    }
  }

  if (customConfigs != nullptr && customConfigGraphsCount > 0) {
    for (size_t gIdx = 0; gIdx < customConfigGraphsCount; gIdx++) {
      auto configPtr = customConfigs[gIdx]->graphConfigs;
      if (*configPtr &&
          (!customConfigs[gIdx]->graphName || strlen(customConfigs[gIdx]->graphName) == 0)) {
        QNN_ERROR("Graph configs specified without a graph name in the backend extensions.");
        return false;
      }
      if (customConfigs[gIdx]->graphName && strlen(customConfigs[gIdx]->graphName) > 0 &&
          *configPtr) {
        if (graphConfigsPointers.find(customConfigs[gIdx]->graphName) ==
            graphConfigsPointers.end()) {
          graphConfigsPointers[customConfigs[gIdx]->graphName] = std::vector<QnnGraph_Config_t*>();
          graphConfigsPointers[customConfigs[gIdx]->graphName].reserve(s_graphConfigsReserveCount);
        }
        while (*configPtr) {
          graphConfigsPointers[customConfigs[gIdx]->graphName].push_back(
              (QnnGraph_Config_t*)*configPtr);
          configPtr++;
        }
      }
    }
  }

  auto graphConfigsInfo = (qnn_wrapper_api::GraphConfigInfo_t**)calloc(
      graphConfigsPointers.size(), sizeof(qnn_wrapper_api::GraphConfigInfo_t*));
  size_t graphIdx{0};
  for (auto const& graphConfig : graphConfigsPointers) {
    if (graphConfigsInfo && graphConfig.second.size() > 0) {
      graphConfigsInfo[graphIdx] =
          (qnn_wrapper_api::GraphConfigInfo_t*)malloc(sizeof(qnn_wrapper_api::GraphConfigInfo_t));
      graphConfigsInfo[graphIdx]->graphName    = (char*)graphConfig.first.c_str();
      graphConfigsInfo[graphIdx]->graphConfigs = (const QnnGraph_Config_t**)calloc(
          graphConfig.second.size() + 1, sizeof(QnnGraph_Config_t*));
      for (size_t cnt = 0; cnt < graphConfig.second.size(); cnt++) {
        graphConfigsInfo[graphIdx]->graphConfigs[cnt] = graphConfig.second[cnt];
      }
    }
    graphIdx++;
  }

  int status = m_composeGraphsFnHandle(m_backendHandle,
                                       m_qnnInterface,
                                       m_contextVec[0],
                                       (const qnn_wrapper_api::GraphConfigInfo_t**)graphConfigsInfo,
                                       graphConfigsPointers.size(),
                                       &m_graphsInfo,
                                       &m_graphsCount,
                                       m_DebugModeRequested,
                                       nullptr,
                                       QnnLog_Level_t::QNN_LOG_LEVEL_VERBOSE);

  if (graphConfigsInfo) {
    for (size_t gIdx = 0; gIdx < graphConfigsPointers.size(); gIdx++) {
      if (graphConfigsInfo[gIdx]) {
        if (graphConfigsInfo[gIdx]->graphConfigs) {
          free(graphConfigsInfo[gIdx]->graphConfigs);
          graphConfigsInfo[gIdx]->graphConfigs = nullptr;
          graphConfigsInfo[gIdx]->graphName    = nullptr;
        }
        free(graphConfigsInfo[gIdx]);
        graphConfigsInfo[gIdx] = nullptr;
      }
    }
    free(graphConfigsInfo);
  }

  for (auto const& graphConfig : graphConfigsPointers) {
    for (size_t cnt = 0; cnt < graphConfig.second.size(); cnt++) {
      if (graphConfig.second[cnt]) {
        free(graphConfig.second[cnt]);
      }
    }
    // graphConfig.second.clear();
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterComposeGraphs()) {
      QNN_ERROR("Extensions Failure in afterComposeGraphs()");
      return false;
    }
  }

  if (0 != status) {
    QNN_ERROR("Failed in composeGraphs()");
    return false;
  }

  // For now, we only handle 1 graph for this framework.
  if (m_graphsCount != 1) {
    QNN_ERROR("Only one graph is supported by framework");
    return false;
  }

  return true;
}

bool QnnApi::composeGraphs(std::vector<GraphConfigs> graphConfigs,
                           uint32_t* inputDim,
                           uint32_t inputRank,
                           uint32_t* outputDim,
                           uint32_t outputRank,
                           uint32_t* kvDim,
                           uint32_t kvRank,
                           uint32_t* kvScaleDim,
                           Qnn_Param_t* params,
                           uint32_t numParams) {
  std::string model_name = "qnn_model";
  static int model_id    = 1;
  qnn_wrapper_api::GraphInfo_t** graphsInfo;
  uint32_t graphsCount = 0;

  qnn_wrapper_api::ModelError status =
      m_genaiComposeGraphsFnHandle(m_backendHandle,
                                   m_qnnInterface,
                                   m_contextVec.back(),
                                   nullptr,
                                   0,
                                   inputDim,
                                   inputRank,
                                   outputDim,
                                   outputRank,
                                   kvDim,
                                   kvRank,
                                   kvScaleDim,
                                   params,
                                   numParams,
                                   (model_name + std::to_string(model_id)).c_str(),
                                   &graphsInfo,
                                   &graphsCount,
                                   m_DebugModeRequested,
                                   nullptr,
                                   QnnLog_Level_t::QNN_LOG_LEVEL_VERBOSE);

  model_id++;
  graphCountPerContext = graphsCount;

  std::vector<qnn_wrapper_api::GraphInfo_t*> graphsInfoVec(m_graphsInfo,
                                                           m_graphsInfo + m_graphsCount);
  free(m_graphsInfo);

  for (size_t graph_idx = 0; graph_idx < graphsCount; graph_idx++) {
    m_contextMap[graphsInfo[graph_idx]] = m_contextVec.back();
    m_graphNameToContextIdx[std::string(graphsInfo[graph_idx]->graphName)] =
        m_contextVec.size() - 1;
    graphsInfoVec.push_back(graphsInfo[graph_idx]);
  }

  free(graphsInfo);
  m_graphsCount += graphsCount;
  m_graphsInfo =
      (qnn_wrapper_api::GraphInfo_t**)malloc(m_graphsCount * sizeof(qnn_wrapper_api::GraphInfo_t*));
  for (size_t graph_idx = 0; graph_idx < m_graphsCount; graph_idx++) {
    m_graphsInfo[graph_idx] = graphsInfoVec[graph_idx];
  }

  if (status == qnn_wrapper_api::MODEL_NO_ERROR) {
    return true;
  }

  return false;
}

bool QnnApi::finalizeCpuGraphs() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeGraphFinalize()) {
      QNN_ERROR("Extensions Failure in beforeGraphFinalize()");
      return false;
    }
  }

  for (size_t graphIdx = (m_graphsCount - graphCountPerContext); graphIdx < m_graphsCount;
       graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnInterface.graphFinalize(m_graphsInfo[graphIdx]->graph, nullptr, nullptr)) {
      return false;
    }

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterGraphFinalize()) {
      QNN_ERROR("Extensions Failure in afterGraphFinalize()");
      return false;
    }
  }

  return true;
}

bool QnnApi::finalizeGraphs() {
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeGraphFinalize()) {
      QNN_ERROR("Extensions Failure in beforeGraphFinalize()");
      return false;
    }
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnInterface.graphFinalize(m_graphsInfo[graphIdx]->graph, nullptr, nullptr)) {
      return false;
    }

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterGraphFinalize()) {
      QNN_ERROR("Extensions Failure in afterGraphFinalize()");
      return false;
    }
  }

  return true;
}

bool QnnApi::freeGraphs() {
  freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  if (m_graphsInfo) {
    free(m_graphsInfo);
  }
  m_graphsInfo  = nullptr;
  m_graphsCount = 0;
  return true;
}

bool QnnApi::mapAndGetContextBinaryInfo(const bool useMmap,
                                        std::shared_ptr<uint8_t>& buffer,
                                        const std::string& binaryPath,
                                        const uint64_t bufferSize,
                                        const size_t contextIdx,
                                        const bool graphSwitching,
                                        QnnSystemContext_Handle_t sysCtxHandle,
                                        const QnnSystemContext_BinaryInfo_t** binaryInfo) {
  if (useMmap) {
    // TODO: Find out why MlgInfra Mmapped doesn't work for Linux OE targets
#if defined(LINUX_OE_HOST)
    // Read binary file with mmapped syscall
    int fd              = open(binaryPath.c_str(), O_RDONLY);
    void* mmappedBuffer = mmap(nullptr, bufferSize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (madvise(mmappedBuffer, bufferSize, MADV_NOHUGEPAGE)) {
      QNN_WARN("Failed to advise OS on memory usage");
    }

    buffer = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(mmappedBuffer),
                                      [graphSwitching, bufferSize](uint8_t* ptr) {
                                        if (!graphSwitching) {
                                          munmap(ptr, bufferSize);
                                        }
                                      });
#else
    // Memory mapped binary allocation
    auto mmf = std::make_shared<mmapped::File>(binaryPath);
    if (!(*mmf)) {
      QNN_ERROR("Failed to allocate memory mapped region for context index = %zu", contextIdx);
    }

#ifndef _WIN32
    // Note: There is no Windows-equivalent of madvise
    if (!mmf->adviseRange(0, bufferSize, MADV_NOHUGEPAGE)) {
      QNN_ERROR("Failed to advise OS on memory usage err: %s", strerror(errno));
      return false;
    }
#endif  // _WIN32

    // Since mmapped::File uses RAII, need to keep shared ptr alive in mmappedFilesVec
    const size_t mmfIndex = mmappedFilesVec.size();
    mmappedFilesVec.push_back(mmf);

    buffer = std::shared_ptr<uint8_t>(
        static_cast<uint8_t*>(mmf->data()), [graphSwitching, mmfIndex](uint8_t* ptr) {
          if (!graphSwitching) {
            auto& mmf = mmappedFilesVec[mmfIndex];
            QNN_DEBUG("Free context bin buffer %p of size %lu", ptr, mmf->size());
            mmf.reset();
          }
        });
#endif
  } else {
    // Regular binary allocation
    buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], [graphSwitching](uint8_t* ptr) {
      if (!graphSwitching) {
        delete[] ptr;
      }
    });

    if (!buffer) {
      QNN_ERROR("Failed to allocate memory for context index = %zu", contextIdx);
      return false;
    }
    if (true != readBinaryFromFile(binaryPath, buffer.get(), bufferSize)) {
      QNN_ERROR("Failed to read binary data for context index = %zu", contextIdx);
      return false;
    }
  }

  if (graphSwitching) {
    m_contextBinBuffersToBeCleared.push_back({buffer.get(), bufferSize});
  }

  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (QNN_SUCCESS !=
      m_qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle,
                                                      static_cast<void*>(buffer.get()),
                                                      bufferSize,
                                                      binaryInfo,
                                                      &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info for context index = %zu", contextIdx);
    return false;
  }

  return true;
}

bool QnnApi::parseIOTensorsAndAccumulate() {
  for (size_t gIdx = 0; gIdx < m_graphsCount; gIdx++) {
    auto& graph_info = m_graphsInfo[gIdx];
    for (bool io : {true, false}) {
      uint32_t n_tensors   = (io) ? graph_info->numInputTensors : graph_info->numOutputTensors;
      auto tensor_wrappers = (io) ? graph_info->inputTensors : graph_info->outputTensors;
      for (size_t tensor_idx = 0; tensor_idx < n_tensors; tensor_idx++) {
        Qnn_Tensor_t& tensor    = tensor_wrappers[tensor_idx];
        std::string tensor_name = QnnApi::getTensorName(tensor);

        std::vector<size_t> tensor_dims;
        if (!QnnApi::getTensorShape(tensor_dims, tensor)) {
          QNN_ERROR("Couldn't get tensor shape : %s", tensor_name.c_str());
          return false;
        }

        std::vector<qualla::QnnUtils::QuantParam> quantParams;
        if (!QnnApi::getTensorQuantParams(&tensor_wrappers[tensor_idx], quantParams)) {
          quantParams.emplace_back(0, 0);
        }

        m_graphtoIOMap[gIdx][tensor_name] = qualla::QnnUtils::Tensor(tensor_wrappers + tensor_idx);
      }
    }
  }

  // Maps tensor_name to context bitVector, each bit representing a context the tensor exists in
  std::map<std::string, CtxBitVector> tensor_ctx_map;
  // Maps a ContextHandle to a one-hot encoded bitVector (e.g. 1, 2, 4, ...)
  std::map<int, CtxBitVector> ctx_to_hash;

  // Iterate over all tensors in all GraphVariants to figure out allocations
  for (size_t gIdx = 0; gIdx < m_graphsCount; gIdx++) {
    // Map the context handle to a hashed bitVector
    auto curContextHandle = m_graphIdxToContextIdx[gIdx];
    if (!ctx_to_hash.contains(curContextHandle)) {
      ctx_to_hash[curContextHandle] = 1 << ctx_to_hash.size();
    }
    for (auto& [tname, tspec] : m_graphtoIOMap[gIdx]) {
      size_t size           = tspec.dims.getAlignedSize();
      CtxBitVector tcontext = ctx_to_hash[curContextHandle];

      // Check if it's LoRA enabled model
      if (!m_loraWeightEnabled && tname.find("lora") != std::string::npos)
        m_loraWeightEnabled = true;
      // Check if graph has lmhead weight input
      if (!m_lmHeadWeightInput && tname.compare("weight") == 0) m_lmHeadWeightInput = true;

      // Allocate KV Tensors as in+out
      if (tname.starts_with("past_")) {
        if (tname.ends_with("_in")) continue;  // kv_in is processed along with kv_out

        // For kv_out, add the size of kv_in as well
        const std::string tname_in = tname.substr(0, tname.rfind('_')).append("_in");

        if (m_graphtoIOMap[gIdx].count(tname_in)) {
          size += m_graphtoIOMap[gIdx][tname_in].dims.getAlignedSize();
        }

        // Allocate extra buffer for pointer shift
        // 1024-n for keys (1024-n)*128 for values
        // For aligned size, we might as well use 1024 and 128*1024
        if (m_kvUpdateMethod == POINTER_SHIFT)
          size += (tname.starts_with("past_key")) ? m_ctxSize : m_ctxSize * m_kvDim;
      }

      if (tensor_ctx_map.contains(tname)) {  // For duplicate tensor names, link them
        CtxBitVector context_bitvec = tensor_ctx_map.at(tname);
        size                        = std::max(m_contextAllocMap[context_bitvec][tname], size);
        if ((context_bitvec & tcontext) == 0)  // Set of contexts needs to be updated
          m_contextAllocMap[context_bitvec].erase(tname);

        tcontext |= context_bitvec;
      }

      m_contextAllocMap[tcontext][tname] = size;
      tensor_ctx_map[tname]              = tcontext;
    }

    // Cleanup is essential in case of very large number of splits
    for (auto it = m_contextAllocMap.cbegin(); it != m_contextAllocMap.cend();)
      it = (it->second.empty()) ? m_contextAllocMap.erase(it) : ++it;
  }

#if QNN_IO_TENSOR_DEBUG
  for (auto& [bitvector, nameMap] : m_contextAllocMap) {
    for (auto& [tname, size] : nameMap)
      QNN_DEBUG("Context: %d Tensor name: %s Tensor size: %zu", bitvector, tname.c_str(), size);
  }
#endif
  return true;
}

bool QnnApi::registerTensorsWithBackend(uint32_t& graphIdx) {
  std::map<std::string, std::tuple<int, size_t, size_t>> graph_allocs;
  for (auto& [tname, tspec] : m_graphtoIOMap[graphIdx]) {
    if (tname.starts_with("past_") && tname.ends_with("_in"))
      continue;  // Process past_key/value_Inputs along with the outputs
    auto& [alloc_idx, offset] = m_tensorAllocInfo.at(tname);

    size_t kv_offset = 0;
    size_t size      = tspec.dims.getAlignedSize();
    if (tname.starts_with("past_")) {
      auto in_name = tname.substr(0, tname.rfind("_")).append("_in");
      if (m_graphtoIOMap[graphIdx].count(in_name)) {
        auto kv_in = m_graphtoIOMap[graphIdx][in_name];
        kv_offset  = kv_in.dims.getAlignedSize();
        if (m_kvUpdateMethod == POINTER_SHIFT)
          kv_offset += (tname.starts_with("past_key")) ? m_ctxSize : m_ctxSize * m_kvDim;
        graph_allocs[in_name] = {alloc_idx, offset, kv_offset};
      }
    }
    graph_allocs[tname] = {alloc_idx, offset + kv_offset, size};
  }
  auto& curContextHandle = m_contextVec[m_graphIdxToContextIdx[graphIdx]];
  if (!m_ioBufferMgr->mapFusedBufferOffset(
          m_graphsInfo[graphIdx], curContextHandle, graph_allocs)) {
    QNN_ERROR("Error mapping tensor to allocation buffers");
    return false;
  }

#if QNN_IO_TENSOR_DEBUG
  for (auto& [tname, data] : graph_allocs) {
    QNN_DEBUG("Tensor Name: %s Alloc Idx: %d Tensor Offset: %zu Tensor Size: %zu",
              tname.c_str(),
              get<0>(data),
              get<1>(data),
              get<2>(data));
  }
#endif

  return true;
}
bool QnnApi::createFromBinaryHtp(std::vector<std::string> cachedBinariesPathVec,
                                 int64_t spill_fill_buffer_size,
                                 uint64_t mmap_budget,
                                 uint32_t dataAlignmentSize,
                                 bool graphSwitching,
                                 const std::vector<std::string>& execSelectGraphs,
                                 bool loadSelectGraphs,
                                 bool skipLoraValidation) {
  // Let backendExtensions populate configs
  QnnContext_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeCreateFromBinary(&customConfigs,
                                                                  &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeCreateFromBinary()");
      return false;
    }
  }

  // baseConfigList holds configs that are common to all contexts.
  ContextConfigList baseConfigList = ContextConfigList::fromArray(customConfigs, customConfigCount);

  if (nullptr == m_qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return false;
  }

  graphCountPerContext = getGraphCountPerContext();

  // Reading Binary Buffer and storing for later use during Deserialization
  std::vector<std::shared_ptr<uint8_t>> bufferVec(cachedBinariesPathVec.size());
  // Stores sizes of all the Binary Buffers
  std::vector<uint64_t> allBuffSizes(cachedBinariesPathVec.size());
  // Stores graphs per Contexts
  std::vector<uint32_t> graphsPerContext(cachedBinariesPathVec.size());

  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    auto _start = std::chrono::steady_clock::now();  // context Loading start
    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t>& buffer{bufferVec[contextIdx]};
    uint32_t graphsCount;

    // read serialized binary into a byte buffer
    bufferSize               = getFileSize(cachedBinariesPathVec[contextIdx]);
    allBuffSizes[contextIdx] = bufferSize;
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }

    // inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }

    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    if (!mapAndGetContextBinaryInfo(m_mmapContextBins,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    graphSwitching,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %zu", contextIdx);
      return false;
    }

    qnn_wrapper_api::GraphInfo_t** graphsInfo{nullptr};
    if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount)) {
      QNN_ERROR("Failed to copy metadata for graph index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    if (graphCountPerContext == -1) {
      graphCountPerContext = graphsCount;
      m_graphsInfo         = (qnn_wrapper_api::GraphInfo_t**)calloc(
          graphCountPerContext * cachedBinariesPathVec.size(),
          sizeof(qnn_wrapper_api::GraphInfo_t*));
    } else if (graphCountPerContext != graphsCount) {
      QNN_ERROR("Different len(graphs) found in different context files. Found %u vs %u",
                graphsCount,
                graphCountPerContext);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    auto _stop     = std::chrono::steady_clock::now();  // context Loading stop
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    (void)_duration;
    QNN_DEBUG("Loading contexts[%lu] took: %lld us", contextIdx, _duration);
    graphsPerContext.push_back(graphsCount);
    for (size_t gIdx = 0; gIdx < graphsCount; gIdx++) {
      std::string graph_name                = graphsInfo[gIdx]->graphName;
      m_graphsInfo[m_graphsCount]           = graphsInfo[gIdx];
      m_graphNameToInfo[graph_name]         = graphsInfo[gIdx];
      m_graphNameToContextIdx[graph_name]   = contextIdx;
      m_graphIdxToContextIdx[m_graphsCount] = contextIdx;
      m_graphsCount++;
    }
    if (graphsInfo) {
      free(graphsInfo);
      graphsInfo = nullptr;
    }
    m_qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;
  }

  // Iterate over all the tensors across the graphs Info and build info about the IO space it is
  // requiring.
  if (false == parseIOTensorsAndAccumulate()) {
    QNN_ERROR("Error in parsing the IO tensor info for all context binaries");
    return false;
  }

  // Spill-Fill configuration
  Qnn_ContextHandle_t first_contextHandle{nullptr};

  if (true !=
      getContextConfigs(baseConfigList, graphSwitching, execSelectGraphs, loadSelectGraphs)) {
    QNN_ERROR("Couldn't populate context configs");
    return false;
  }

  // I/O estimation configuration
  bool ioMemEstimationEnable = true;
#if defined(__QNX__) || (defined(__aarch64__) && defined(__linux__))
  const QnnDevice_PlatformInfo_t* platformInfo{nullptr};
  if (nullptr != m_qnnInterface.deviceGetPlatformInfo) {
    auto qnnStatus = m_qnnInterface.deviceGetPlatformInfo(nullptr, &platformInfo);
    if (QNN_SUCCESS != qnnStatus) {
      QNN_ERROR("Failed to get platform info.");
      return false;
    }
  }
  if (platformInfo->v1.hwDevices->v1.numCores > 1) {
    ioMemEstimationEnable = false;
  }
#endif /* #if defined (__QNX__) || (defined(__aarch64__) && \
          defined(__linux__)) */
  if (ioMemEstimationEnable) {
    QnnHtpContext_CustomConfig_t ioMemEstimation;
    ioMemEstimation.option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    ioMemEstimation.ioMemEstimation = true;
    baseConfigList.add(std::make_unique<ContextCustomHtpConfig>(ioMemEstimation));
  }
  if (mmap_budget > 0) {
    QnnHtpContext_CustomConfig_t customConfigReadBudget;
    customConfigReadBudget.option = QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
    customConfigReadBudget.fileReadMemoryBudgetInMb = mmap_budget;
    baseConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigReadBudget));
  }

  if (skipLoraValidation) {
    QnnHtpContext_CustomConfig_t customConfigSkipLoraValidation;
    customConfigSkipLoraValidation.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION;
    customConfigSkipLoraValidation.skipValidationOnBinarySection = true;
    baseConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigSkipLoraValidation));
  }

  bool isIOBufferMgrInitialized = false;
  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    if (nullptr == m_qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr for context index = %zu", contextIdx);
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    Qnn_ContextHandle_t contextHandle{nullptr};

    ContextConfigList configList(baseConfigList);
    if (spill_fill_buffer_size > 0) {
      QnnHtpContext_CustomConfig_t customConfigSF;
      customConfigSF.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
      QnnHtpContext_GroupRegistration_t groupInfo{nullptr};
      if (contextIdx == 0) {
        groupInfo.firstGroupHandle = 0x0;
      } else {
        groupInfo.firstGroupHandle = first_contextHandle;
      }
      groupInfo.maxSpillFillBuffer     = spill_fill_buffer_size;
      customConfigSF.groupRegistration = groupInfo;
      configList.add(std::make_unique<ContextCustomHtpConfig>(customConfigSF));
    }

    const QnnContext_Config_t** contextConfigs =
        static_cast<const QnnContext_Config_t**>(configList);

    auto start = std::chrono::steady_clock::now();  // context Deserialization starts

    auto errCode = m_qnnInterface.contextCreateFromBinary(m_backendHandle,
                                                          m_deviceHandle,
                                                          contextConfigs,
                                                          (const void*)bufferVec[contextIdx].get(),
                                                          allBuffSizes[contextIdx],
                                                          &contextHandle,
                                                          nullptr  // profile handle

    );

    auto stop     = std::chrono::steady_clock::now();  // context Deserialization stops
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    (void)duration;
    QNN_DEBUG("Initializing context[%lu] with %u graphs took: %lld us",
              contextIdx,
              graphsPerContext[contextIdx],
              duration);

    if (!isIOBufferMgrInitialized) {
      if (true != m_ioBufferMgr->initialize(contextHandle, dataAlignmentSize)) {
        QNN_ERROR("qnn-htp: failure to initialize IOTensor");
        return false;
      }

      isIOBufferMgrInitialized = true;

      // Calculate total allocation sizes and offset of each tensor within its allocated buffer
      if (m_ioBufferMgr->allocateBuffers(m_contextAllocMap, m_tensorAllocInfo) == false) {
        QNN_ERROR("Failed to allocate the Memory across the context buffers.");
        return false;
      }
    }

    if (errCode != QNN_SUCCESS) {
      QNN_ERROR("Could not create context from binary for context index = %zu : err %d",
                contextIdx,
                (int)errCode);
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    // Clearing buffer which is deseralized to reduce Memory footprint
    bufferVec[contextIdx].reset();

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }

    m_contextVec.push_back(contextHandle);
    m_contextIdtoHandle[contextIdx] = contextHandle;
    for (int n_graph = 0; n_graph < graphCountPerContext; n_graph++) {
      uint32_t graphIdx = contextIdx * graphCountPerContext + n_graph;

      qnn_wrapper_api::GraphInfo_t* cur_graph = m_graphsInfo[graphIdx];
      m_contextMap[cur_graph]                 = contextHandle;

      if (nullptr == m_qnnInterface.graphRetrieve) {
        QNN_ERROR("graphRetrieveFnHandle is nullptr.");
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }

      if (!m_graphsInfo ||
          QNN_SUCCESS != m_qnnInterface.graphRetrieve(
                             contextHandle, cur_graph->graphName, &(cur_graph->graph))) {
        QNN_ERROR("Unable to retrieve graph handle for graph index = %d", graphIdx);
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }

      // Register all the Tensors per graph.
      if (false == registerTensorsWithBackend(graphIdx)) {
        QNN_ERROR("Unable to MemRegister IO Tensors for graph index = %d", graphIdx);
        freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        return false;
      }
    }

    if (spill_fill_buffer_size > 0 && contextIdx == 0) {
      first_contextHandle = contextHandle;
    }
  }

  m_isContextCreated = true;
  m_contextCreatedFromBinary = true;

  QNN_DEBUG("Initialized %u graphs from %lu contexts", m_graphsCount, cachedBinariesPathVec.size());

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterCreateFromBinary()) {
      QNN_ERROR("Extensions Failure in afterCreateFromBinary()");
      return false;
    }
  }

  return true;
}

bool QnnApi::checkCapabilityOfCreateAsync(bool& propRet) {
  if (nullptr == m_qnnInterface.propertyHasCapability) {
    QNN_ERROR("propertyHasCapability is nullptr.......");
    return false;
  }
  if (QNN_PROPERTY_SUPPORTED == m_qnnInterface.propertyHasCapability(
                                    QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_LIST_ASYNC)) {
    propRet = true;
  } else {
    propRet = false;
  }
  return true;
}

bool freeContextParams(QnnContext_Params_t** context_params_list, uint32_t numParams) {
  if (context_params_list == nullptr || *context_params_list == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numParams; i++) {
    if (nullptr != context_params_list[i]) {
      delete context_params_list[i];
    }
  }
  return true;
}

void QnnApi::contextNotifyFn(Qnn_ContextHandle_t context,
                             Qnn_GraphHandle_t graph,
                             const char* graph_name,
                             QnnContext_createFromBinaryAsyncNotifyType_t completeType,
                             void* notifyParam,
                             Qnn_ErrorHandle_t status) {
  std::pair<QnnApi*, uint32_t>* pair = reinterpret_cast<std::pair<QnnApi*, uint32_t>*>(notifyParam);
  QnnApi* QnnApi                     = pair->first;
  uint32_t contextId                 = pair->second;

  if (completeType ==
      QnnContext_createFromBinaryAsyncNotifyType_t::QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) {
    QnnApi->updateContext(context, contextId);
  } else if (completeType ==
             QnnContext_createFromBinaryAsyncNotifyType_t::QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT) {
    QnnApi->updateQnnApiGraphsandContextsInfo(graph_name, graph, contextId);
  }
}

bool QnnApi::createFromBinaryListAsyncHtp(std::vector<std::string> cachedBinariesPathVec,
                                          int64_t spill_fill_buffer_size,
                                          uint64_t mmap_budget,
                                          bool graphSwitching,
                                          const std::vector<std::string>& execSelectGraphs,
                                          bool loadSelectGraphs,
                                          bool skipLoraValidation) {
  auto _start = std::chrono::steady_clock::now();

  // Let backendExtensions populate configs
  QnnContext_Config_t** customConfigs{nullptr};
  uint32_t customConfigCount{0};
  std::map<std::string, std::tuple<QnnContext_Config_t**, uint32_t>> contextKeyToCustomConfigsMap;
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeCreateContextsFromBinaryList(
            &contextKeyToCustomConfigsMap, &customConfigs, &customConfigCount)) {
      QNN_ERROR("Extensions Failure in beforeCreateContextsFromBinaryList()");
      return false;
    }
  }

  // groupConfigList holds configs that are common to all contexts.
  ContextConfigList groupConfigList =
      ContextConfigList::fromArray(customConfigs, customConfigCount);
  const QnnContext_Config_t** groupConfigs =
      static_cast<const QnnContext_Config_t**>(groupConfigList);

  if (nullptr == m_qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return false;
  }

  // contextConfigList contains "per-context" configs provided to the context params lists
  ContextConfigList contextConfigList;
  if (true !=
      getContextConfigs(contextConfigList, graphSwitching, execSelectGraphs, loadSelectGraphs)) {
    QNN_ERROR("Couldn't populate context configs");
    return false;
  }

  if (mmap_budget > 0) {
    QnnHtpContext_CustomConfig_t customConfigReadBudget;
    customConfigReadBudget.option = QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
    customConfigReadBudget.fileReadMemoryBudgetInMb = mmap_budget;
    contextConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigReadBudget));
  }

  if (skipLoraValidation) {
    QnnHtpContext_CustomConfig_t customConfigSkipLoraValidation;
    customConfigSkipLoraValidation.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION;
    customConfigSkipLoraValidation.skipValidationOnBinarySection = true;
    contextConfigList.add(std::make_unique<ContextCustomHtpConfig>(customConfigSkipLoraValidation));
  }

  const QnnContext_Config_t** contextConfigs =
      static_cast<const QnnContext_Config_t**>(contextConfigList);

  graphCountPerContext = getGraphCountPerContext();
  std::vector<QnnContext_Params_t*> context_params_list(cachedBinariesPathVec.size() + 1, nullptr);
  std::vector<std::shared_ptr<uint8_t>> bufferVec(cachedBinariesPathVec.size());
  // for every context's graph info
  qnn_wrapper_api::GraphInfo_t*** graphsInfo = (qnn_wrapper_api::GraphInfo_t***)calloc(
      cachedBinariesPathVec.size(), sizeof(qnn_wrapper_api::GraphInfo_t**));
  uint32_t graphsTotalNum = 0;

  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t>& buffer{bufferVec[contextIdx]};
    uint32_t graphsCount;

    // read serialized binary into a byte buffer
    bufferSize = getFileSize(cachedBinariesPathVec[contextIdx]);
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }

    // inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    if (!mapAndGetContextBinaryInfo(m_mmapContextBins,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    graphSwitching,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary.");
      return false;
    }

    if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo[contextIdx], graphsCount)) {
      QNN_ERROR("Failed to copy metadata for graph index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo[contextIdx], graphsCount);
      freeGraphsInfo(&m_graphsInfo, graphsCount);
      return false;
    }

    if (graphCountPerContext == -1) {
      graphCountPerContext = graphsCount;
      graphsTotalNum       = graphCountPerContext * cachedBinariesPathVec.size();
      m_graphsInfo         = (qnn_wrapper_api::GraphInfo_t**)calloc(graphsTotalNum,
                                                            sizeof(qnn_wrapper_api::GraphInfo_t*));

    } else if (graphCountPerContext != graphsCount) {
      QNN_ERROR("Different len(graphs) found in different context files. Found %u vs %u",
                graphsCount,
                graphCountPerContext);
      freeGraphsInfo(&graphsInfo[contextIdx], graphsCount);
      freeGraphsInfo(&m_graphsInfo, graphsTotalNum);
      return false;
    }
    for (size_t gIdx = 0; gIdx < graphsCount; gIdx++) {
      int graphIdxOfAll                                         = contextIdx * graphsCount + gIdx;
      m_graphsInfo[graphIdxOfAll]                               = graphsInfo[contextIdx][gIdx];
      m_graphNameToInfo[m_graphsInfo[graphIdxOfAll]->graphName] = m_graphsInfo[graphIdxOfAll];
      m_graphIdxToContextIdx[graphIdxOfAll]                     = contextIdx;
    }
    m_qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (m_profileBackendHandle) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }

    // passing class QnnApi pointer into callback funtion(notifyFn)
    std::pair<QnnApi*, uint32_t>* notifyParam =
        new std::pair<QnnApi*, uint32_t>(this, (size_t)contextIdx);
    QnnContext_Params_t* contextParam =
        new QnnContext_Params_t{.version = QNN_CONTEXT_PARAMS_VERSION_1,
                                .v1      = QnnContext_ParamsV1_t{contextConfigs,
                                                            (const void*)buffer.get(),
                                                            bufferSize,
                                                            nullptr,
                                                            QnnApi::contextNotifyFn,
                                                            (void*)notifyParam}};

    context_params_list[contextIdx] = contextParam;

    auto _stop     = std::chrono::steady_clock::now();
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    (void)_duration;
    QNN_DEBUG("Loading contexts[%lu] took: %lld us", contextIdx, _duration);
  }

  if (nullptr == m_qnnInterface.contextCreateFromBinaryListAsync) {
    QNN_ERROR("contextCreateFromBinaryListAsyncFnHandle is nullptr");
    freeGraphsInfo(&m_graphsInfo, graphsTotalNum);
    freeContextParams(context_params_list.data(), cachedBinariesPathVec.size());
    return false;
  }

  auto start   = std::chrono::steady_clock::now();
  auto errCode = m_qnnInterface.contextCreateFromBinaryListAsync(
      m_backendHandle,
      m_deviceHandle,
      const_cast<const QnnContext_Params_t**>(context_params_list.data()),
      groupConfigs,
      nullptr);
  auto stop     = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  (void)duration;
  QNN_DEBUG("Initializing %lu context with %u graphs took: %lld us",
            cachedBinariesPathVec.size(),
            graphsTotalNum,
            duration);

  // Explicitly free the context binary buffers. This ensures that the lifecycle
  // of the buffers outlasts the API call where their raw pointers are referenced.
  for (auto contextBinaryBuffer : bufferVec) {
    QNN_DEBUG("Freeing context binary buffer @%p", contextBinaryBuffer.get());
    contextBinaryBuffer.reset();
  }

  if (errCode != QNN_SUCCESS) {
    QNN_ERROR("Could not create context from binary List Async for context, err %d", (int)errCode);
    freeGraphsInfo(&m_graphsInfo, graphsTotalNum);
    freeContextParams(context_params_list.data(), cachedBinariesPathVec.size());
    return false;
  }

  // set graphInfo in m_graphsInfo
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    int contextIdxOfgraphsInfo    = graphIdx / graphCountPerContext;
    uint32_t contexIdxofCurrGraph = m_graphNameToContextIdx[m_graphsInfo[graphIdx]->graphName];
    m_graphsInfo[graphIdx] = graphsInfo[contextIdxOfgraphsInfo][graphIdx % graphCountPerContext];
    m_contextMap[m_graphsInfo[graphIdx]] = m_contextIdtoHandle[contexIdxofCurrGraph];
  }

  m_isContextCreated = true;
  m_contextCreatedFromBinary = true;

  if (true != freeContextParams(context_params_list.data(), cachedBinariesPathVec.size())) {
    QNN_ERROR("Couldn't free context params list");
    return false;
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterCreateContextsFromBinaryList()) {
      QNN_ERROR("Extensions Failure in afterCreateContextsFromBinaryList()");
      return false;
    }
  }
  return true;
}

static std::vector<std::string> __split(std::string_view str, char delim) {
  std::vector<std::string> split;

  size_t i = 0, p = 0;

  for (; i <= str.size(); ++i) {
    if (i == str.size() || str[i] == delim) {
      split.push_back(std::string(str.data() + p, i - p));
      p = ++i;
    }
  }

  return split;
}

bool QnnApi::registerOpPackage(std::string opPackagePath) {
  const size_t pathIdx              = 0;
  const size_t interfaceProviderIdx = 1;
  const size_t targetIdx            = 2;

  auto opPackage = __split(opPackagePath, ':');

  if (opPackage.size() != 2 && opPackage.size() != 3) {
    return false;
  }

  if (nullptr == m_qnnInterface.backendRegisterOpPackage) {
    return false;
  }

  const char* target = nullptr;
  if (opPackage.size() == 3) {
    target = (char*)opPackage[targetIdx].c_str();
  }

  Qnn_ErrorHandle_t errCode =
      m_qnnInterface.backendRegisterOpPackage(m_backendHandle,
                                              (char*)opPackage[pathIdx].c_str(),
                                              (char*)opPackage[interfaceProviderIdx].c_str(),
                                              target);
  if (QNN_SUCCESS != errCode) {
    QNN_ERROR("Could not register OpPackage backend due to error = %llu",
              static_cast<unsigned long long>(errCode));
    return false;
  }

  return true;
}

// Performance Setting for HTP
bool QnnApi::initializePerformance() {
  QnnDevice_Infrastructure_t deviceInfra = nullptr;
  if (QNN_SUCCESS != m_qnnInterface.deviceGetInfrastructure(&deviceInfra)) {
    QNN_ERROR("Failure in deviceGetInfrastructure()");
    return false;
  }

  QnnHtpDevice_Infrastructure_t* htpInfra =
      static_cast<QnnHtpDevice_Infrastructure_t*>(deviceInfra);
  m_perfInfra       = &(htpInfra->perfInfra);
  uint32_t deviceId = 0;
  uint32_t coreId   = 0;
  if (QNN_SUCCESS != m_perfInfra->createPowerConfigId(deviceId, coreId, &m_powerConfigId)) {
    QNN_ERROR("Failure in createPowerConfigId()");
    return false;
  }

  return true;
}

bool QnnApi::destroyPerformance() {
  if (nullptr != m_perfInfra && QNN_SUCCESS != m_perfInfra->destroyPowerConfigId(m_powerConfigId)) {
    QNN_ERROR("Failure in destroyPowerConfigId()");
    return false;
  }

  return true;
}

bool QnnApi::boostPerformance() {
  // Initialize the power config and select the voltage corner values for the performance setting.
  QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
  memset(&powerConfig, 0, sizeof(powerConfig));

  powerConfig.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  powerConfig.dcvsV3Config.dcvsEnable    = 1;
  powerConfig.dcvsV3Config.setDcvsEnable = 1;
  powerConfig.dcvsV3Config.contextId     = m_powerConfigId;

  // refer QnnHtpPerfInfrastructure.h
  powerConfig.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;

  // Set Sleep-Disable latency parameter
  powerConfig.dcvsV3Config.setSleepDisable = 0;
  powerConfig.dcvsV3Config.sleepDisable    = 0;

  // Set Sleep latency parameter
  powerConfig.dcvsV3Config.setSleepLatency = 0;
  powerConfig.dcvsV3Config.sleepLatency    = 1000;  // range 40-2000 micro sec

  // Set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
  powerConfig.dcvsV3Config.setBusParams           = 1;
  powerConfig.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  powerConfig.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

  // set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
  powerConfig.dcvsV3Config.setCoreParams           = 1;
  powerConfig.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  powerConfig.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

  // Set power config with different performance parameters
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfig, NULL};
  if (QNN_SUCCESS != m_perfInfra->setPowerConfig(m_powerConfigId, powerConfigs)) {
    QNN_ERROR("Failure in setPowerConfig() from boostPerformance");
    return false;
  }

  return true;
}

bool QnnApi::resetPerformance() {
  // Initialize the power config and select the voltage corner values for the performance setting.
  QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
  memset(&powerConfig, 0, sizeof(powerConfig));

  powerConfig.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  powerConfig.dcvsV3Config.dcvsEnable    = 1;
  powerConfig.dcvsV3Config.setDcvsEnable = 1;
  powerConfig.dcvsV3Config.contextId     = m_powerConfigId;

  // refer QnnHtpPerfInfrastructure.h
  powerConfig.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;

  // Set Sleep-Disable latency parameter
  powerConfig.dcvsV3Config.setSleepDisable = 0;
  powerConfig.dcvsV3Config.sleepDisable    = 0;

  // Set Sleep latency parameter
  powerConfig.dcvsV3Config.setSleepLatency = 0;
  powerConfig.dcvsV3Config.sleepLatency    = 1000;  // range 40-2000 micro sec

  // Set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
  powerConfig.dcvsV3Config.setBusParams           = 1;
  powerConfig.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_NOM;
  powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
  powerConfig.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;

  // set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
  powerConfig.dcvsV3Config.setCoreParams           = 1;
  powerConfig.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_NOM;
  powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
  powerConfig.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;

  // Set power config with different performance parameters
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfig, NULL};
  if (QNN_SUCCESS != m_perfInfra->setPowerConfig(m_powerConfigId, powerConfigs)) {
    QNN_ERROR("Failure in setPowerConfig() from resetPerformance");
    return false;
  }

  return true;
}

bool QnnApi::initializeHtp(std::string backendPath,
                           std::vector<std::string> modelPathOrCachedBinaryPathVec,
                           BackendExtensionsConfigs backendExtensionsConfig,
                           qnn::tools::netrun::PerfProfile parsedPerfProfile,
                           std::vector<GraphConfigs> graphConfigs,
                           bool loadFromCachedBinary,
                           std::string systemLibraryPath,
                           bool debugModeRequested,
                           int64_t spill_fill_buffer_size,
                           uint32_t dataAlignmentSize,
                           bool mmapContextBins,
                           bool asyncInit,
                           uint64_t mmap_budget,
                           bool debug_qnn,
                           bool graphSwitching,
                           const std::vector<std::string>& execSelectGraphs,
                           bool loadSelectGraphs,
                           bool skipLoraValidation,
                           uint32_t logLevel,
                           LogCallback inLogCallBack) {
  m_perfProfile = parsedPerfProfile;
  if (modelPathOrCachedBinaryPathVec.size() > 1 && false == loadFromCachedBinary) {
    QNN_ERROR(
        "Currently only 1 model file is supported for this framework! \
            Although multiple context files are supported!");
    return false;
  }

  m_mmapContextBins = mmapContextBins;

  // Setting up Debug mode
  m_DebugModeRequested = debugModeRequested;
  if (m_DebugModeRequested) {
    QNN_WARN("Warning: Debug mode set to true.");
  }

  // Initialize the QNN run time
  if (false == getQnnInterface(backendPath)) {
    QNN_ERROR("Qnn getQnnInterface FAILED!");
    return false;
  }

  if (loadFromCachedBinary) {
    if (false == getQnnSystemInterface(systemLibraryPath)) {
      QNN_ERROR("Qnn getQnnSystemInterface FAILED!");
      return false;
    }
  } else {
    if (false == loadModel(modelPathOrCachedBinaryPathVec[0])) {
      QNN_ERROR("Loading model FAILED!");
      return false;
    }
  }

  QnnLog_Level_t qnnLogLevel = static_cast<QnnLog_Level_t>(logLevel);
  if (false == initializeLogging(qnnLogLevel, debug_qnn, inLogCallBack)) {
    QNN_ERROR("Unable to Initialize logging in backend");
    return false;
  }

  // initialize backend extensions
#ifdef QUALLA_INTERNAL_QNN_SDK
  // Initialize backendExtensions only when both backend ext config and backend ext lib are provided
  if (!backendExtensionsConfig.configFilePath.empty() &&
      false == initializeBackendExtensions(
                   backendExtensionsConfig, parsedPerfProfile, debug_qnn, qnnLogLevel)) {
    QNN_WARN("Failure in initializing backend extensions.");
  }
#else
  if (false == initializeBackendExtensions(
                   backendExtensionsConfig, parsedPerfProfile, debug_qnn, qnnLogLevel)) {
    QNN_ERROR("Failure in initializing backend extensions.");
    return false;
  }
#endif
  if (false == initializeBackend()) {
    QNN_ERROR("Qnn initializeBackend FAILED!");
    return false;
  }
  if (false == createDevice()) {
    QNN_ERROR("Device Creation failure");
    setDeviceStatus(false);
    return false;
  } else {
    setDeviceStatus(true);
  }
  if (!loadFromCachedBinary) {
    if (false == createContext()) {
      QNN_ERROR("Qnn createContext FAILED!");
      return false;
    }
    if (false == composeGraphs(graphConfigs)) {
      QNN_ERROR("composeGraphs FAILED!");
      return false;
    }
    if (false == finalizeGraphs()) {
      QNN_ERROR("finalizeGraphs FAILED!");
      return false;
    }
  } else {
    bool cfb_ret         = false;
    bool asyncCapability = false;
    if (asyncInit == true) {
      if (!checkCapabilityOfCreateAsync(asyncCapability)) {
        QNN_ERROR("Capabilty checked failed");
        return false;
      }
      asyncInit = asyncCapability && asyncInit;
    }
    if (asyncInit == true) {
      QNN_INFO("Using create From Binary List Async");
      cfb_ret = createFromBinaryListAsyncHtp(modelPathOrCachedBinaryPathVec,
                                             spill_fill_buffer_size,
                                             mmap_budget,
                                             graphSwitching,
                                             execSelectGraphs,
                                             loadSelectGraphs,
                                             skipLoraValidation);
      if (cfb_ret == false) {
        QNN_ERROR("Create From Binary List Async FAILED!");
        return false;
      }

    } else {
      QNN_INFO("Using create From Binary");
      cfb_ret = createFromBinaryHtp(modelPathOrCachedBinaryPathVec,
                                    spill_fill_buffer_size,
                                    mmap_budget,
                                    dataAlignmentSize,
                                    graphSwitching,
                                    execSelectGraphs,
                                    loadSelectGraphs,
                                    skipLoraValidation);
      if (false == cfb_ret) {
        QNN_ERROR("Create From Binary FAILED!");
        return false;
      }
    }
  }

  // if (false == initializePerformance()) {
  //     QNN_ERROR("initialize Performance FAILED!");
  //     return false;
  // }

  // Apply the HTP perf profile now that the context/graphs exist. The BackendExtensions ctor cannot do
  // this - it runs before context creation, so a vote set there is dropped and decode ends up coupled to
  // the calling CPU core's clock (collapsing on a throttled/slow core). Applying it here keeps decode at
  // full rate regardless of CPU clock, matching Genie.
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->setPerfProfile(parsedPerfProfile)) {
      QNN_WARN("Unable to set perf profile after context creation.");
    }
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    m_graphNameToIndex[m_graphsInfo[graphIdx]->graphName] = graphIdx;
  }

#if NSP_LOG_LEVEL > 1
  for (const auto& graphNameIndex : m_graphNameToIndex) {
    QNN_DEBUG("Found Graph name %s corresponding to index %d",
              graphNameIndex.first.c_str(),
              graphNameIndex.second);
  }

  fprintf(stderr, "context_handles = [");
  for (auto ctx_handle : m_contextVec) fprintf(stderr, "%p, ", ctx_handle);
  fprintf(stderr, "]\n");
#endif
  return true;
}

bool QnnApi::initializeCpu(std::string backendPath,
                           std::string modelPath,
                           std::string opPackage,
                           std::vector<GraphConfigs> graphConfigs,
                           uint32_t* inputDim,
                           uint32_t inputRank,
                           uint32_t* outputDim,
                           uint32_t outputRank,
                           uint32_t* kvDim,
                           uint32_t kvRank,
                           uint32_t* kvScaleDim,
                           Qnn_Param_t* params,
                           uint32_t numParams,
                           bool debugModeRequested,
                           bool debug_qnn,
                           uint32_t logLevel,
                           LogCallback inLogCallBack) {
  // Setting up Debug mode
  m_DebugModeRequested = debugModeRequested;
  if (m_DebugModeRequested) {
    QNN_WARN("Warning: Debug mode set to true.");
  }

  // Initialize the QNN run time
  if (false == getQnnInterface(backendPath)) {
    QNN_ERROR("Qnn getQnnInterface FAILED!");
    return false;
  }

  QnnLog_Level_t qnnLogLevel = static_cast<QnnLog_Level_t>(logLevel);
  if (false == initializeLogging(qnnLogLevel, debug_qnn, inLogCallBack)) {
    QNN_ERROR("Unable to Initialize logging in backend");
  }

  if (!m_backendHandle) {
    if (false == initializeBackend()) {
      QNN_ERROR("Qnn initializeBackend FAILED!");
      return false;
    }

    // CPU does not support createDevice.
    setDeviceStatus(false);
    if (false == registerOpPackage(opPackage)) {
      QNN_ERROR("Qnn initializeBackend FAILED!");
      return false;
    }
  }

// Change to 1 to enable QNN Basic profiling
#if 0
    if (false == initProfiling()) {
        QNN_ERROR("Profiling init failure");
        return false;
    }
#endif
  if (false == loadModel(modelPath)) {
    QNN_ERROR("Loading model FAILED!");
    return false;
  }
  if (false == createContext()) {
    QNN_ERROR("Qnn createContext FAILED!");
    return false;
  }
  if (false == composeGraphs(graphConfigs,
                             inputDim,
                             inputRank,
                             outputDim,
                             outputRank,
                             kvDim,
                             kvRank,
                             kvScaleDim,
                             params,
                             numParams)) {
    QNN_ERROR("composeGraphs FAILED!");
    return false;
  }
  if (false == finalizeCpuGraphs()) {
    QNN_ERROR("finalizeGraphs FAILED!");
    return false;
  }

  for (size_t graphIdx = (m_graphsCount - graphCountPerContext); graphIdx < m_graphsCount;
       graphIdx++) {
    m_graphNameToIndex[m_graphsInfo[graphIdx]->graphName] = graphIdx;
  }
#if NSP_LOG_LEVEL > 1
  for (const auto& graphNameIndex : m_graphNameToIndex) {
    QNN_DEBUG("Found Graph name %s corresponding to index %d",
              graphNameIndex.first.c_str(),
              graphNameIndex.second);
  }
#endif
  return true;
}

bool QnnApi::graphExecute(Qnn_Tensor_t* input,
                          Qnn_Tensor_t* output,
                          std::string graphName,
                          std::map<std::string, std::pair<double, uint16_t>>& timeLogs) {
  qnn_wrapper_api::GraphInfo_t* graph_info = m_graphsInfo[m_graphNameToIndex[graphName]];
  return graphExecute(graph_info, input, output, timeLogs);
}

bool QnnApi::graphExecute(qnn_wrapper_api::GraphInfo_t* graph_info,
                          const Qnn_Tensor_t* input,
                          Qnn_Tensor_t* output,
                          std::map<std::string, std::pair<double, uint16_t>>& timeLogs) {
  std::string graphName = graph_info->graphName;
  QnnGraph_Config_t** customGraphConfigs{nullptr};
  uint32_t configCount{0};
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->beforeExecute(
            graphName.c_str(), &customGraphConfigs, &configCount)) {
      QNN_ERROR("Extensions Failure in beforeExecute()");
      return false;
    }
    if (customGraphConfigs) {
      if (true !=
          setGraphConfigsBeforeExecute(graph_info->graph, customGraphConfigs, configCount)) {
        QNN_ERROR("Failure in setGraphConfigsBeforeExecute()");
        return false;
      }
    }
  }

  // if (true != boostPerformance()) {
  //     QNN_ERROR("Couldn't boost the performance");
  //     return false;
  // }

  Qnn_ErrorHandle_t ret = QNN_GRAPH_NO_ERROR;
  try {
#if NSP_LOG_LEVEL > 1
    auto start = std::chrono::steady_clock::now();
#endif

    ret = m_qnnInterface.graphExecute(graph_info->graph,
                                      input,
                                      graph_info->numInputTensors,
                                      output,
                                      graph_info->numOutputTensors,
                                      m_profileBackendHandle,
                                      nullptr);
#if NSP_LOG_LEVEL > 1
    auto stop     = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    QNN_DEBUG("graphExecute[%s] took: %lld us", graphName.c_str(), duration);
#endif
#if NSP_LOG_LEVEL > 6
    timeLogs[graphName].first += static_cast<double>(duration);
    timeLogs[graphName].second++;
#endif

  } catch (const std::exception&) {
    QNN_ERROR("ERROR executing inference ret");
  } catch (...) {
    QNN_ERROR("ERROR executing inference ret");
  }

  if (m_profileBackendHandle) {
    extractBackendProfilingInfo(m_profileBackendHandle, timeLogs, graphName);
  }

  // if (true != resetPerformance()) {
  //     QNN_ERROR("Couldn't reset the performance");
  //     return false;
  // }

  if (ret != QNN_GRAPH_NO_ERROR) {
    QNN_ERROR("Failed to execute graph. Error %llu", static_cast<unsigned long long>(ret));
    return false;
  }

  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (!m_backendExtensions->interface()->afterExecute()) {
      QNN_ERROR("Extensions Failure in afterExecute()");
      return false;
    }
  }

  return true;
}

bool QnnApi::getTensorQuantParams(const Qnn_Tensor_t* tensor,
                                  std::vector<QuantParam>& quantParamsVec) {
  bool status      = false;
  auto dataType    = QNN_TENSOR_GET_DATA_TYPE(tensor);
  auto quantParams = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  if (dataType == QNN_DATATYPE_UFIXED_POINT_8 || dataType == QNN_DATATYPE_SFIXED_POINT_8 ||
      dataType == QNN_DATATYPE_UFIXED_POINT_16) {
    auto quantEncodingType = quantParams.quantizationEncoding;
    if (quantEncodingType == Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      status         = true;
      double scale   = quantParams.scaleOffsetEncoding.scale;
      int32_t offset = quantParams.scaleOffsetEncoding.offset;
      quantParamsVec.emplace_back(scale, offset);
    } else if (quantEncodingType ==
               Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      status              = true;
      auto encodingStruct = quantParams.axisScaleOffsetEncoding;
      for (uint32_t n = 0; n < encodingStruct.numScaleOffsets; n++) {
        auto scaleOffset = encodingStruct.scaleOffset[n];
        quantParamsVec.emplace_back(scaleOffset.scale, scaleOffset.offset);
      }
    } else {
      QNN_ERROR("quant encoding type not supported");
    }
  }
  return status;
}

bool QnnApi::getTensorShape(std::vector<size_t>& tensorDims, const Qnn_Tensor_t& tensor) {
  if (false == fillDims(tensorDims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor)))
    return false;

  tensorDims.push_back(getDataTypeSize(QNN_TENSOR_GET_DATA_TYPE(tensor)));
  return true;
}

bool QnnApi::getTensorNameAndShape(std::string& tensorName,
                                   std::vector<size_t>& tensorDims,
                                   Qnn_Tensor_t& tensor) {
  tensorName = std::string(QNN_TENSOR_GET_NAME(tensor));
  if (false == fillDims(tensorDims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor)))
    return false;

  tensorDims.push_back(g_qnnDataTypeToSize[QNN_TENSOR_GET_DATA_TYPE(tensor)]);
  return true;
}

bool QnnApi::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle,
    std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
    std::string graphName) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("QNN HTP Profile handle is nullptr; may not be initialized.");
    return false;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get events.");
    return false;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event), timeLogs, graphName);
    extractProfilingSubEvents(*(profileEvents + event), timeLogs, graphName);
  }
  return true;
}

bool QnnApi::extractProfilingSubEvents(QnnProfile_EventId_t profileEventId,
                                       std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                       std::string graphName) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get sub events.");
    return false;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent), timeLogs, graphName);
    extractProfilingSubEvents(*(profileSubEvents + subEvent), timeLogs, graphName);
  }
  return true;
}

bool QnnApi::extractProfilingEvent(QnnProfile_EventId_t profileEventId,
                                   std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                   std::string graphName) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR != m_qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return false;
  }

  QNN_DEBUG(
      "Event Info - Event Type: [%d], Event Value: [%lu], Event Identifier: [%s], Event Unit: [%d]",
      eventData.type,
      eventData.value,
      eventData.identifier,
      eventData.unit);
#if NSP_LOG_LEVEL > 6
  timeLogs[graphName + "_" + eventData.identifier].first += static_cast<double>(eventData.value);
  timeLogs[graphName + "_" + eventData.identifier].second++;
#endif

  return true;
}

bool QnnApi::extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("QNN HTP Profile handle is nullptr; may not be initialized.");
    return false;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get events.");
    return false;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event));
    extractProfilingSubEvents(*(profileEvents + event));
  }
  return true;
}

bool QnnApi::extractProfilingSubEvents(QnnProfile_EventId_t profileEventId) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in QNN HTP profile get sub events.");
    return false;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent));
    extractProfilingSubEvents(*(profileSubEvents + subEvent));
  }
  return true;
}

bool QnnApi::extractProfilingEvent(QnnProfile_EventId_t profileEventId) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR != m_qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return false;
  }

  QNN_DEBUG(
      "Event Info - Event Type: [%d], Event Value: [%lu], Event Identifier: [%s], Event Unit: [%d]",
      eventData.type,
      eventData.value,
      eventData.identifier,
      eventData.unit);

  return true;
}

bool QnnApi::applyBinarySection(std::string graphName, std::string binSectionPath) {
  auto graph_id = m_graphNameToIndex[graphName];
  return applyBinarySection(graph_id, binSectionPath);
}

bool QnnApi::applyBinarySection(uint32_t graphId, std::string binSectionPath) {
  // assumption splitNum  from 0
  QNN_DEBUG("QnnApi::applyBinarySection %d ", graphId);
  if (nullptr == m_qnnInterface.contextApplyBinarySection) {
    QNN_ERROR("contextApplyBinarySection Interface not suported!!");
    return false;
  }
  if (graphId >= m_graphsCount) {
    QNN_ERROR(" Passed split %d  base Model graphcount %d ", graphId, m_graphsCount);
    return false;
  }
  uint64_t bufferSize{0};
  std::shared_ptr<uint8_t> buffer{nullptr};
  bufferSize = getFileSize(binSectionPath);
  buffer     = std::shared_ptr<uint8_t>(new uint8_t[bufferSize]);
  if (true != readBinaryFromFile(binSectionPath, buffer.get(), bufferSize)) {
    QNN_ERROR("Failed to read binary data for context index = %d", graphId);
    return false;
  }

  QnnContext_Buffer_t qnnBuffer;
  qnnBuffer.version               = QNN_CONTEXT_BUFFER_VERSION_1;
  qnnBuffer.v1.memType            = QNN_CONTEXTMEMTYPE_RAW;
  qnnBuffer.v1.binaryBuf.dataSize = bufferSize;
  qnnBuffer.v1.binaryBuf.data     = static_cast<void*>(buffer.get());
  auto graphCountPerContext       = getGraphCountPerContext();
  if (graphCountPerContext <= 0) {
    QNN_ERROR(" graphCountPerContext is <=0 ");
    return false;
  }

  auto contextHandle = m_contextVec[graphId / graphCountPerContext];
  auto graphHandle   = m_graphsInfo[graphId]->graph;
  if (contextHandle == nullptr || graphHandle == nullptr) {
    QNN_ERROR(" contexthandle or graph handle is null for patch no = %d ", graphId);
    return false;
  }

  auto errorCode = m_qnnInterface.contextApplyBinarySection(contextHandle,
                                                            graphHandle,
                                                            QNN_CONTEXT_SECTION_UPDATABLE,
                                                            &qnnBuffer,
                                                            nullptr,  // profile handle is null
                                                            nullptr   // singal handle is null
  );
  if (errorCode != QNN_SUCCESS) {
    QNN_ERROR("Could not Apply Patch for graph = %d errocode = %zu ", graphId, errorCode);
    return false;
  }
  return true;
}

bool QnnApi::applyBinarySection(uint32_t binIndex,
                                std::string binSectionPath,
                                bool useMmap,
                                bool graphSwitch,
                                std::string& lazyLora) {
  // assumption splitNum  from 0
  QNN_DEBUG("QnnApi::applyBinarySection %d ", binIndex);
  uint32_t numAdapterGraph = 0;
  if (nullptr == m_qnnInterface.contextApplyBinarySection) {
    QNN_ERROR("contextApplyBinarySection Interface not suported!!");
    return false;
  }
  if (binIndex >= m_graphsCount) {
    QNN_ERROR(" Passed split %d  base Model graphcount %d ", binIndex, m_graphsCount);
    return false;
  }
  uint64_t bufferSize{0};
  std::shared_ptr<uint8_t> buffer{nullptr};
  bufferSize = getFileSize(binSectionPath);

  auto graphCountPerContext = getGraphCountPerContext();
  if (graphCountPerContext <= 0) {
    QNN_ERROR(" graphCountPerContext is <=0 ");
    return false;
  }
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle for context index = %u", binIndex);
    return false;
  }
  Qnn_ContextBinarySize_t binaryInfoSize{0};

  if (m_adapterNameToBuffer[binSectionPath]) {
    buffer = m_adapterNameToBuffer[binSectionPath];
    if (QNN_SUCCESS !=
        m_qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle,
                                                        static_cast<void*>(buffer.get()),
                                                        bufferSize,
                                                        &binaryInfo,
                                                        &binaryInfoSize)) {
      QNN_ERROR("Failed to get context binary info for context index = %u", binIndex);
      return false;
    }
  } else {
    if (!mapAndGetContextBinaryInfo(useMmap,
                                    buffer,
                                    binSectionPath,
                                    bufferSize,
                                    binIndex,
                                    graphSwitch,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %u", binIndex);
      return false;
    }
    m_adapterNameToBuffer[binSectionPath] = buffer;
  }
  numAdapterGraph = getNumGraphInBinary(binaryInfo);
  if (numAdapterGraph <= 0) {
    QNN_ERROR(" numAdapterGraph is <=0 ");
    return false;
  }
  uint32_t contextId = 0;
  uint32_t graphId   = 0;
  for (size_t idx = 0; idx < numAdapterGraph; idx++) {
    graphId            = numAdapterGraph * binIndex + idx;
    contextId          = graphId / graphCountPerContext;
    auto contextHandle = m_contextVec[contextId];
    auto graphHandle   = m_graphsInfo[graphId]->graph;
    if (contextHandle == nullptr || graphHandle == nullptr) {
      QNN_ERROR(" contexthandle or graph handle is null for patch no = %d ", graphId);
      return false;
    }

    QnnContext_Buffer_t qnnBuffer;
    qnnBuffer.version               = QNN_CONTEXT_BUFFER_VERSION_1;
    qnnBuffer.v1.memType            = QNN_CONTEXTMEMTYPE_RAW;
    qnnBuffer.v1.binaryBuf.dataSize = bufferSize;
    qnnBuffer.v1.binaryBuf.data     = static_cast<void*>(buffer.get());

    if (graphSwitch && lazyLora == "lazy") {
      // Cache info for deferred call during execute
      m_adapterCache[graphHandle] = std::make_tuple(contextHandle, qnnBuffer, graphId, false);
    } else {
      auto errorCode = m_qnnInterface.contextApplyBinarySection(contextHandle,
                                                                graphHandle,
                                                                QNN_CONTEXT_SECTION_UPDATABLE,
                                                                &qnnBuffer,
                                                                nullptr,  // profile handle is null
                                                                nullptr   // signal handle is null
      );
      if (errorCode != QNN_SUCCESS) {
        QNN_ERROR("Could not Apply Patch for graph = %d error code = %zu ", graphId, errorCode);
        return false;
      }
    }
  }
  if (updateIOEncodings(buffer, bufferSize, numAdapterGraph * binIndex) == false) {
    QNN_ERROR("qnn-htp: Adapter updateIOEncodings failed");
    return false;
  }
  return true;
}

bool QnnApi::setPerfProfile(qualla::PerformanceProfile& perfProfile) {
  qnn::tools::netrun::PerfProfile qnnPerfProfile =
      qualla::QnnUtils::quallaToQnnPerformanceProfile(perfProfile);
  if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
    if (qnnPerfProfile != m_perfProfile)
      m_backendExtensions->interface()->setPerfProfile(qnnPerfProfile);
  }
  m_perfProfile = qnnPerfProfile;
  return true;
}

qualla::PerformanceProfile QnnApi::getPerfProfile() {
  return qualla::QnnUtils::qnnToQuallaPerformanceProfile(m_perfProfile);
}

bool QnnApi::applyCachedAdapter(Qnn_GraphHandle_t graphHandle) {
  auto contextHandle = std::get<0>(m_adapterCache[graphHandle]);
  auto qnnBuffer     = std::get<1>(m_adapterCache[graphHandle]);
  uint32_t graphId   = std::get<2>(m_adapterCache[graphHandle]);

  auto errorCode = m_qnnInterface.contextApplyBinarySection(contextHandle,
                                                            graphHandle,
                                                            QNN_CONTEXT_SECTION_UPDATABLE,
                                                            &qnnBuffer,
                                                            nullptr,  // profile handle is null
                                                            nullptr   // signal handle is null
  );
  if (errorCode != QNN_SUCCESS) {
    QNN_ERROR("Could not Apply Patch for graph = %d error code = %zu ", graphId, errorCode);
    return false;
  }
  std::get<3>(m_adapterCache[graphHandle]) = true;
  return true;
}

bool QnnApi::updateIOEncodings(std::shared_ptr<uint8_t>& buffer,
                               uint64_t bufferSize,
                               uint32_t graphIndex) {
  QNN_DEBUG("Applying adapter Encodings");
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle for context index = %u", graphIndex);
    return false;
  }
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (QNN_SUCCESS !=
      m_qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle,
                                                      static_cast<void*>(buffer.get()),
                                                      bufferSize,
                                                      &binaryInfo,
                                                      &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info for context index = %u", graphIndex);
    return false;
  }
  if (!updateMetaDataToGraphsInfo(binaryInfo, m_graphsInfo, graphIndex)) {
    QNN_ERROR("Failed to copy metadata for graph index = %u", graphIndex);
    return false;
  }
  m_qnnSystemInterface.systemContextFree(sysCtxHandle);
  sysCtxHandle = nullptr;
  QNN_DEBUG(" updateIOEncodings success ");
  return true;
}

bool QnnApi::createFromBinaryGpu(std::vector<std::string> cachedBinariesPathVec) {
  auto _start = std::chrono::steady_clock::now();

  if (nullptr == m_qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return false;
  }

  graphCountPerContext = getGraphCountPerContext();

  for (size_t contextIdx = 0; contextIdx < cachedBinariesPathVec.size(); contextIdx++) {
    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t> buffer{nullptr};
    uint32_t graphsCount;

    // read serialized binary into a byte buffer
    bufferSize = getFileSize(cachedBinariesPathVec[contextIdx]);
    if (0 == bufferSize) {
      QNN_ERROR("Received path to an empty file for context index = %zu. Nothing to deserialize.",
                contextIdx);
      return false;
    }

    // inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
      QNN_ERROR("Could not create system handle for context index = %zu", contextIdx);
      return false;
    }

    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    bool useMmap           = true;
    const bool graphSwitch = false;

    if (!mapAndGetContextBinaryInfo(useMmap,
                                    buffer,
                                    cachedBinariesPathVec[contextIdx],
                                    bufferSize,
                                    contextIdx,
                                    graphSwitch,
                                    sysCtxHandle,
                                    &binaryInfo)) {
      QNN_ERROR("Failed to map context Binary for contextIdx: %zu", contextIdx);
      return false;
    }

    qnn_wrapper_api::GraphInfo_t** graphsInfo;
    if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount)) {
      QNN_ERROR("Failed to copy metadata for graph index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    if (graphCountPerContext == -1) {
      graphCountPerContext = graphsCount;
      m_graphsInfo         = (qnn_wrapper_api::GraphInfo_t**)calloc(
          graphCountPerContext * cachedBinariesPathVec.size(),
          sizeof(qnn_wrapper_api::GraphInfo_t*));
    } else if (graphCountPerContext != graphsCount) {
      QNN_ERROR("Different len(graphs) found in different context files. Found %u vs %u",
                graphsCount,
                graphCountPerContext);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }
    m_qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (nullptr == m_qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr for context index = %zu", contextIdx);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }
    Qnn_ContextHandle_t contextHandle{nullptr};
    auto _stop     = std::chrono::steady_clock::now();
    auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count();
    (void)_duration;
    QNN_DEBUG("Loading contexts[%lu] took: %lld us", contextIdx, _duration);

    auto start = std::chrono::steady_clock::now();

    auto errCode = m_qnnInterface.contextCreateFromBinary(m_backendHandle,
                                                          m_deviceHandle,
                                                          nullptr,
                                                          (const void*)buffer.get(),
                                                          bufferSize,
                                                          &contextHandle,
                                                          nullptr  // profile handle

    );

    if (errCode != QNN_SUCCESS) {
      QNN_ERROR("Could not create context from binary for context index = %zu : err %d",
                contextIdx,
                (int)errCode);
      freeGraphsInfo(&graphsInfo, graphsCount);
      if (contextIdx > 0) freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }

    auto stop     = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    (void)duration;
    QNN_DEBUG("Initializing context[%lu] with %u graphs took: %lld us",
              contextIdx,
              graphsCount,
              duration);

    for (size_t n_graph = 0; n_graph < graphsCount; n_graph++) {
      // Allocate inputTensors and outputTensors
      qnn_wrapper_api::GraphInfo_t* cur_graph = graphsInfo[n_graph];

      m_graphsInfo[m_graphsCount++] = cur_graph;
      m_contextMap[cur_graph]       = contextHandle;
    }
    m_contextVec.push_back(contextHandle);
  }

  m_isContextCreated = true;

  QNN_DEBUG("Initialized %u graphs from %lu contexts", m_graphsCount, cachedBinariesPathVec.size());

  if (nullptr == m_qnnInterface.graphRetrieve) {
    QNN_ERROR("graphRetrieveFnHandle is nullptr.");
    freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    return false;
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (!m_graphsInfo ||
        QNN_SUCCESS != m_qnnInterface.graphRetrieve(m_contextVec[graphIdx / graphCountPerContext],
                                                    m_graphsInfo[graphIdx]->graphName,
                                                    &(m_graphsInfo[graphIdx]->graph))) {
      QNN_ERROR("Unable to retrieve graph handle for graph index = %zu", graphIdx);
      freeGraphsInfo(&m_graphsInfo, m_graphsCount);
      return false;
    }
  }

  return true;
}

bool QnnApi::initializeGpu(std::string backendPath,
                           std::vector<std::string> modelPathOrCachedBinaryPath,
                           bool debug_qnn,
                           uint32_t logLevel,
                           LogCallback inLogCallBack) {
  if (modelPathOrCachedBinaryPath.size() != 1) {
    QNN_ERROR("Multiple Files not supported for now!!");
    return false;
  }

  if (false == getQnnInterface(backendPath)) {
    QNN_ERROR("Qnn getQnnInterface FAILED!");
    return false;
  }

  const std::string systemLibraryPath = "libQnnSystem.so";
  if (false == getQnnSystemInterface(systemLibraryPath)) {
    QNN_ERROR("Qnn getQnnSystemInterface FAILED!");
    return false;
  }

  QnnLog_Level_t qnnLogLevel = static_cast<QnnLog_Level_t>(logLevel);
  if (false == initializeLogging(qnnLogLevel, debug_qnn, inLogCallBack)) {
    QNN_ERROR("Unable to Initialize logging in backend");
    return false;
  }

  // Initialize Backend
  if (false == initializeBackend()) {
    QNN_ERROR("Qnn initializeBackend FAILED!");
    return false;
  }

  if (false == createFromBinaryGpu(modelPathOrCachedBinaryPath)) {
    QNN_ERROR("Create From Binary FAILED!");
    return false;
  }

  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    m_graphNameToIndex[m_graphsInfo[graphIdx]->graphName] = graphIdx;
  }
  QNN_DEBUG("Model Initialized");

  return true;
}

bool QnnApi::setOemKey(const std::string& oemKey) {
  if (nullptr == m_qnnInterface.propertyHasCapability) {
    QNN_ERROR("propertyHasCapability is nullptr.");
    return false;
  }

  if (m_qnnInterface.propertyHasCapability(QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS) !=
      QNN_PROPERTY_SUPPORTED) {
    QNN_ERROR("Backend does not support QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS");
  }

  if (nullptr == m_qnnInterface.backendSetConfig) {
    QNN_ERROR("backendSetConfig is nullptr.");
    return false;
  }

  QnnBackend_Config_t backendConfig           = QNN_BACKEND_CONFIG_INIT;
  std::string oem_string                      = "oem:" + oemKey;
  backendConfig.option                        = QNN_BACKEND_CONFIG_OPTION_PLATFORM;
  backendConfig.platformOption                = oem_string.c_str();
  const QnnBackend_Config_t* backendConfigs[] = {&backendConfig, nullptr};

  Qnn_ErrorHandle_t err = m_qnnInterface.backendSetConfig(
      m_backendHandle, (const QnnBackend_Config_t**)(backendConfigs));
  if (QNN_SUCCESS != err) {
    QNN_ERROR("backendSetConfig for OEM key failed.");
    return false;
  }
  return true;
}

bool QnnApi::setExecutionPriority(Qnn_Priority_t priority) {
  if (nullptr == m_qnnInterface.propertyHasCapability) {
    QNN_ERROR("propertyHasCapability is nullptr.");
    return false;
  }

  if (m_qnnInterface.propertyHasCapability(QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION) !=
      QNN_PROPERTY_SUPPORTED) {
    QNN_ERROR("Backend does not support QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION");
  }

  if (nullptr == m_qnnInterface.contextSetConfig) {
    QNN_ERROR("contextSetConfig is nullptr.");
    return false;
  }

  QnnContext_Config_t contextConfig           = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option                        = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  contextConfig.priority                      = priority;
  const QnnContext_Config_t* contextConfigs[] = {&contextConfig, nullptr};

  for (auto ctxtHandle : m_contextVec) {
    Qnn_ErrorHandle_t err =
        m_qnnInterface.contextSetConfig(ctxtHandle, (const QnnContext_Config_t**)contextConfigs);
    if (QNN_SUCCESS != err) {
      QNN_ERROR("contextSetConfig for priority failed.");
      return false;
    }
  }

  return true;
}

static std::string dataFormatToString(const Qnn_TensorDataFormat_t format) {
  switch (format) {
    case QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER:
      return "QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER";
    case QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT:
      return "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT";
    default:
      return std::to_string(format);
  }
}

bool QnnApi::initializeScorer(
    const std::string scorerPath,
    const std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>>& scorerAllocs,
    std::map<uint32_t, uint8_t*>& scorerMemptrs,
    const size_t expectedContextLength,
    const Qnn_TensorDataFormat_t ExpectedCacheFormat) {
  // Current design assumes a scorer network that takes in n_layer anchors + n_layer keys
  // The output is n_layer scores, where score = anchor @ key

  // Load the model
  uint64_t scorerSize = getFileSize(scorerPath);
  if (scorerSize == 0) {
    QNN_ERROR("Scorer file %s couldn't be read, or is empty", scorerPath.c_str());
    return false;
  }

  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle for scorer");
    return false;
  }

  std::shared_ptr<uint8_t> buffer;
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  if (!mapAndGetContextBinaryInfo(m_mmapContextBins,
                                  buffer,
                                  scorerPath,
                                  scorerSize,
                                  m_contextVec.size(),
                                  false,
                                  sysCtxHandle,
                                  &binaryInfo)) {
    QNN_ERROR("Failed to map context Binary for scorer");
    return false;
  };

  uint32_t graphsCount;
  qnn_wrapper_api::GraphInfo_t** graphsInfo{nullptr};
  if (!copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount)) {
    QNN_ERROR("Failed to copy metadata for scorer");
    return false;
  }

  if (QNN_SUCCESS != m_qnnSystemInterface.systemContextFree(sysCtxHandle)) {
    QNN_ERROR("Could not free system context object");
    return false;
  }

  QnnContext_Config_t** contextConfigs = nullptr;
  uint32_t contextConfigCount          = 0;
  if (!getContextConfigs(&contextConfigs, contextConfigCount)) {
    QNN_ERROR("Couldn't popular context configs for scorer");
    return false;
  }

  QnnContext_Config_t** allContextConfigs{nullptr};
  if (!mergeAllContextConfigs(&allContextConfigs, nullptr, contextConfigs, 0, contextConfigCount)) {
    QNN_ERROR("Error merging context configs for scorer");
    return false;
  }

  Qnn_ContextHandle_t contextHandle{nullptr};
  Qnn_ErrorHandle_t errCode =
      m_qnnInterface.contextCreateFromBinary(m_backendHandle,
                                             m_deviceHandle,
                                             (const QnnContext_Config_t**)allContextConfigs,
                                             (const void*)buffer.get(),
                                             scorerSize,
                                             &contextHandle,
                                             nullptr);
  if (errCode != QNN_SUCCESS) {
    QNN_ERROR("Couldn't initialize scorer %s : err %llu",
              scorerPath.c_str(),
              static_cast<unsigned long long>(errCode));
    return false;
  };

  m_contextVec.push_back(contextHandle);
  buffer.reset();

  m_scorer = graphsInfo[0];
  errCode  = m_qnnInterface.graphRetrieve(contextHandle, m_scorer->graphName, &(m_scorer->graph));
  if (errCode != QNN_SUCCESS) {
    QNN_ERROR("Unable to retrieve scorer graph handle");
    return false;
  }

  std::map<std::string, std::tuple<int, size_t, size_t>> scorerAllocMap;
  for (size_t idx = 0; idx < m_scorer->numInputTensors; idx++) {
    auto tensor = qualla::QnnUtils::Tensor(&m_scorer->inputTensors[idx]);

    // Parse the layer index based on the layer name. We expect anchor_0_in and keys_0_in
    std::string tname    = tensor.tensor->v1.name;  // Use std::string for bounds checking
    const uint32_t index = qualla::QnnUtils::parseNumberFromString<1>(tname)[0] << 16;

    // Map the anchor and past_key tensor to use the same buffer as the LLM model
    if (tname.starts_with("anchor")) {
      const auto& [alloc_idx, offset] = scorerAllocs.at(index)[0];
      scorerAllocMap[tname]           = {alloc_idx, offset, tensor.dims.getAlignedSize()};
      // m_ioBufferMgr->useSameMemory(&tensor, scorerTensors[index][0]);
    } else if (tname.starts_with("key") || tname.starts_with("past")) {
      if (tensor.tensor->v1.dataFormat != ExpectedCacheFormat) {
        QNN_ERROR("Scorer network KV dataFormat does not match the model. Expected %s, found %s",
                  dataFormatToString(ExpectedCacheFormat).c_str(),
                  dataFormatToString(tensor.tensor->v1.dataFormat).c_str());
        return false;
      }
      const auto& [alloc_idx, offset] = scorerAllocs.at(index)[1];
      scorerAllocMap[tname]           = {alloc_idx, offset, tensor.dims.getAlignedSize()};
      // m_ioBufferMgr->useSameMemory(&tensor, scorerTensors[index][1]);
    }
  }
  m_ioBufferMgr->mapFusedBufferOffset(m_scorer, contextHandle, scorerAllocMap);

  // Score tensor outputs need to be allocated
  size_t total_size = 0;
  std::map<uint32_t, std::tuple<size_t, size_t, Qnn_Tensor_t*>> score_tensor_offsets;
  for (size_t idx = 0; idx < m_scorer->numOutputTensors; idx++) {
    Qnn_Tensor_t& tensor = m_scorer->outputTensors[idx];

    // Parse the layer index based on the layer name. We expect anchor_0_in and keys_0_in
    std::string tname    = tensor.v1.name;  // Use std::string for readability
    const uint32_t index = qualla::QnnUtils::parseNumberFromString<1>(tname)[0] << 16;

    auto score_tensor       = qualla::QnnUtils::Tensor(&tensor);
    const size_t score_size = score_tensor.dims.getAlignedSize();
    const size_t scoreCount = score_tensor.dims.channel;

    if (scoreCount != expectedContextLength) {
      QNN_ERROR(
          "Error validating scoring network. Expected %zu scores, but network produces %zu scores.",
          expectedContextLength,
          scoreCount);
      return false;
    }

    score_tensor_offsets[index] = {score_size, total_size, &tensor};
    total_size += score_size;
  }

  // Allocate buffer for scores
  int32_t score_fd;
  uint8_t* score_memptr = (uint8_t*)m_ioBufferMgr->allocateTensorFusedBuffer(total_size, &score_fd);

  // Register and accumulate set of all score buffers
  for (const auto& [index, score_tensor_offset] : score_tensor_offsets) {
    const auto& [alloc_size, alloc_offset, tensor] = score_tensor_offset;

    scorerMemptrs[index] = score_memptr + alloc_offset;
    if (!m_ioBufferMgr->mapFusedBufferOffset(
            tensor, alloc_size, score_fd, alloc_offset, total_size, score_memptr, contextHandle)) {
      QNN_ERROR(
          "Error registering output tensor %s for scorer %s", tensor->v1.name, m_scorer->graphName);
      return false;
    }
  }

  return true;
}

bool QnnApi::executeScorer() {
  QNN_DEBUG("Executing scorer %s", m_scorer->graphName);
#if NSP_LOG_LEVEL > 1
  auto start = std::chrono::steady_clock::now();
#endif

  std::map<std::string, std::pair<double, uint16_t>> timeLogs;
  if (!graphExecute(m_scorer, m_scorer->inputTensors, m_scorer->outputTensors, timeLogs)) {
    QNN_ERROR("Error executing scorer network");
    return false;
  }

#if NSP_LOG_LEVEL > 1
  auto stop     = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  QNN_DEBUG("graphExecute[%s] took: %lld us\n", m_scorer->graphName, duration);
#endif
  return true;
}

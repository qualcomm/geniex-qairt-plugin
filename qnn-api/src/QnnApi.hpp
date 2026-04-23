//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "BackendExtensions.hpp"
#include "IOTensor.hpp"
#include "QnnConfig.hpp"
#include "QnnHtpDevice.h"
#include "QnnHtpPerfInfrastructure.h"
#include "QnnWrapperUtils.hpp"
#include "config/ConfigList.hpp"
#include "config/ContextConfig.hpp"
#include "qnn-utils.hpp"

#define QNN_IO_TENSOR_DEBUG 0

enum KVManagerMode { POINTER_SHIFT = 0x0, SHIFT_CONCAT = 0x1, SMART_MASK = 0x2, NATIVE_KV = 0x3 };

using qualla::QnnUtils::QuantParam;

static std::map<Qnn_DataType_t, size_t> g_qnnDataTypeToSize = {
    {QNN_DATATYPE_INT_8, 1},
    {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},
    {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},
    {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},
    {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},
    {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2},
    {QNN_DATATYPE_SFIXED_POINT_32, 4},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},
    {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4},
    {QNN_DATATYPE_BOOL_8, 1},
};

using ContextConfigList = ConfigList<QnnContext_Config_t>;

using LogCallback =
    std::function<void(const char* fmt, uint32_t level, uint64_t timestamp, va_list args)>;

void registerUserCb(LogCallback l);
void userLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args);

class QnnApi {
 private:
  const uint32_t s_graphConfigsReserveCount = 16;

  // Model vars
  typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                            uint32_t* numProviders);
  typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
      const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

  // Graph Related Function Handle Types
  typedef qnn_wrapper_api::ModelError_t (*ComposeGraphsFnHandleType_t)(
      Qnn_BackendHandle_t,
      QNN_INTERFACE_VER_TYPE,
      Qnn_ContextHandle_t,
      const qnn_wrapper_api::GraphConfigInfo_t**,
      const uint32_t,
      qnn_wrapper_api::GraphInfo_t***,
      uint32_t*,
      bool,
      QnnLog_Callback_t,
      QnnLog_Level_t);

  typedef qnn_wrapper_api::ModelError_t (*GenAIComposeGraphsFnHandleType_t)(
      Qnn_BackendHandle_t,
      QNN_INTERFACE_VER_TYPE,
      Qnn_ContextHandle_t,
      const qnn_wrapper_api::GraphConfigInfo_t**,
      const uint32_t,
      uint32_t* inputDim,
      uint32_t inputRank,
      uint32_t* outputDim,
      uint32_t outputRank,
      uint32_t* kvDim,
      uint32_t kvRank,
      uint32_t* kvScaleDim,
      Qnn_Param_t* params,
      uint32_t numParam,
      const char* modelName,
      qnn_wrapper_api::GraphInfo_t***,
      uint32_t*,
      bool,
      QnnLog_Callback_t,
      QnnLog_Level_t);

  typedef qnn_wrapper_api::ModelError_t (*FreeGraphInfoFnHandleType_t)(
      qnn_wrapper_api::GraphInfo_t***, uint32_t);

  void* m_libModelHandle{nullptr};
  void* m_backendHandle{nullptr};
  void* m_backendLibraryHandle{nullptr};

  QNN_INTERFACE_VER_TYPE m_qnnInterface{nullptr};
  QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface{nullptr};
  std::unique_ptr<BackendExtensions> m_backendExtensions{nullptr};
  ComposeGraphsFnHandleType_t m_composeGraphsFnHandle{nullptr};
  GenAIComposeGraphsFnHandleType_t m_genaiComposeGraphsFnHandle{nullptr};
  FreeGraphInfoFnHandleType_t m_freeGraphInfoFnHandle{nullptr};
  uint32_t m_backendId{0};
  Qnn_LogHandle_t m_logHandle{nullptr};
  Qnn_DeviceHandle_t m_deviceHandle{nullptr};

  Qnn_ProfileHandle_t m_profileBackendHandle{nullptr};

  std::vector<Qnn_ContextHandle_t> m_contextVec;
  std::unordered_map<qnn_wrapper_api::GraphInfo*, Qnn_ContextHandle_t> m_contextMap;
  uint32_t m_graphsCount{0};
  int32_t graphCountPerContext{-1};
  qnn_wrapper_api::GraphInfo_t** m_graphsInfo{nullptr};
  std::unordered_map<std::string, uint32_t> m_graphNameToIndex;
  std::unordered_map<std::string, qnn_wrapper_api::GraphInfo*> m_graphNameToInfo;
  std::unordered_map<std::string, uint32_t> m_graphNameToContextIdx;
  std::unordered_map<uint32_t, Qnn_ContextHandle_t> m_contextIdtoHandle;
  std::mutex m_updateCallBackMutex;

  // Useful Structure for IO Esimtation
  std::unordered_map<int, qualla::QnnUtils::TensorMap>
      m_graphtoIOMap;  // stores {GraphId -> IOTensorMap}
  typedef int CtxBitVector;
  std::map<CtxBitVector, std::map<std::string, size_t>>
      m_contextAllocMap;  // stores {Translated ContextId -> {Tensor name, size}}
  std::map<std::string, std::pair<int, size_t>>
      m_tensorAllocInfo;  // stores {Tensor name -> (fd of RPC buffer, offset)}
  std::unordered_map<uint32_t, uint32_t>
      m_graphIdxToContextIdx;  // stores {Graph Idx -> Context Idx}
  std::unordered_map<std::string, std::shared_ptr<uint8_t>> m_adapterNameToBuffer;

  uint32_t m_backendConfigCount{0};
  QnnBackend_Config_t** m_backendConfigs{nullptr};

  QnnHtpDevice_PerfInfrastructure_t* m_perfInfra{nullptr};
  uint32_t m_powerConfigId = 1;

  // Useful Structure for IO Esimtation
  IOTensor* m_ioBufferMgr{nullptr};
  int32_t m_ctxSize{-1};
  int32_t m_kvDim{-1};
  bool m_loraWeightEnabled{false};
  bool m_lmHeadWeightInput{false};
  KVManagerMode m_kvUpdateMethod{SMART_MASK};

  // For KeyDiff, keep track of the scorer network graph
  qnn_wrapper_api::GraphInfo* m_scorer{nullptr};

  bool m_isLogInitialized{false};
  bool m_isBackendInitialized{false};
  bool m_isContextCreated{false};
  // True when contexts were loaded from binary (createFromBinaryHtp / async).
  // In that path, QnnHtpNetRunExtensions::beforeContextFree() crashes because
  // afterCreateFromBinary() does not register handles the way afterContextCreate()
  // does. Skip the before/afterContextFree extension hooks for this path.
  bool m_contextCreatedFromBinary{false};

  // Variable to keep track of debug mode
  bool m_DebugModeRequested;
  bool m_debugQnn{false};

  // Variable to indicate whether to mmap context bins or read them in memory
  bool m_mmapContextBins;
  bool m_isDeviceCreated = false;

  qnn::tools::netrun::PerfProfile m_perfProfile;

  std::vector<std::pair<uint8_t*, uint64_t>> m_contextBinBuffersToBeCleared;

  void setDeviceStatus(bool status) { m_isDeviceCreated = status; }
  bool getDeviceStatus() { return m_isDeviceCreated; }
  bool getContextConfigs(QnnContext_Config_t*** configs,
                         uint32_t& contextConfigCount,
                         bool graphSwitching                              = false,
                         const std::vector<std::string>& execSelectGraphs = {},
                         bool loadSelectGraphs                            = false);
  bool getContextConfigs(ContextConfigList& configList,
                         bool graphSwitching                              = false,
                         const std::vector<std::string>& execSelectGraphs = {},
                         bool loadSelectGraphs                            = false);
  bool mergeAllContextConfigs(QnnContext_Config_t*** allCustomContextConfigs,
                              QnnContext_Config_t** customConfigs,
                              QnnContext_Config_t**& contextConfigs,
                              uint32_t customConfigCount,
                              uint32_t contextConfigCount);
  bool freeContextConfigs(QnnContext_Config_t** contextConfigs, uint32_t contextConfigCount);
  bool setGraphConfigsBeforeExecute(Qnn_GraphHandle_t graphHandle,
                                    QnnGraph_Config_t** graphConfigs,
                                    uint32_t configCount);

  bool getQnnInterface(std::string backendPath);
  bool getQnnSystemInterface(std::string systemLibraryPath);
  bool loadModel(std::string model_path);
  bool initializeLogging(const QnnLog_Level_t& logLevel,
                         bool debug_qnn,
                         LogCallback userCallback = nullptr);
  void terminateLog();
  bool initializeBackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                   qnn::tools::netrun::PerfProfile parsedPerfProfile,
                                   bool debug_qnn,
                                   QnnLog_Level_t qnnLogLevel);
  bool initializeBackend();
  bool terminateBackend();
  bool createDevice();
  bool freeDevice();
  bool createContext();
  bool freeContext();
  bool composeGraphs(std::vector<GraphConfigs> graphConfigs);
  bool composeGraphs(std::vector<GraphConfigs> graphConfigs,
                     uint32_t* inputDim,
                     uint32_t inputRank,
                     uint32_t* outputDim,
                     uint32_t outputRank,
                     uint32_t* kvDim,
                     uint32_t kvRank,
                     uint32_t* kvScaleDim,
                     Qnn_Param_t* params,
                     uint32_t numParams);

  bool mapAndGetContextBinaryInfo(const bool useMmap,
                                  std::shared_ptr<uint8_t>& buffer,
                                  const std::string& binaryPath,
                                  const uint64_t bufferSize,
                                  const size_t contextIdx,
                                  const bool graphSwitching,
                                  QnnSystemContext_Handle_t sysCtxHandle,
                                  const QnnSystemContext_BinaryInfo_t** binaryInfo);

  bool parseIOTensorsAndAccumulate();
  bool registerTensorsWithBackend(uint32_t& graphIdx);

  bool finalizeGraphs();
  bool finalizeCpuGraphs();
  bool initializePerformance();
  bool destroyPerformance();
  bool boostPerformance();
  bool resetPerformance();
  bool checkCapabilityOfCreateAsync(bool& propRet);

  bool initProfiling();
  bool extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle,
                                   std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                   std::string graphName);
  bool extractProfilingSubEvents(QnnProfile_EventId_t profileEventId,
                                 std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                                 std::string graphName);
  bool extractProfilingEvent(QnnProfile_EventId_t profileEventId,
                             std::map<std::string, std::pair<double, uint16_t>>& timeLogs,
                             std::string graphName);
  bool extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);
  bool extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);
  bool extractProfilingEvent(QnnProfile_EventId_t profileEventId);

  Qnn_ContextHandle_t getContextWithId(uint32_t contextId) {
    return m_contextIdtoHandle[contextId];
  }

 public:
  QnnApi(){};
  ~QnnApi();

  bool freeGraphs();
  bool freeCurrentContext(std::string graphName);
  bool freeCurrentGraph(std::string graphName);
  static QnnApi& getInstance();
  static void contextNotifyFn(Qnn_ContextHandle_t context,
                              Qnn_GraphHandle_t graph,
                              const char* graph_name,
                              QnnContext_createFromBinaryAsyncNotifyType_t completeType,
                              void* notifyParam,
                              Qnn_ErrorHandle_t status);
  bool createFromBinaryGpu(std::vector<std::string> cachedBinariesPathVec);
  bool createFromBinaryHtp(std::vector<std::string> cachedBinariesPathVec,
                           int64_t spill_fill_buffer_size                   = 0,
                           uint64_t mmap_budget                             = 0,
                           uint32_t dataAlignmentSize                       = 0,
                           bool graphSwitching                              = false,
                           const std::vector<std::string>& execSelectGraphs = {},
                           bool loadSelectGraphs                            = false,
                           bool skipLoraValidation                          = false);
  bool createFromBinaryListAsyncHtp(std::vector<std::string> cachedBinariesPathVec,
                                    int64_t spill_fill_buffer_size                   = 0,
                                    uint64_t mmap_budget                             = 0,
                                    bool graphSwitching                              = false,
                                    const std::vector<std::string>& execSelectGraphs = {},
                                    bool loadSelectGraphs                            = false,
                                    bool skipLoraValidation                          = false);
  bool initializeGpu(std::string backendPath,
                     std::vector<std::string> modelPathOrCachedBinaryPath,
                     bool debug_qnn            = false,
                     uint32_t logLevel         = 1,
                     LogCallback inLogCallBack = nullptr);
  bool initializeCpu(std::string backendPath,
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
                     bool debug_qnn            = false,
                     uint32_t logLevel         = 1,
                     LogCallback inLogCallBack = nullptr);
  bool initializeHtp(
      std::string backendPath,
      std::vector<std::string> modelPathOrCachedBinaryPathVec,
      BackendExtensionsConfigs backendExtensionsConfig,
      qnn::tools::netrun::PerfProfile parsedPerfProfile = qnn::tools::netrun::PerfProfile::BURST,
      std::vector<GraphConfigs> graphConfigs            = {},
      bool loadFromCachedBinary                         = false,
      std::string systemLibraryPath                     = "",
      bool debugModeRequested                           = false,
      int64_t spill_fill_buffer_size                    = 0,
      uint32_t dataAlignmentSize                        = 0,
      bool mmapContextBins                              = false,
      bool asyncInit                                    = true,
      uint64_t mmap_budget                              = 0,
      bool debug_qnn                                    = false,
      bool graphSwitching                               = false,
      const std::vector<std::string>& execSelectGraphs  = {},
      bool loadSelectGraphs                             = false,
      bool skipLoraValidation                           = false,
      uint32_t logLevel                                 = 1,
      LogCallback inLogCallBack                         = nullptr);

  bool registerOpPackage(std::string opPackagePath);

  void setIOTensorBufferMgr(IOTensor* ioBufferMgr) { m_ioBufferMgr = ioBufferMgr; }

  void setKVDim(int32_t kvDim) { m_kvDim = kvDim; }

  void setContextSize(int32_t ctxSize) { m_ctxSize = ctxSize; }

  void setKVUpdateMethod(KVManagerMode kvUpdateMethod) { m_kvUpdateMethod = kvUpdateMethod; }

  std::map<std::string, std::pair<int, size_t>>* getTensorAllocInfo() { return &m_tensorAllocInfo; }

  bool getLmHeadWeightInputEnabled() { return m_lmHeadWeightInput; }

  bool getLoraWeightEnabled() { return m_loraWeightEnabled; }

  bool graphExecute(Qnn_Tensor_t* input,
                    Qnn_Tensor_t* output,
                    std::string graphName,
                    std::map<std::string, std::pair<double, uint16_t>>& timeLogs);
  bool graphExecute(qnn_wrapper_api::GraphInfo_t* graph_info,
                    const Qnn_Tensor_t* input,
                    Qnn_Tensor_t* output,
                    std::map<std::string, std::pair<double, uint16_t>>& timeLogs);

  // Lazy LoRA variables
  std::unordered_map<Qnn_GraphHandle_t,
                     std::tuple<Qnn_ContextHandle_t, QnnContext_Buffer_t, uint32_t, bool>>
      m_adapterCache;

  bool applyBinarySection(uint32_t binIndex,
                          std::string binSectionPath,
                          bool useMmap,
                          bool graphSwitch,
                          std::string& lazyLora);

  bool applyBinarySection(uint32_t graphId, std::string binSectionPath);

  bool applyCachedAdapter(Qnn_GraphHandle_t graphHandle);

  bool applyBinarySection(std::string graphName, std::string binSectionPath);

  bool setPerfProfile(qualla::PerformanceProfile& perfProfile);

  qualla::PerformanceProfile getPerfProfile();

  QNN_INTERFACE_VER_TYPE* getQnnInterfaceVer() { return &m_qnnInterface; };
  qnn_wrapper_api::GraphInfo_t**& getGraphsInfo() { return m_graphsInfo; };
  uint32_t getGraphsCount() { return m_graphsCount; };
  int32_t getGraphCountPerContext() { return graphCountPerContext; }
  std::vector<Qnn_ContextHandle_t>& getContexts() { return m_contextVec; };
  const Qnn_ContextHandle_t getContexts(qnn_wrapper_api::GraphInfo_t* const graph) {
    return m_contextMap.at(graph);
  };
  const uint32_t getContextIndex(qnn_wrapper_api::GraphInfo_t* const graph) {
    return m_graphNameToContextIdx.at(graph->graphName);
  }

  void updateContext(Qnn_ContextHandle_t context, uint32_t contextId) {
    std::lock_guard<std::mutex> lock(m_updateCallBackMutex);
    m_contextVec.push_back(context);
    m_contextIdtoHandle[contextId] = context;
  }

  void updateQnnApiGraphsandContextsInfo(std::string graphName,
                                         Qnn_GraphHandle_t graph,
                                         uint32_t contextId) {
    // set graph handle to GraphInfo
    std::lock_guard<std::mutex> lock(m_updateCallBackMutex);
    m_graphNameToInfo[graphName]->graph = graph;
    m_graphNameToContextIdx[graphName]  = contextId;
    m_graphsCount++;
  }

  static inline size_t getDataTypeSize(const Qnn_DataType_t& datatype) {
    return g_qnnDataTypeToSize[datatype];
  }
  static inline std::string getTensorName(const Qnn_Tensor_t& tensor) {
    return QNN_TENSOR_GET_NAME(tensor);
  }
  static bool getTensorQuantParams(const Qnn_Tensor_t* tensor,
                                   std::vector<QuantParam>& quantParamsVec);
  static bool getTensorShape(std::vector<size_t>& tensorDims, const Qnn_Tensor_t& tensor);
  static inline Qnn_DataType_t getTensorDtype(const Qnn_Tensor_t* tensor) {
    return QNN_TENSOR_GET_DATA_TYPE(tensor);
  }

  bool getTensorNameAndShape(std::string& tensorName,
                             std::vector<size_t>& tensorDims,
                             Qnn_Tensor_t& tensor);
  static void emptyLogCallback(const char* fmt,
                               QnnLog_Level_t level,
                               uint64_t timestamp,
                               va_list args){};
  bool updateIOEncodings(std::shared_ptr<uint8_t>& buffer,
                         uint64_t bufferSize,
                         uint32_t graphIndex);

  bool setOemKey(const std::string& oemKey);
  bool setExecutionPriority(Qnn_Priority_t priority);

  // KeyDiff - scoring network. Initialize and execute for each ctx_size and each layer
  bool initializeScorer(
      const std::string scorerPath,
      const std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>>& scorerAllocs,
      std::map<uint32_t, uint8_t*>& scorerMemptrs,
      const size_t expectedContextLength,
      const Qnn_TensorDataFormat_t ExpectedCacheFormat);
  bool executeScorer();
};

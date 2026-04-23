//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <errno.h>
#include <fcntl.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "QnnApiUtils.hpp"
#include "QnnTypeMacros.hpp"

#ifdef _WIN32
#include <windows.h>
#define __open   ::_open
#define __strdup ::_strdup
#else
#include <sys/mman.h>
#include <unistd.h>
#define __open   ::open
#define __strdup ::strdup
#endif

bool freeQnnTensorWrapper(Qnn_Tensor_t& tensor) {
  // free all pointer allocations in struct
  if (nullptr != QNN_TENSOR_GET_NAME(tensor)) {
    free((void*)QNN_TENSOR_GET_NAME(tensor));
  }
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
  return true;
}

bool freeQnnTensorWrappers(Qnn_Tensor_t*& tensorWrappers, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensorWrapper(tensorWrappers[i]);
  }
  free(tensorWrappers);

  return true;
}

bool freeGraphsInfo(qnn_wrapper_api::GraphInfoPtr_t** graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    if (nullptr != (*graphsInfo)[i]) {
      free((*graphsInfo)[i]->graphName);
      freeQnnTensorWrappers((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
      freeQnnTensorWrappers((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
      free((*graphsInfo)[i]);
      (*graphsInfo)[i] = nullptr;
    }
  }
  free(*graphsInfo);
  *graphsInfo = nullptr;

  return true;
}

bool freeGraphInfo(qnn_wrapper_api::GraphInfo_t* graphInfo) {
  if (graphInfo == nullptr) {
    return false;
  }
  if (nullptr != graphInfo->graphName) {
    free(graphInfo->graphName);
  }
  freeQnnTensorWrappers(graphInfo->inputTensors, graphInfo->numInputTensors);
  freeQnnTensorWrappers(graphInfo->outputTensors, graphInfo->numOutputTensors);
  free(graphInfo);
  return true;
}

bool updateTensorInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                      Qnn_Tensor_t* tensorWrappers,
                      uint32_t tensorsCount) {
  for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
    QNN_DEBUG("Extracting tensorInfo for tensor Idx: %d", (int)tIdx);
    Qnn_Tensor_t& tensor = tensorWrappers[tIdx];

    QNN_TENSOR_SET_ID(tensor, QNN_TENSOR_GET_ID(&tensorsInfoSrc[tIdx]));
    QNN_TENSOR_SET_TYPE(tensor, QNN_TENSOR_GET_TYPE(&tensorsInfoSrc[tIdx]));
    QNN_TENSOR_SET_DATA_FORMAT(tensor, QNN_TENSOR_GET_DATA_FORMAT(&tensorsInfoSrc[tIdx]));
    QNN_TENSOR_SET_DATA_TYPE(tensor, QNN_TENSOR_GET_DATA_TYPE(&tensorsInfoSrc[tIdx]));
    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition =
        QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding ==
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      qParams.quantizationEncoding =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding;
      qParams.scaleOffsetEncoding =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding ==
               QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      qParams.quantizationEncoding =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding;
      qParams.axisScaleOffsetEncoding.axis =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).axisScaleOffsetEncoding.axis;
      qParams.axisScaleOffsetEncoding.numScaleOffsets =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
              .axisScaleOffsetEncoding.numScaleOffsets;
      if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
              .axisScaleOffsetEncoding.numScaleOffsets > 0) {
        qParams.axisScaleOffsetEncoding.scaleOffset =
            (Qnn_ScaleOffset_t*)malloc(QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                                           .axisScaleOffsetEncoding.numScaleOffsets *
                                       sizeof(Qnn_ScaleOffset_t));
        if (qParams.axisScaleOffsetEncoding.scaleOffset) {
          for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                                         .axisScaleOffsetEncoding.numScaleOffsets;
               idx++) {
            qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                    .axisScaleOffsetEncoding.scaleOffset[idx]
                    .scale;
            qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                    .axisScaleOffsetEncoding.scaleOffset[idx]
                    .offset;
          }
        }
      }
    }
    QNN_TENSOR_SET_QUANT_PARAMS(tensor, qParams);
    QNN_TENSOR_SET_RANK(tensor, QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]));
    if (QNN_TENSOR_GET_RANK(tensorsInfoSrc[tIdx]) > 0) {
      if (QNN_TENSOR_GET_DIMENSIONS(tensor)) {
        memcpy(QNN_TENSOR_GET_DIMENSIONS(tensor),
               QNN_TENSOR_GET_DIMENSIONS(&tensorsInfoSrc[tIdx]),
               QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
      }
    }
  }
  return true;
}

bool copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                     Qnn_Tensor_t*& tensorWrappers,
                     uint32_t tensorsCount) {
  auto returnStatus = true;
  tensorWrappers    = (Qnn_Tensor_t*)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    QNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
      // QNN_DEBUG("Extracting tensorInfo for tensor Idx: %d", (int)tIdx);
      Qnn_Tensor_t& tensor = tensorWrappers[tIdx];
      tensor               = QNN_TENSOR_INIT;

      const char* tensorName = QNN_TENSOR_GET_NAME(&tensorsInfoSrc[tIdx]);
      if (!tensorName) {
        QNN_TENSOR_SET_NAME(tensor, nullptr);
      } else {
        QNN_TENSOR_SET_NAME(tensor, __strdup(tensorName));
      }

      QNN_TENSOR_SET_ID(tensor, QNN_TENSOR_GET_ID(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_TYPE(tensor, QNN_TENSOR_GET_TYPE(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_DATA_FORMAT(tensor, QNN_TENSOR_GET_DATA_FORMAT(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_DATA_TYPE(tensor, QNN_TENSOR_GET_DATA_TYPE(&tensorsInfoSrc[tIdx]));
      Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
      qParams.encodingDefinition =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).encodingDefinition;
      qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
      if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding ==
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qParams.quantizationEncoding =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding;
        qParams.scaleOffsetEncoding =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).scaleOffsetEncoding;
      } else if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding ==
                 QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qParams.quantizationEncoding =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding;
        qParams.axisScaleOffsetEncoding.axis =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).axisScaleOffsetEncoding.axis;
        qParams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                .axisScaleOffsetEncoding.numScaleOffsets;
        if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                .axisScaleOffsetEncoding.numScaleOffsets > 0) {
          qParams.axisScaleOffsetEncoding.scaleOffset =
              (Qnn_ScaleOffset_t*)malloc(QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                                             .axisScaleOffsetEncoding.numScaleOffsets *
                                         sizeof(Qnn_ScaleOffset_t));
          if (qParams.axisScaleOffsetEncoding.scaleOffset) {
            for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                                           .axisScaleOffsetEncoding.numScaleOffsets;
                 idx++) {
              qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                  QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                      .axisScaleOffsetEncoding.scaleOffset[idx]
                      .scale;
              qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                  QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                      .axisScaleOffsetEncoding.scaleOffset[idx]
                      .offset;
            }
          }
        }
      }
      QNN_TENSOR_SET_QUANT_PARAMS(tensor, qParams);
      QNN_TENSOR_SET_RANK(tensor, QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_DIMENSIONS(tensor, nullptr);
      if (QNN_TENSOR_GET_RANK(tensorsInfoSrc[tIdx]) > 0) {
        QNN_TENSOR_SET_DIMENSIONS(
            tensor,
            (uint32_t*)malloc(QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t)));
        if (QNN_TENSOR_GET_DIMENSIONS(tensor)) {
          memcpy(QNN_TENSOR_GET_DIMENSIONS(tensor),
                 QNN_TENSOR_GET_DIMENSIONS(&tensorsInfoSrc[tIdx]),
                 QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
        }
      }
    }
  }

  return returnStatus;
}

bool updateGraphInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                       qnn_wrapper_api::GraphInfo_t* graphInfoDst) {
  if (graphInfoSrc->graphInputs) {
    if (!updateTensorInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
  }
  if (graphInfoSrc->graphOutputs) {
    if (!updateTensorInfo(graphInfoSrc->graphOutputs,
                          graphInfoDst->outputTensors,
                          graphInfoSrc->numGraphOutputs)) {
      return false;
    }
  }
  return true;
}

bool updateGraphInfoV3(const QnnSystemContext_GraphInfoV3_t* graphInfoSrc,
                       qnn_wrapper_api::GraphInfo_t* graphInfoDst) {
  if (graphInfoSrc->graphInputs) {
    if (!updateTensorInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
  }
  if (graphInfoSrc->graphOutputs) {
    if (!updateTensorInfo(graphInfoSrc->graphOutputs,
                          graphInfoDst->outputTensors,
                          graphInfoSrc->numGraphOutputs)) {
      return false;
    }
  }
  return true;
}

bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                      qnn_wrapper_api::GraphInfo_t* graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName = __strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t* graphInfoSrc,
                      qnn_wrapper_api::GraphInfo_t* graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName = __strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

bool updateGraphInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                     const uint32_t numGraphs,
                     qnn_wrapper_api::GraphInfo_t** graphsInfo,
                     uint32_t& graphsCount) {
  for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
    if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      if (updateGraphInfoV1(&graphsInput[gIdx].graphInfoV1, graphsInfo[graphsCount]) == false) {
        return false;
      }
    }
    if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      if (updateGraphInfoV3(&graphsInput[gIdx].graphInfoV3, graphsInfo[graphsCount]) == false) {
        return false;
      }
    }
    graphsCount++;
  }
  return true;
}

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                    const uint32_t numGraphs,
                    qnn_wrapper_api::GraphInfo_t**& graphsInfo) {
  if (!graphsInput) {
    QNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (qnn_wrapper_api::GraphInfo_t**)calloc(numGraphs, sizeof(qnn_wrapper_api::GraphInfo_t*));
  if (nullptr == graphsInfo) {
    QNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
      QNN_DEBUG("Extracting graphsInfo for graph Idx: %d", (int)gIdx);
      auto graphInfoArr =
          (qnn_wrapper_api::GraphInfo_t*)malloc(sizeof(qnn_wrapper_api::GraphInfo_t));
      if (nullptr == graphInfoArr) {
        QNN_ERROR("Failure to allocate memory for graphInfoArr");
        returnStatus = false;
      }
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1, graphInfoArr);
      }
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        copyGraphsInfoV3(&graphsInput[gIdx].graphInfoV3, graphInfoArr);
      }
      graphsInfo[gIdx] = graphInfoArr;
    }
  }
  if (true != returnStatus) {
    QNN_DEBUG("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensorWrappers(graphsInfo[gIdx]->inputTensors, graphsInfo[gIdx]->numInputTensors);
          freeQnnTensorWrappers(graphsInfo[gIdx]->outputTensors,
                                graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }

  return true;
}

uint32_t getNumGraphInBinary(const QnnSystemContext_BinaryInfo_t* binaryInfo) {
  uint32_t numGraph = 0;
  if (nullptr == binaryInfo) {
    QNN_ERROR("binaryInfo is nullptr.");
    return false;
  }
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    numGraph = binaryInfo->contextBinaryInfoV1.numGraphs;
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    numGraph = binaryInfo->contextBinaryInfoV2.numGraphs;
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    numGraph = binaryInfo->contextBinaryInfoV3.numGraphs;
  }
  return numGraph;
}

bool updateMetaDataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                                qnn_wrapper_api::GraphInfo_t** graphsInfo,
                                uint32_t& graphsCount) {
  if (nullptr == binaryInfo) {
    QNN_ERROR("binaryInfo is nullptr.");
    return false;
  }
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    if (binaryInfo->contextBinaryInfoV1.graphs) {
      if (!updateGraphInfo(binaryInfo->contextBinaryInfoV1.graphs,
                           binaryInfo->contextBinaryInfoV1.numGraphs,
                           graphsInfo,
                           graphsCount)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    if (binaryInfo->contextBinaryInfoV2.graphs) {
      if (!updateGraphInfo(binaryInfo->contextBinaryInfoV2.graphs,
                           binaryInfo->contextBinaryInfoV2.numGraphs,
                           graphsInfo,
                           graphsCount)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    if (binaryInfo->contextBinaryInfoV3.graphs) {
      if (!updateGraphInfo(binaryInfo->contextBinaryInfoV3.graphs,
                           binaryInfo->contextBinaryInfoV3.numGraphs,
                           graphsInfo,
                           graphsCount)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      return true;
    }
  }
  QNN_ERROR("Unrecognized system context binary info version.");
  return false;
}

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                              qnn_wrapper_api::GraphInfo_t**& graphsInfo,
                              uint32_t& graphsCount) {
  if (nullptr == binaryInfo) {
    QNN_ERROR("binaryInfo is nullptr.");
    return false;
  }
  graphsCount = 0;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    if (binaryInfo->contextBinaryInfoV1.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                          binaryInfo->contextBinaryInfoV1.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    if (binaryInfo->contextBinaryInfoV2.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                          binaryInfo->contextBinaryInfoV2.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    if (binaryInfo->contextBinaryInfoV3.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV3.graphs,
                          binaryInfo->contextBinaryInfoV3.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV3.numGraphs;
      return true;
    }
  }
  QNN_ERROR("Unrecognized system context binary info version.");
  return false;
}

size_t getFileSize(std::string filePath) {
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return 0;
  }
  in.seekg(0, in.end);
  const size_t length = in.tellg();
  in.seekg(0, in.beg);
  return length;
}

bool readBinaryFromFile(std::string filePath, void* buffer, size_t bufferSize) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return false;
  }
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return false;
  }
  if (!in.read(reinterpret_cast<char*>(buffer), bufferSize)) {
    QNN_ERROR("Failed to read the contents of: %s", filePath.c_str());
    return false;
  }
  return true;
}

bool fillDims(std::vector<size_t>& dims, uint32_t* inDimensions, uint32_t rank) {
  if (nullptr == inDimensions) {
    QNN_ERROR("input dimensions is nullptr");
    return false;
  }

  if (rank < 1) {
    QNN_ERROR("invalid rank : %d", rank);
    return false;
  }

  // In case, rank is less than 4, we are pushing 1s
  for (size_t r = 0; r < 4 - rank; r++) {
    dims.push_back(1);
  }

  for (size_t r = 0; r < rank; r++) {
    dims.push_back(inDimensions[r]);
  }

  return true;
}

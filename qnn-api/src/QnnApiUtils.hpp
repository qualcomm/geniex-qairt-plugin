//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "Log.hpp"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"
#include "System/QnnSystemInterface.h"

/**
 * @brief Frees all memory allocated tensor attributes.
 *
 * @param[in] tensorWrapper tensor object to free
 *
 * @return Error code
 */
bool freeQnnTensorWrapper(Qnn_Tensor_t& tensorWrapper);

/**
 * @brief Loops through and frees all memory allocated tensor attributes for each tensorWrapper
 * object.
 *
 * @param[in] tensorWrappers array of tensor objects to free
 *
 * @param[in] numTensors length of the above tensorWrappers array
 *
 * @return Error code
 */
bool freeQnnTensorWrappers(Qnn_Tensor_t*& tensorWrappers, uint32_t numTensors);

/**
 * @brief A helper function to free memory malloced for communicating the Graph for a model(s)
 *
 * @param[in] graphsInfo Pointer pointing to location of graph objects
 *
 * @param[in] numGraphs The number of graph objects the above pointer is pointing to
 *
 * @return Error code
 *
 */
bool freeGraphsInfo(qnn_wrapper_api::GraphInfoPtr_t** graphsInfo, uint32_t numGraphs);

bool freeGraphInfo(qnn_wrapper_api::GraphInfo_t* graphInfo);

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                              qnn_wrapper_api::GraphInfo_t**& graphsInfo,
                              uint32_t& graphsCount);

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                    const uint32_t numGraphs,
                    qnn_wrapper_api::GraphInfo_t**& graphsInfo);

bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                      qnn_wrapper_api::GraphInfo_t* graphInfoDst);

bool copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                     Qnn_Tensor_t*& tensorWrappers,
                     uint32_t tensorsCount);

bool fillDims(std::vector<size_t>& dims, uint32_t* inDimensions, uint32_t rank);

size_t getFileSize(std::string filePath);

bool readBinaryFromFile(std::string filePath,
                        void* buffer,
                        size_t bufferSize);

bool updateMetaDataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                                qnn_wrapper_api::GraphInfo_t** graphsInfo,
                                uint32_t& graphsCount);

bool updateGraphInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                     const uint32_t currCount,
                     qnn_wrapper_api::GraphInfo_t* graphsInfo);

bool updateGraphInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                       qnn_wrapper_api::GraphInfo_t* graphInfoDst);

bool updateTensorInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                      Qnn_Tensor_t* tensorWrappers,
                      uint32_t tensorsCount);

uint32_t getNumGraphInBinary(const QnnSystemContext_BinaryInfo_t* binaryInfo);
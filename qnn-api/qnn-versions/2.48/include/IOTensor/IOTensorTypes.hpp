//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <string>
#include <cstdint>
#include <unordered_map>
#include <vector>

#pragma once
namespace qnn {
namespace tools {
namespace iotensor {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INITIALIZE,
  ERROR_CACHE_NOT_EMPTY,
  ERROR_CACHE_ADD_DUPLICATE,
  ERROR_CACHE_NOT_SUFFICIENT,
  ERROR_CACHE_ENTRY_NOT_FOUND,
  ERROR_ENTRY_BEING_CACHED,
};
enum class OutputDataType { FLOAT_ONLY, NATIVE_ONLY, FLOAT_AND_NATIVE, INVALID };
enum class InputDataType { FLOAT, NATIVE, INVALID };

using TensorBufferInfo =
    std::tuple<std::vector<uint8_t *>, std::vector<uint8_t *>, std::vector<std::vector<uint32_t>>>;

struct TensorMetaData {
  std::string filePath;
  std::string name;
  bool isDynamic;
  std::vector<uint32_t> dimensions;

  TensorMetaData() : filePath(""), name(""), isDynamic(false), dimensions() {}

  TensorMetaData(std::string tensorName, std::string inputFilePathIn)
      : filePath(std::move(inputFilePathIn)),
        name(std::move(tensorName)),
        isDynamic(false),
        dimensions() {}

  bool operator==(const TensorMetaData &other) const {
    if (filePath != other.filePath) {
      return false;
    }
    if (name != other.name) {
      return false;
    }
    if (isDynamic != other.isDynamic) {
      return false;
    }
    if (dimensions != other.dimensions) {
      return false;
    }

    return true;
  }
};

struct TensorSet {
  std::string name;
  std::vector<TensorMetaData> inputTensors;
  std::vector<TensorMetaData> outputTensors;
  bool optionalOutputTensorsPresent{false};
  std::vector<std::string> optionalOutputTensorNames;

  bool operator==(const TensorSet &other) const {
    if (name != other.name) {
      return false;
    }
    if (inputTensors != other.inputTensors) {
      return false;
    }
    if (outputTensors != other.outputTensors) {
      return false;
    }
    if (optionalOutputTensorsPresent != other.optionalOutputTensorsPresent) {
      return true;
    }
    if (optionalOutputTensorNames != other.optionalOutputTensorNames) {
      return false;
    }
    return true;
  }
};

struct GraphInputInfo {
 private:
  bool containsDynamicTensors = false;
  std::vector<std::vector<std::string>> inputTensorsFilesList;
  std::unordered_map<std::string, uint32_t> inputNameToIndex;

  bool equals(const GraphInputInfo &other) const {
    if (name != other.name) {
      return false;
    }
    if (intermediateOutputTensors != other.intermediateOutputTensors) {
      return false;
    }
    if (tensorSets != other.tensorSets) {
      return false;
    }
    return true;
  }

 public:
  bool isInitialized = false;
  std::string name;
  std::vector<std::string> intermediateOutputTensors;
  std::vector<TensorSet> tensorSets;

  bool operator==(const GraphInputInfo &other) const { return equals(other); }

  bool operator!=(const GraphInputInfo &other) const { return !equals(other); }

  bool initialize() {
    isInitialized = true;
    if (tensorSets.size() == 0) {
      return true;
    }
    if (tensorSets[0].inputTensors.size() == 0) {
      return false;
    }
    bool areTensorsNamesPresent = !(tensorSets[0].inputTensors[0].name.empty());
    size_t numInputTensors      = tensorSets[0].inputTensors.size();
    if (areTensorsNamesPresent == true) {
      uint32_t idx = 0;
      for (auto const &inputTensorMetaData : tensorSets[0].inputTensors) {
        inputNameToIndex.emplace(inputTensorMetaData.name, idx);
        idx += 1;
      }
    }
    inputTensorsFilesList.resize(numInputTensors);
    for (auto const &tensorSet : tensorSets) {
      uint32_t tensorIdx = 0;
      for (auto const &inputTensorMetaData : tensorSet.inputTensors) {
        if (areTensorsNamesPresent) {
          tensorIdx = inputNameToIndex[inputTensorMetaData.name];
        }
        inputTensorsFilesList[tensorIdx].emplace_back(inputTensorMetaData.filePath);
        if (inputTensorMetaData.isDynamic) {
          containsDynamicTensors = true;
        }
        if (!areTensorsNamesPresent) {
          tensorIdx += 1;
        }
      }
    }
    return true;
  }

  bool getContainsDynamicTensors() const { return containsDynamicTensors; }

  const std::vector<std::vector<std::string>> &getInputFilesLists() const {
    return inputTensorsFilesList;
  }

  const std::unordered_map<std::string, uint32_t> &getInputNameToIndex() const {
    return inputNameToIndex;
  }

  bool getGraphInputInfo(std::vector<std::vector<std::string>> &inputTensorsFilesListOut,
                         std::unordered_map<std::string, uint32_t> &inputNameToIndexOut) const {
    if (tensorSets.size() == 0) {
      return true;
    }
    size_t numInputTensors = tensorSets[0].inputTensors.size();
    if (numInputTensors == 0) {
      return true;
    }
    bool areTensorsNamesPresent = !(tensorSets[0].inputTensors[0].name.empty());
    if (areTensorsNamesPresent == true) {
      uint32_t idx = 0;
      for (auto const &inputTensorMetaData : tensorSets[0].inputTensors) {
        inputNameToIndexOut.emplace(inputTensorMetaData.name, idx);
        idx += 1;
      }
    }
    inputTensorsFilesListOut.resize(numInputTensors);
    for (auto const &tensorSet : tensorSets) {
      uint32_t tensorIdx = 0;
      for (auto const &inputTensorMetaData : tensorSet.inputTensors) {
        if (areTensorsNamesPresent) {
          tensorIdx = inputNameToIndexOut[inputTensorMetaData.name];
        }
        inputTensorsFilesListOut[tensorIdx].emplace_back(inputTensorMetaData.filePath);
        if (!areTensorsNamesPresent) {
          tensorIdx += 1;
        }
      }
    }
    return true;
  }
};

}  // namespace iotensor
}  // namespace tools
}  // namespace qnn

//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 4068)
#endif

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "QnnApiUtils.hpp"
#include "QnnInterface.h"
#include "utils/detail/utils.hpp"
#include "utils/performance_profile.hpp"

namespace qualla {

namespace QnnUtils {
class DataType {
 private:
  Qnn_DataType_t _dtype{QNN_DATATYPE_UNDEFINED};

 public:
  DataType() = default;
  DataType(const Qnn_Tensor_t* tensor) : _dtype(QNN_TENSOR_GET_DATA_TYPE(tensor)) {}
  DataType(Qnn_DataType_t dtype) : _dtype(dtype){};

  // Enable switch and comparisons
  constexpr operator Qnn_DataType_t() const { return _dtype; }

  int bw();
  int type();

  int32_t val();

  const char* str() const;
};

bool writeRawData(void* tensorData, size_t tensorSize, const std::filesystem::path& path);
bool readRawData(void* tensorData, size_t tensorSize, const std::filesystem::path& path);

struct Dims {
  uint32_t batch = 1;
  uint32_t height, width, channel, bitWidth;
  Dims() : height(0), width(0), channel(0), bitWidth(0) {}
  Dims(uint32_t height, uint32_t width, uint32_t channel, uint32_t bitWidth)
      : height(height), width(width), channel(channel), bitWidth(bitWidth) {}
  Dims(std::vector<size_t>& tDims)
      : height((uint32_t)tDims[1]),
        width((uint32_t)tDims[2]),
        channel((uint32_t)tDims[3]),
        bitWidth((uint32_t)tDims[4]) {
    // Hack to mix batch dimension
    if (tDims[0] != 1 && tDims[1] == 1) height = tDims[0];
    if (tDims[0] > 1 && tDims[1] != 1) batch = tDims[0];
  }
  bool operator==(const Dims& rhs) const {
    return (height == rhs.height) && (width == rhs.width) && (channel == rhs.channel) &&
           (bitWidth == rhs.bitWidth);
  }
  bool operator!=(const Dims& rhs) const { return !(operator==(rhs)); }
  size_t getNumElements() const { return (size_t)(height * width * channel); }
  size_t getSize() const { return (size_t)(batch * height * width * channel * bitWidth); }
  size_t getAlignedSize() const {
    size_t size = getSize();
    if ((size & uint64_t{7}) != uint64_t{0}) {
      size += (uint64_t{8} - (size & uint64_t{7}));
    }
    return size;
  }
  int32_t getMaxDim() const { return std::max({height, width, channel}); };
  Dims T() const { return Dims(width, height, channel, bitWidth); }
};

struct QuantParam {
  double scale;
  int32_t offset;
  QuantParam() {}
  QuantParam(double scale_val, int32_t offset_val) : scale(scale_val), offset(offset_val) {}
};

struct Tensor {
  Qnn_Tensor_t* tensor = nullptr;
  Dims dims;
  std::vector<QuantParam> quantParam;
  DataType dtype;
  Tensor() {}
  Tensor(Qnn_Tensor_t* tensor);
};

// Maps tensor name to QnnUtils::Tensor<Qnn_Tensor_t* tensor, dims, quantparams>
typedef std::map<std::string, Tensor> TensorMap;

static inline uint8_t sat_round(const uint16_t x) {
  const uint16_t rounded   = x + 0x80;              // add 0.5
  const uint16_t corrected = std::max(rounded, x);  // catch unsigned wrap around
  const uint16_t shifted   = corrected >> 8;        // divide by 256
  return static_cast<uint8_t>(shifted);             // to 8-bit
}

static inline void downcast_u16_to_u8(uint8_t* dest, const uint16_t* src, size_t nmemb) {
  for (size_t i = 0; i < nmemb; i++) dest[i] = sat_round(src[i]);
}

template <typename FloatType, typename IntType>
static inline IntType quantize(const FloatType val, int32_t offset, double scale) {
  return static_cast<IntType>(val / scale - offset);
}

template <typename FloatType, typename IntType>
static inline void quantizeTensorPtr(
    FloatType* tensor_float, IntType* tensor_quant, int32_t offset, double scale, size_t nmemb) {
  static const int qmin = std::numeric_limits<IntType>::min();
  static const int qmax = std::numeric_limits<IntType>::max();

  PRAGMA_LOOP_VECTORIZE
  for (size_t i = 0; i < nmemb; i++) {
    double val          = tensor_float[i];
    const int quantized = static_cast<int32_t>(val / scale) - offset;
    const int clamped   = quantized < qmin ? qmin : (quantized > qmax ? qmax : quantized);
    tensor_quant[i]     = static_cast<IntType>(clamped);
  }
}

template <typename FloatType, typename IntType>
static inline void perWidthQuantizeTensorPtr(FloatType* tensor_float,
                                             IntType* tensor_quant,
                                             std::vector<QnnUtils::QuantParam>& quantParam,
                                             int32_t height,
                                             int32_t width,
                                             int32_t channel) {
  for (size_t h = 0; h < height; h++) {
    for (size_t w = 0; w < width; w++) {
      double scale   = quantParam[w].scale;
      int32_t offset = quantParam[w].offset;

      PRAGMA_LOOP_VECTORIZE
      for (size_t c = 0; c < channel; c++) {
        int32_t i       = (h * width * channel) + (w * channel) + c;
        double val      = tensor_float[i];
        tensor_quant[i] = static_cast<IntType>(val / scale - offset);
      }
    }
  }
}

void getQuantParamString(const std::vector<QuantParam>& quantParam,
                         std::string& scale_string,
                         std::string& offset_string);

// String parser that returns a list of upto N numbers from the string
template <size_t N>
std::array<uint16_t, N> parseNumberFromString(const std::string& name) {
  std::array<uint16_t, N> parsed_numbers = {0};  // Fixed-size array with template parameter

  size_t n_found  = 0;  // Count of numbers found
  bool in_number  = false;
  uint16_t number = 0;
  for (const char& ch : name) {
    if (ch >= '0' && ch <= '9') {
      in_number = true;
      number    = number * 10 + (ch - '0');
    } else if (in_number) {
      parsed_numbers[n_found++] = number;
      in_number                 = false;
      number                    = 0;            // Reset number after pushing to the array
      if (n_found >= N) return parsed_numbers;  // Early exit if we've found N numbers
    }
  }

  // Add the last number if the string ends with a number
  if (in_number) parsed_numbers[n_found] = number;
  return parsed_numbers;
}

qnn::tools::netrun::PerfProfile quallaToQnnPerformanceProfile(
    qualla::PerformanceProfile perfProfile);
qualla::PerformanceProfile qnnToQuallaPerformanceProfile(
    qnn::tools::netrun::PerfProfile perfProfile);

}  // namespace QnnUtils
}  // namespace qualla

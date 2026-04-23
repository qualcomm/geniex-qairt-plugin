//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <utils/fmt/format.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "QnnApi.hpp"
#include "qnn-utils.hpp"

namespace fs = std::filesystem;

namespace qualla {
namespace QnnUtils {

Tensor::Tensor(Qnn_Tensor_t* tensor) : tensor(tensor), dtype(QNN_TENSOR_GET_DATA_TYPE(tensor)) {
  std::vector<size_t> tensorDims;
  QnnApi::getTensorShape(tensorDims, *tensor);
  dims = Dims(tensorDims);

  if (!QnnApi::getTensorQuantParams(tensor, quantParam)) quantParam.emplace_back(0, 0);
}

// Alternate implementation for bw() = lambda x: (10 * ((x & 0xf0)>>4) + (x & 0xf)) // 8
int DataType::bw() {
  return (_dtype == QNN_DATATYPE_UNDEFINED) ? -1 : QnnApi::getDataTypeSize(_dtype);
}
int DataType::type() { return (_dtype == QNN_DATATYPE_UNDEFINED) ? -1 : _dtype >> 8; }

int32_t DataType::val() { return static_cast<int32_t>(_dtype); }

bool writeRawData(void* data, size_t size, const fs::path& path) {
  auto p = path.parent_path();
  if (!fs::exists(p) && !fs::create_directories(p)) return false;

  std::ofstream f(path, std::ofstream::binary);
  f.write((char*)data, size);
  f.close();

  return true;
}

bool readRawData(void* data, size_t size, const fs::path& path) {
  if (fs::file_size(path) != size) {
    throw std::runtime_error(fmt::format("file size doesnot match: {} size {}, buf-size {}",
                                         path.string(),
                                         fs::file_size(path),
                                         size));
  }

  std::ifstream f(path, std::ifstream::binary);
  f.read((char*)data, size);
  f.close();

  return true;
}

void getQuantParamString(const std::vector<QuantParam>& quantParam,
                         std::string& scale_string,
                         std::string& offset_string) {
  std::ostringstream scales_s;
  std::ostringstream offsets_s;
  for (int i = 0; i < quantParam.size(); i++) {
    if (i != 0) {
      scales_s << ", ";
      offsets_s << ", ";
    }
    scales_s << std::fixed << std::setprecision(20) << quantParam[i].scale;
    offsets_s << quantParam[i].offset;
  }
  scale_string  = scales_s.str();
  offset_string = offsets_s.str();
}

const char* DataType::str() const {
  switch (_dtype) {
    case QNN_DATATYPE_INT_8:
      return "QNN_DATATYPE_INT_8";
    case QNN_DATATYPE_INT_16:
      return "QNN_DATATYPE_INT_16";
    case QNN_DATATYPE_INT_32:
      return "QNN_DATATYPE_INT_32";
    case QNN_DATATYPE_INT_64:
      return "QNN_DATATYPE_INT_64";
    case QNN_DATATYPE_UINT_8:
      return "QNN_DATATYPE_UINT_8";
    case QNN_DATATYPE_UINT_16:
      return "QNN_DATATYPE_UINT_16";
    case QNN_DATATYPE_UINT_32:
      return "QNN_DATATYPE_UINT_32";
    case QNN_DATATYPE_UINT_64:
      return "QNN_DATATYPE_UINT_64";
    case QNN_DATATYPE_FLOAT_16:
      return "QNN_DATATYPE_FLOAT_16";
    case QNN_DATATYPE_FLOAT_32:
      return "QNN_DATATYPE_FLOAT_32";
    case QNN_DATATYPE_FLOAT_64:
      return "QNN_DATATYPE_FLOAT_64";
    case QNN_DATATYPE_SFIXED_POINT_4:
      return "QNN_DATATYPE_SFIXED_POINT_4";
    case QNN_DATATYPE_SFIXED_POINT_8:
      return "QNN_DATATYPE_SFIXED_POINT_8";
    case QNN_DATATYPE_SFIXED_POINT_16:
      return "QNN_DATATYPE_SFIXED_POINT_16";
    case QNN_DATATYPE_SFIXED_POINT_32:
      return "QNN_DATATYPE_SFIXED_POINT_32";
    case QNN_DATATYPE_UFIXED_POINT_4:
      return "QNN_DATATYPE_UFIXED_POINT_4";
    case QNN_DATATYPE_UFIXED_POINT_8:
      return "QNN_DATATYPE_UFIXED_POINT_8";
    case QNN_DATATYPE_UFIXED_POINT_16:
      return "QNN_DATATYPE_UFIXED_POINT_16";
    case QNN_DATATYPE_UFIXED_POINT_32:
      return "QNN_DATATYPE_UFIXED_POINT_32";
    case QNN_DATATYPE_BOOL_8:
      return "QNN_DATATYPE_BOOL_8";
    case QNN_DATATYPE_STRING:
      return "QNN_DATATYPE_STRING";
    default:
      return "QNN_DATATYPE_UNDEFINED";
  }
}

std::unordered_map<qualla::PerformanceProfile, qnn::tools::netrun::PerfProfile>
    QuallaToQnnPerformanceProfile = {
        {qualla::PerformanceProfile::PERFORMANCE_BURST, qnn::tools::netrun::PerfProfile::BURST},
        {qualla::PerformanceProfile::PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE,
         qnn::tools::netrun::PerfProfile::SUSTAINED_HIGH_PERFORMANCE},
        {qualla::PerformanceProfile::PERFORMANCE_HIGH_PERFORMANCE,
         qnn::tools::netrun::PerfProfile::HIGH_PERFORMANCE},
        {qualla::PerformanceProfile::PERFORMANCE_BALANCED,
         qnn::tools::netrun::PerfProfile::BALANCED},
        {qualla::PerformanceProfile::PERFORMANCE_LOW_BALANCED,
         qnn::tools::netrun::PerfProfile::LOW_BALANCED},
        {qualla::PerformanceProfile::PERFORMANCE_HIGH_POWER_SAVER,
         qnn::tools::netrun::PerfProfile::HIGH_POWER_SAVER},
        {qualla::PerformanceProfile::PERFORMANCE_POWER_SAVER,
         qnn::tools::netrun::PerfProfile::POWER_SAVER},
        {qualla::PerformanceProfile::PERFORMANCE_LOW_POWER_SAVER,
         qnn::tools::netrun::PerfProfile::LOW_POWER_SAVER},
        {qualla::PerformanceProfile::PERFORMANCE_EXTREME_POWER_SAVER,
         qnn::tools::netrun::PerfProfile::EXTREME_POWER_SAVER}};

std::unordered_map<qnn::tools::netrun::PerfProfile, qualla::PerformanceProfile>
    QnnToQuallaPerformanceProfile = {
        {qnn::tools::netrun::PerfProfile::BURST, qualla::PerformanceProfile::PERFORMANCE_BURST},
        {qnn::tools::netrun::PerfProfile::SUSTAINED_HIGH_PERFORMANCE,
         qualla::PerformanceProfile::PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE},
        {qnn::tools::netrun::PerfProfile::HIGH_PERFORMANCE,
         qualla::PerformanceProfile::PERFORMANCE_HIGH_PERFORMANCE},
        {qnn::tools::netrun::PerfProfile::BALANCED,
         qualla::PerformanceProfile::PERFORMANCE_BALANCED},
        {qnn::tools::netrun::PerfProfile::LOW_BALANCED,
         qualla::PerformanceProfile::PERFORMANCE_LOW_BALANCED},
        {qnn::tools::netrun::PerfProfile::HIGH_POWER_SAVER,
         qualla::PerformanceProfile::PERFORMANCE_HIGH_POWER_SAVER},
        {qnn::tools::netrun::PerfProfile::POWER_SAVER,
         qualla::PerformanceProfile::PERFORMANCE_POWER_SAVER},
        {qnn::tools::netrun::PerfProfile::LOW_POWER_SAVER,
         qualla::PerformanceProfile::PERFORMANCE_LOW_POWER_SAVER},
        {qnn::tools::netrun::PerfProfile::EXTREME_POWER_SAVER,
         qualla::PerformanceProfile::PERFORMANCE_EXTREME_POWER_SAVER}};

qnn::tools::netrun::PerfProfile quallaToQnnPerformanceProfile(
    qualla::PerformanceProfile perfProfile) {
  // Translate qualla profile to QNN perf profiles
  if (auto itr = QuallaToQnnPerformanceProfile.find(perfProfile);
      itr != QuallaToQnnPerformanceProfile.end()) {
    return itr->second;
  }
  return qnn::tools::netrun::PerfProfile::BALANCED;
}

qualla::PerformanceProfile qnnToQuallaPerformanceProfile(
    qnn::tools::netrun::PerfProfile perfProfile) {
  // Translate QNN profile to qualla perf profiles
  if (auto itr = QnnToQuallaPerformanceProfile.find(perfProfile);
      itr != QnnToQuallaPerformanceProfile.end()) {
    return itr->second;
  }
  return qualla::PerformanceProfile::PERFORMANCE_BALANCED;
}

}  // namespace QnnUtils
}  // namespace qualla

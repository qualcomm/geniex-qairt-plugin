//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <stdexcept>

#include "Config.hpp"

template <class T>
class ConfigList final {
  using Config = IConfig<T>;

 private:
  std::vector<std::shared_ptr<Config>> m_configs;
  std::vector<const T*> m_rawConfigs;

 public:
  ConfigList() = default;
  /**
   * Creates a shallow copy of an existing ConfigList.
   * Do not copy rawConfigs. This allows copied lists
   * to be modified prior to casting them for QNN API calls.
   */
  ConfigList(const ConfigList& other) {
    for (auto& otherConfig : other.m_configs) {
      m_configs.push_back(otherConfig);
    }
  };

  ConfigList& operator=(const ConfigList& other) {
    for (auto& otherConfig : other.m_configs) {
      m_configs.push_back(otherConfig);
    }
    return *this;
  };

  ConfigList(ConfigList&& other)            = default;
  ConfigList& operator=(ConfigList&& other) = default;

  ~ConfigList() = default;

  inline void add(std::shared_ptr<Config> config) {
    if (!m_rawConfigs.empty()) {
      throw std::runtime_error("Cannot modify ConfigList after it is cast to a C-style array.");
    }

    m_configs.push_back(config);
  }

  inline size_t size() const { return m_configs.size(); }

  /**
   * Moves all entries of the provided ConfigList into this ConfigList.
   */
  void consume(ConfigList<T>& other) {
    if (!other.m_rawConfigs.empty()) {
      throw std::runtime_error("Cannot consume ConfigList after it is cast to a C-style array.");
    }
    for (std::shared_ptr<Config>& config : other.m_configs) {
      add(std::move(config));
    }
    other.m_configs.clear();
  }

  void serialize(qualla::ordered_json& json) const {
    for (const auto& config : m_configs) {
      qualla::ordered_json configJson;
      config->serialize(configJson);
      json.push_back(configJson);
    }
  }

  /**
   * Casts the ConfigList to a null-terminated array. The ConfigList becomes
   * immutable after typecasting to prevent dangling pointers.
   */
  operator const T**() {
    if (m_rawConfigs.empty()) {
      for (auto& config : m_configs) {
        m_rawConfigs.push_back(static_cast<const T*>(*config));
      }
      m_rawConfigs.push_back(nullptr);
    }

    return m_rawConfigs.data();
  }

  friend std::ostream& operator<<(std::ostream& os, const ConfigList<T>& list) {
    qualla::ordered_json configListJson;
    list.serialize(configListJson);
    os << configListJson.dump(2) << std::endl;
    return os;
  }

  /**
   * Creates a non-owning ConfigList from a C-style array of configs.
   *
   * configArray must be null-terminated.
   */
  static ConfigList<T> fromArray(const T** configArray) {
    ConfigList<T> configList;
    size_t idx      = 0;
    const T* config = configArray[0];
    while (config != nullptr) {
      configList.add(std::make_unique<ConfigView<T>>(*config));
      config = configArray[++idx];
    }
    return configList;
  }

  /**
   * Creates a non-owning ConfigList from a C-style array of configs.
   */
  static ConfigList<T> fromArray(T** configArray, size_t configArraySize) {
    ConfigList<T> configList;
    for (int i = 0; i < configArraySize; i++) {
      configList.add(std::make_unique<ConfigView<T>>(*configArray[i]));
    }
    return configList;
  }
};

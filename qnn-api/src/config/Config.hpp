//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "utils/detail/json.hpp"

template <class T>
class IConfig {
 public:
  IConfig()                                = default;
  IConfig(IConfig&& other)                 = default;
  IConfig(const IConfig& other)            = default;
  IConfig& operator=(IConfig&& other)      = default;
  IConfig& operator=(const IConfig& other) = default;
  virtual ~IConfig()                       = default;

  virtual const T& get() const = 0;

  operator const T*() { return &get(); };

  virtual void serialize(qualla::ordered_json& json) const = 0;

  friend std::ostream& operator<<(std::ostream& os, const IConfig<T>& config) {
    qualla::ordered_json json;
    config.serialize(json);
    return os << json.dump(2);
  }
};

/**
 * A QNN config wrapper. Instances of GenericConfig should own
 * all necessary memory referenced by the underlying config
 * data structure.
 */
template <class T>
class GenericConfig : public IConfig<T> {
 protected:
  T m_config;

 public:
  GenericConfig(const T config) : m_config(config){};
  GenericConfig() = default;

  virtual ~GenericConfig() = default;
  virtual const T& get() const override { return m_config; }

  operator T*() { return &m_config; };
};

/**
 * A non-owning view of an underlying QNN config data structure.
 */
template <class T>
class ConfigView : public IConfig<T> {
 protected:
  const T& m_config;

 public:
  ConfigView(const T& config) : m_config(config){};
  virtual ~ConfigView() = default;
  virtual const T& get() const override { return m_config; }
  virtual void serialize(qualla::ordered_json& json) const override { json["type"] = "ConfigView"; }
};

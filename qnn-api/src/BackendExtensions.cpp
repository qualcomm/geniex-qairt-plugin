//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstring>
#include <stdexcept>

#include "BackendExtensions.hpp"
#include "Log.hpp"
#include "dlwrap.hpp"

#ifdef _MSC_VER
#include <excpt.h>
#endif

// IBackend is a C++ interface from the SDK's qnn-net-run sample, and it is passed
// across a DLL boundary: the vtable layout the extensions library was built with
// comes from *its* SDK, while the one we call through comes from the headers
// vendored in qnn-api/include. QNN does not version this interface and offers no
// handshake, so when the two disagree we dispatch to the wrong slot and take an
// access violation.
//
// That was observed with the QAIRT 2.48/2.49 QnnHtpNetRunExtensions.dll against
// these headers: setupLogging() and initialize() happen to land on compatible
// slots, then loadConfig() traps, killing the process before a single log line is
// emitted -- so a plain load failure, and any real error after it, surfaced only
// as 0xC0000005 with no diagnostic. Substituting the 2.45 DLL, or removing it,
// made the same run report its error and exit 1.
//
// A version gate is not usable here: 2.45 works with these headers and supplies
// the HTP JSON settings (perf profile, RPC latency) that the model depends on for
// throughput, so refusing anything newer than the headers would regress a working
// configuration. Instead contain the fault, so a mismatched DLL degrades to
// "run without extensions" -- which the rest of QnnApi already handles, as every
// call site checks interface() for null.
namespace {

// Every call into the extensions DLL happens here. Kept in its own frame because
// the SEH frame below must not contain objects that require unwinding (MSVC
// rejects that with C2712), and IBackend::loadConfig takes its std::string by
// value.
bool runExtensionSetup(qnn::tools::netrun::IBackend* iface,
                       void* backendLibHandle,
                       bool debugQnn,
                       QnnLog_Callback_t logCallback,
                       QnnLog_Level_t logLevel,
                       const std::string* configFilePath,
                       const char** stage) {
  *stage = "setupLogging";
  if (debugQnn && !iface->setupLogging(logCallback, logLevel)) return false;
  *stage = "initialize";
  if (!iface->initialize(backendLibHandle)) return false;
  *stage = "loadConfig";
  if (!iface->loadConfig(*configFilePath)) return false;
  *stage = nullptr;
  return true;
}

// Returns false on a clean rejection by the library; sets *faulted when the call
// trapped instead, which means the vtable did not match and the library must not
// be touched again (not even to destroy it).
bool runExtensionSetupGuarded(qnn::tools::netrun::IBackend* iface,
                              void* backendLibHandle,
                              bool debugQnn,
                              QnnLog_Callback_t logCallback,
                              QnnLog_Level_t logLevel,
                              const std::string* configFilePath,
                              const char** stage,
                              bool* faulted) {
#ifdef _MSC_VER
  __try {
    return runExtensionSetup(
        iface, backendLibHandle, debugQnn, logCallback, logLevel, configFilePath, stage);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    *faulted = true;
    return false;
  }
#else
  // No portable equivalent; a mismatched .so still crashes here.
  return runExtensionSetup(
      iface, backendLibHandle, debugQnn, logCallback, logLevel, configFilePath, stage);
#endif
}

}  // namespace

BackendExtensions::BackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                     void* backendLibHandle,
                                     qnn::tools::netrun::PerfProfile perfProfile,
                                     bool debug_qnn,
                                     QnnLog_Callback_t registeredLogCallback,
                                     QnnLog_Level_t qnnLogLevel)
    : m_backendInterface(nullptr), m_destroyBackendInterfaceFn(nullptr) {
  QNN_DEBUG("DEBUG: backendExtensionsConfig.sharedLibraryPath=%s\n",
            backendExtensionsConfig.sharedLibraryPath.c_str());
  if (backendExtensionsConfig.sharedLibraryPath.empty()) {
    throw std::runtime_error("Empty backend extensions library path.");
  }

  QNN_DEBUG("DEBUG: backendExtensionsConfig.configFilePath=%s\n",
            backendExtensionsConfig.configFilePath.c_str());
  if (backendExtensionsConfig.configFilePath.empty()) {
    throw std::runtime_error("Empty backend extensions config path.");
  }

  void* libHandle =
      dlopen(backendExtensionsConfig.sharedLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (nullptr == libHandle) {
    QNN_ERROR("Unable to load backend extensions lib: [%s]. dlerror(): [%s]",
              backendExtensionsConfig.sharedLibraryPath.c_str(),
              dlerror());
    throw std::runtime_error("Unable to open backend extension library.");
  }

  auto createBackendInterfaceFn = (qnn::tools::netrun::CreateBackendInterfaceFnType_t)dlsym(
      libHandle, "createBackendInterface");
  if (nullptr == createBackendInterfaceFn) {
    throw std::runtime_error("Unable to resolve createBackendInterface.");
  }

  m_destroyBackendInterfaceFn = (qnn::tools::netrun::DestroyBackendInterfaceFnType_t)dlsym(
      libHandle, "destroyBackendInterface");
  if (nullptr == m_destroyBackendInterfaceFn) {
    throw std::runtime_error("Unable to resolve destroyBackendInterface.");
  }

  m_backendInterface = createBackendInterfaceFn();
  if (nullptr == m_backendInterface) {
    throw std::runtime_error("Unable to load backend extensions interface.");
  }

  // The perf profile must be applied AFTER the QNN context/graphs exist; setting it here (before
  // context create) leaves the vote unattached and it is silently dropped, which lets the DSP clock
  // track the calling CPU core and collapses decode on a slow/throttled core. QnnApi applies the
  // profile post-context-create instead. (perfProfile is kept in the signature for API parity.)
  (void)perfProfile;

  const char* stage = nullptr;
  bool faulted      = false;
  if (!runExtensionSetupGuarded(m_backendInterface,
                                backendLibHandle,
                                debug_qnn,
                                registeredLogCallback,
                                qnnLogLevel,
                                &backendExtensionsConfig.configFilePath,
                                &stage,
                                &faulted)) {
    if (faulted) {
      // The vtable did not match. Abandon the library without destroying it --
      // destroyBackendInterface dispatches through the same bad vtable, and the
      // DLL's state is undefined after the trap. Leaking one handle at init is
      // the cheaper outcome. Extensions stay disabled; QnnApi skips every hook
      // because interface() is null.
      // One string literal on purpose: QNN_ERROR stringizes its format argument, so
      // adjacent literals would render as separate quoted chunks.
      QNN_ERROR("Backend extensions library [%s] is not ABI-compatible with this build: it trapped during %s. Continuing WITHOUT backend extensions; the HTP config JSON is not applied, so perf_profile / rpc_control_latency are lost and throughput may be lower. Use the QAIRT version this build was compiled against, or remove the HTP config JSON to silence this.",
                backendExtensionsConfig.sharedLibraryPath.c_str(),
                (nullptr != stage) ? stage : "an unknown stage");
      m_backendInterface          = nullptr;
      m_destroyBackendInterfaceFn = nullptr;
      return;
    }
    // A clean rejection by a compatible library is still a hard configuration
    // error, so keep the original throwing behaviour for it.
    if (nullptr != stage && 0 == strcmp(stage, "setupLogging")) {
      throw std::runtime_error("Unable to initialize logging in backend extensions.");
    }
    if (nullptr != stage && 0 == strcmp(stage, "initialize")) {
      throw std::runtime_error("Unable to initialize backend extensions interface.");
    }
    throw std::runtime_error("Unable to load backend extensions config.");
  }
}
BackendExtensions::~BackendExtensions() {
  if (nullptr != m_destroyBackendInterfaceFn && nullptr != m_backendInterface) {
    m_destroyBackendInterfaceFn(m_backendInterface);
  }
}

qnn::tools::netrun::IBackend* BackendExtensions::interface() { return m_backendInterface; }

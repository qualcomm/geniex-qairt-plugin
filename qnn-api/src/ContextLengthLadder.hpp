//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Back-off policy for context-length graph-variant selection.
//
// A bundle may ship several context-length variants of the same graph
// (`..._cl512_...`, `..._cl4096_...`, ...). HTP reserves persistent per-graph I/O
// for every graph it deserializes, and only one context length is active at a
// time, so on a large multi-shard model the unused variants can overrun the
// protection domain and fail the load of a later shard.
//
// There is no way to ask the device how much room is left: no free-memory or
// PD-capacity query exists, and QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION
// performs its fit check internally without reporting anything back. So the
// selection is empirical -- attempt a set, and let the device's refusal drive the
// next attempt. This header holds that policy, deliberately free of any QnnApi
// dependency so it can be unit-tested without a device.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "QnnCommon.h"
#include "QnnContext.h"

namespace qnn_api {

// Extracts the `_cl<digits>` field from a graph name ("prompt_ar128_cl4096_1_of_4").
// Returns nullopt when the name carries no context-length field, which callers must
// treat as "not ours to classify" rather than as a context length of zero.
//
// Every `_cl` occurrence is examined, not just the first: a name containing an
// unrelated `_cl...` substring (`_cluster_`) would otherwise mask the real field and
// make the graph unclassifiable. That matters more now that back-off is automatic --
// a masked field silently removes a variant from the ladder's view.
inline std::optional<size_t> parseGraphContextLength(const char* graphName) {
  if (nullptr == graphName) return std::nullopt;

  for (const char* p = graphName; nullptr != (p = std::strstr(p, "_cl")); p += 3) {
    const char* d = p + 3;
    if (*d < '0' || *d > '9') continue;  // not the field we are looking for
    size_t cl = 0;
    for (; *d >= '0' && *d <= '9'; d++) {
      cl = cl * 10 + static_cast<size_t>(*d - '0');
    }
    return cl;
  }
  return std::nullopt;
}

// Context-length sets to attempt, in order, when the caller pinned none.
//
// `available` is normalized internally (sorted ascending, deduplicated), so a raw
// scan result can be passed straight in.
//
// Every rung keeps `available.back()`. The largest context length is the model's
// hard capacity -- LLMModel derives its usable context from the largest variant
// actually loaded and throws once a sequence exceeds it -- so dropping it changes
// what the model can do. Dropping smaller variants only costs short-prompt speed,
// because multi-CL promotion starts at the smallest and moves up.
//
// Rungs are coarse: everything, then just the extremes, then just the largest.
// A finer ladder (one middle variant at a time) was measured to settle on a set
// that is both slower and marginally over the I/O budget, surviving only by a
// driver-level retry that does not exist on every SDK. Going straight to the
// extremes is both safer and faster. Keep this comment if that changes: the fine
// variant is a one-line edit here and nowhere else.
//
//   {}                          -> {}
//   {c}                         -> {{c}}
//   {c0, c1}                    -> {{c0,c1}, {c1}}
//   {c0, ..., cN}   (N >= 2)    -> {{c0,...,cN}, {c0,cN}, {cN}}
inline std::vector<std::vector<size_t>> contextLengthLadder(std::vector<size_t> available) {
  std::sort(available.begin(), available.end());
  available.erase(std::unique(available.begin(), available.end()), available.end());

  if (available.empty()) return {};

  const size_t smallest = available.front();
  const size_t largest  = available.back();

  std::vector<std::vector<size_t>> rungs;
  rungs.push_back(available);
  // Drop the middles, keeping both extremes. Skipped when there are none to drop,
  // which also keeps the rung sizes strictly decreasing.
  if (available.size() > 2) rungs.push_back({smallest, largest});
  // Drop the smallest. Skipped when the first rung was already {largest}.
  if (available.size() > 1) rungs.push_back({largest});

  return rungs;
}

// Runs `attempt` over `rungs` in order until one succeeds.
//
// `attempt(rung, index)` returns true on success. Returns the index of the rung
// that loaded, or nullopt if every rung was tried and none did.
//
// Split out from the QNN call so that the ordering, the stopping condition and the
// give-up behaviour are testable without a device -- the state reset that a real
// retry needs is not.
template <class Attempt>
std::optional<size_t> driveContextLengthLadder(const std::vector<std::vector<size_t>>& rungs,
                                               Attempt&& attempt) {
  for (size_t i = 0; i < rungs.size(); i++) {
    if (attempt(rungs[i], i)) return i;
  }
  return std::nullopt;
}

// Whether a contextCreateFromBinary failure could plausibly be fixed by
// deserializing fewer graphs.
//
// Deliberately a deny-list. The error HTP returns when the protection domain
// overruns is not documented, and the first reproduction of the failure recorded
// only the backend's log line ("Current PD has ~N MB in use"), not the code. So
// anything not known to be terminal is treated as worth one more attempt, and the
// structural gates at the call site -- automatic mode only, more than one variant
// present, and a bounded attempt count -- are what bound the cost of guessing
// wrong. Update the deny-list once the real code has been observed on device.
inline bool isRetryableContextCreateError(Qnn_ErrorHandle_t error) {
  switch (error) {
    // The binary does not match this SDK or this SoC, the handle or arguments are
    // wrong, or the config was rejected. None of these depend on how many graphs
    // are being deserialized.
    case QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE:
    case QNN_CONTEXT_ERROR_INVALID_ARGUMENT:
    case QNN_CONTEXT_ERROR_INVALID_HANDLE:
    case QNN_CONTEXT_ERROR_BINARY_VERSION:
    case QNN_CONTEXT_ERROR_BINARY_CONFIGURATION:
    case QNN_CONTEXT_ERROR_INVALID_CONFIG:
      return false;
    default:
      // Includes QNN_CONTEXT_ERROR_CREATE_FROM_BINARY, QNN_CONTEXT_ERROR_MEM_ALLOC,
      // the system-communication errors, and anything undocumented -- the PD
      // overrun is expected to land in one of these.
      return true;
  }
}

}  // namespace qnn_api

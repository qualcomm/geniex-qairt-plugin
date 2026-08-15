// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace geniex {

// Byte-level stop-sequence matcher mirroring llama_cpp's native stop handling,
// so every consumer of the pipeline gets the same behavior.
//
// Tokens arrive as individually decoded text pieces and a stop string can be
// split across token boundaries, so the matcher holds back the longest tail of
// the stream that could still become a match ("pending") and only releases
// bytes through takeReady() once they can no longer start one. When a stop
// completes, the byte offset where it began is recorded via matchOffset() so
// the caller can truncate the accumulated output and cancel generation; on a
// clean end of generation the caller releases any remaining held-back tail
// with flush().
class StopMatcher {
   public:
    StopMatcher() = default;
    explicit StopMatcher(std::vector<std::string> stops);

    // True when at least one non-empty stop sequence is configured.
    bool active() const { return !stops_.empty(); }

    // True once a stop sequence completed.
    bool matched() const { return matched_; }

    // Byte offset in the fed stream where the matched stop sequence begins.
    // Only meaningful when matched() is true.
    std::size_t matchOffset() const { return match_offset_; }

    // Feeds one decoded text piece. Returns true when a stop sequence
    // completed; the caller should then stop generation and truncate its
    // accumulated output at matchOffset().
    bool feed(const std::string& piece);

    // Returns the bytes that are safe to emit — any pending prefix that can no
    // longer start a match (on a match, the pending bytes strictly before it).
    // Returns an empty string when nothing is safe. Call repeatedly until it
    // returns empty.
    std::string takeReady();

    // Returns any remaining held-back bytes when generation ends without a
    // match. Empty once a stop has matched.
    std::string flush();

   private:
    // Offset within `pending_` of the earliest start of any stop sequence.
    std::size_t findMatch() const;

    std::vector<std::string> stops_;
    std::string              pending_;
    std::size_t              hold_         = 0;  // max stop byte length - 1
    std::size_t              emitted_      = 0;  // bytes already released
    std::size_t              match_offset_ = 0;  // stream offset of the match
    bool                     matched_      = false;
};

inline StopMatcher::StopMatcher(std::vector<std::string> stops) {
    for (auto& s : stops) {
        if (s.empty()) {
            continue;
        }
        stops_.push_back(s);
        if (s.size() - 1 > hold_) {
            hold_ = s.size() - 1;
        }
    }
}

inline std::size_t StopMatcher::findMatch() const {
    std::size_t best = std::string::npos;
    for (const auto& s : stops_) {
        const std::size_t pos = pending_.find(s);
        if (pos != std::string::npos && pos < best) {
            best = pos;
        }
    }
    return best;
}

inline bool StopMatcher::feed(const std::string& piece) {
    pending_ += piece;
    if (matched_) {
        return true;
    }
    const std::size_t rel = findMatch();
    if (rel == std::string::npos) {
        return false;
    }
    matched_      = true;
    match_offset_ = emitted_ + rel;
    return true;
}

inline std::string StopMatcher::takeReady() {
    if (pending_.empty()) {
        return {};
    }
    std::size_t safe = 0;
    if (matched_) {
        // Only the pending bytes strictly before the match are valid output.
        safe = match_offset_ - emitted_;
    } else {
        // Bytes further from the end than `hold_` can no longer start a match.
        safe = pending_.size() > hold_ ? pending_.size() - hold_ : 0;
    }
    if (safe == 0) {
        return {};
    }
    std::string out = pending_.substr(0, safe);
    pending_.erase(0, safe);
    emitted_ += safe;
    return out;
}

inline std::string StopMatcher::flush() {
    if (matched_) {
        return {};
    }
    std::string out = pending_;
    pending_.clear();
    emitted_ += out.size();
    return out;
}

}  // namespace geniex

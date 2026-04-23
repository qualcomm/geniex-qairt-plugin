//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "MmappedReader.hpp"

namespace mmapped {

bool MmappedReader::get(uint8_t* dst, uint64_t bytes) {
  if (bytes == 0) return true;
  if (remaining() < bytes) {
    m_failBit = true;
    return false;
  }
  directRead(current<uint8_t>(), bytes, dst);
  return true;
}

bool MmappedReader::seek(int64_t offset) {
  const auto siz = size();
  // Allow negative seeking from the end
  if (offset < 0) {
    offset = siz + offset;
    if (offset < 0) {
      m_offset  = 0;
      m_failBit = true;
      return false;
    }
  }
  // Clamp the seek to the end of the file. Return false if it goes beyond EOF
  if (static_cast<uint64_t>(offset) >= siz) {
    m_offset = siz;
    if (static_cast<uint64_t>(offset) > siz) {
      m_failBit = true;
      return false;
    }
    return true;
  }
  m_offset  = offset;
  m_failBit = false;
  return true;
}

bool MmappedReader::step(int64_t bytes) {
  if (bytes == 0) return true;
  const auto targetOffset = static_cast<int64_t>(m_offset) + bytes;
  if (targetOffset < 0) {
    (void)seek(0);
    m_failBit = true;
    return false;
  }
  return seek(targetOffset);
}

}  // namespace mmapped

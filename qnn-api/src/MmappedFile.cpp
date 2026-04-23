//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "MmappedFile.hpp"

#ifndef _MSC_VER

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#else

#include <fileapi.h>
#include <memoryapi.h>
#include <winnt.h>

#endif  // _MSC_VER

#include <tuple>
#include <type_traits>

namespace mmapped {

#ifndef _MSC_VER
File::~File() { close(); }

File::File(std::string filename, bool readwrite) : m_filename(std::move(filename)) {
  open(readwrite);
}

File::File(File&& other) noexcept
    : m_filename(std::move(other.m_filename)),
      m_fileDescriptor{other.m_fileDescriptor},
      m_address{other.m_address},
      m_size{other.m_size},
      m_readWrite{other.m_readWrite} {
  other.resetMembers();
}

File& File::operator=(File&& other) noexcept {
  if (this != &other) {
    swap(other);
    other.close();
  }
  return *this;
}

const std::string& File::filename() const { return m_filename; }

bool File::isOpen() const noexcept { return m_fileDescriptor != -1 && m_address != nullptr; }

File::operator bool() const noexcept { return isOpen(); }

uint64_t File::size() const noexcept { return m_size; }

bool File::isWritable() const noexcept { return m_readWrite; }

uint8_t* File::data() {
  // NOTE: Will result in COW if written to without the file being opened in RW mode
  return static_cast<uint8_t*>(m_address);
}

const uint8_t* File::data() const noexcept { return static_cast<const uint8_t*>(m_address); }

void File::swap(File& other) {
  std::swap(m_filename, other.m_filename);
  std::swap(m_fileDescriptor, other.m_fileDescriptor);
  std::swap(m_address, other.m_address);
  std::swap(m_size, other.m_size);
  std::swap(m_readWrite, other.m_readWrite);
}

bool File::resize(uint64_t newSize) {
  if (!m_readWrite) {
    if (!close()) {
      return false;
    }

    if (!open(true, newSize)) {
      return false;
    }
    // TODO: ftrunc
    return true;
  } else {
    if (m_size != newSize && newSize != 0) {
      if (ftruncate(m_fileDescriptor, newSize)) {
        // TODO: This might be unrecoverable! Figure out what to do here for error handling
        return false;
      }
    }
    std::tie(m_address, m_size) = remapInternal(newSize);
    if (!m_address) {
      close();
      return false;
    }
    return true;
  }
}

bool File::remap(bool readwrite) {
  if (readwrite == false) {
    close();
    return open(false);  // NOTE: This is probably something that won't be used
  } else {
    // TODO: Could pass in size=0 to avoid overrideSize ever getting hit or just m_size? should be
    // same
    return resize(0);  // Can just use this a shortcut to close and reopen without affecting size
  }
}

bool File::close() {
  bool toret = unmapInternal();
  toret &= closeInternal();
  m_readWrite = false;

  return toret;
}

std::pair<void*, uint64_t> File::getRange(uint64_t offset, uint64_t length) {
  const auto pageSize = getPageSize();

  // Ensure data is page aligned
  if (reinterpret_cast<std::uintptr_t>(data()) % pageSize != 0) {
    return {nullptr, 0};
  }

  const auto start = ((offset + (pageSize - 1)) / pageSize) * pageSize;
  const auto stop  = ((offset + length) / pageSize) * pageSize;

  if (start >= stop) {
    return {nullptr, 0};
  }
  const auto len = stop - start;

  return {static_cast<void*>(data() + start), len};
}

bool File::adviseRange(uint64_t offset, uint64_t length, int advice) {
#ifdef NO_MADVISE_SUPPORT
  (void)std::forward_as_tuple(offset, length, advice);
  return false;
#else
  const auto range = getRange(offset, length);

  if (!range.first) return false;
#ifdef USE_POSIX_MADVISE
  const auto status = posix_madvise(range.first, range.second, advice);
#else
  const auto status = madvise(range.first, range.second, advice);
#endif  // USE_POSIX_MADVISE
  return status == 0;

#endif  // NO_MADVISE_SUPPORT
}

bool File::freeRange(uint64_t offset, uint64_t length) {
  // MADV_PAGEOUT takes longer and drops it from the filesystem cache, so it's not worth it
  // TODO: Make performance comparisons for DONTNEED and COLD
#ifdef MADV_DONTNEED
  return adviseRange(offset, length, MADV_DONTNEED);
#elif defined(MADV_COLD)
  return adviseRange(offset, length, MADV_COLD);
#elif defined(POSIX_MADV_DONTNEED)
  return adviseRange(offset, length, POSIX_MADV_DONTNEED);
#else
  (void)std::forward_as_tuple(offset, length);
  return false;
#endif
}

uint64_t File::getPageSize() {
//  static uint64_t s_pageSize = static_cast<uint64_t>(getpagesize());
  static uint64_t s_pageSize = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
  return s_pageSize;
}

// Open the file, but do not map it
std::pair<int, uint64_t> File::openFileInternal(const char* filename, int openflags, int modeFlags) {
  const auto fd = ::open(filename, openflags, modeFlags);
  if (fd < 0) {
    return {-1, 0};
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    return {-1, 0};
  }
  return {fd, st.st_size};
}

std::pair<void*, uint64_t> File::mapFileInternal(uint64_t size, bool readwrite) {
  void* addr = nullptr;
  if (size > 0) {
    const int prot  = readwrite ? PROT_READ | PROT_WRITE : PROT_READ;
    const int flags = readwrite ? MAP_SHARED : MAP_PRIVATE;
    //    flags |= MAP_HUGE_2MB; // Does not significantly improve performance
    addr = mmap(nullptr, size, prot, flags, m_fileDescriptor, {});
  }

  if (addr == MAP_FAILED || addr == nullptr) {
    return {nullptr, 0};
  }

  return {addr, size};
}

std::pair<void*, uint64_t> File::remapInternal(uint64_t size) {
  if (size == 0) {
    return {nullptr, 0};
  }
#if defined(USE_POSIX_MADVISE) || defined(NO_MADVISE_SUPPORT) // TODO: Should have more specific macro for remap
  return {nullptr, 0}; // does not support remap
#else
  auto addr = mremap(m_address, m_size, size, MREMAP_MAYMOVE);
  if (addr == MAP_FAILED || addr == nullptr) {
    return {nullptr, 0};
  }
  return {addr, size};
#endif
}

bool File::open(bool readwrite, uint64_t overrideSize) {
  if (isOpen()) {
    if (!close()) {
      return false;
    }
  }
  if (!readwrite && overrideSize) {
    return false;
  }

  const int openflags = readwrite ? O_RDWR | O_CREAT : O_RDONLY;
  const int modeFlags = readwrite ? S_IRUSR | S_IWUSR : 0;
  uint64_t openedSize{};
  std::tie(m_fileDescriptor, openedSize) = openFileInternal(m_filename.c_str(), openflags, modeFlags);
  if (m_fileDescriptor == -1) {
    return false;
  }

  if (overrideSize != 0 && openedSize != overrideSize) {
    if (ftruncate(m_fileDescriptor, overrideSize)) {
      return false;
    }
  }

  std::tie(m_address, m_size) =
      mapFileInternal(overrideSize ? overrideSize : openedSize, readwrite);

  m_readWrite = readwrite;
  return true;
}

bool File::unmapInternal() {
  bool toret{true};
  if (m_address != nullptr) {
    if (munmap(m_address, m_size) != 0) {
      toret = false;
    }
    m_address = nullptr;
    m_size    = 0;
  }
  return toret;
}

bool File::closeInternal() {
  bool toret{true};
  if (m_fileDescriptor != -1) {
    if (::close(m_fileDescriptor) != 0) {
      toret = false;
    }
    m_fileDescriptor = -1;
  }
  return toret;
}

void File::resetMembers() {
  m_fileDescriptor = -1;
  m_address        = nullptr;
  m_size           = 0;
  m_readWrite      = 0;
}

#else  // ndef _MSC_VER
File::File(std::string filename, bool readwrite) : m_filename(std::move(filename)) {
  (void)readwrite;
  open();
}

File::~File() { close(); }

bool File::close() {
  bool toret = true;
  if (m_mappingPtr) {
    if (!UnmapViewOfFile(m_mappingPtr)) {
      toret = false;
    }
  }
  if (m_mappingHandle != NULL) {
    if (!CloseHandle(m_mappingHandle)) {
      toret = false;
    }
  }
  if (m_fileHandle != INVALID_HANDLE_VALUE) {
    if (!CloseHandle(m_fileHandle)) {
      toret = false;
    }
  }
  return toret;
}

File::File(File&& other) noexcept
    : m_filename(std::move(other.m_filename)),
      m_fileHandle{other.m_fileHandle},
      m_mappingHandle{other.m_mappingHandle},
      m_mappingPtr{other.m_mappingPtr},
      m_size{other.m_size} {
  other.resetMembers();
}

const std::string& File::filename() const { return m_filename; }

bool File::isOpen() const noexcept {
  return m_fileHandle != INVALID_HANDLE_VALUE && m_mappingHandle != NULL && m_mappingPtr != nullptr;
}

File::operator bool() const noexcept { return isOpen(); }

uint64_t File::size() const noexcept { return m_size; }

bool File::isWritable() const noexcept { return false; }

uint8_t* File::data() {
  // NOTE: Will result in COW if written to without the file being opened in RW mode
  return static_cast<uint8_t*>(m_mappingPtr);
}

const uint8_t* File::data() const noexcept { return static_cast<const uint8_t*>(m_mappingPtr); }

void File::swap(File& other) {
  std::swap(m_filename, other.m_filename);
  std::swap(m_fileHandle, other.m_fileHandle);
  std::swap(m_mappingHandle, other.m_mappingHandle);
  std::swap(m_mappingPtr, other.m_mappingPtr);
  std::swap(m_size, other.m_size);
}

bool File::resize(uint64_t newSize) {
  (void)newSize;
  return false;
}

bool File::remap(bool readwrite) {
  if (readwrite == false) {
    close();
    return open();
  }
  return false;
}

bool File::adviseRange(uint64_t offset, uint64_t length, int advice) {
  (void)std::forward_as_tuple(offset, length, advice);
  return false;
}

bool File::freeRange(uint64_t offset, uint64_t length) { return adviseRange(offset, length, -1); }

bool File::open() {
  auto bail = [this](const char* msg) {
    (void)msg;
    return false;
  };

  m_fileHandle = CreateFileA(
      m_filename.c_str(),
      GENERIC_READ,                                             // dwDesiredAccess
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,   // dwSharedMode with shared delete to mimic Linux behavior
      NULL,                                                     // lpSecurityAttributes
      OPEN_EXISTING,                                            // dwCreatonDisposition
      FILE_ATTRIBUTE_READONLY,                                  // Or NORMAL or some of the other hinting options
                                                                // e.g.FILE_FLAG_SEQUENTIAL_SCAN, FILE_FLAG_RANDOM_ACCESS
      NULL);
  if (m_fileHandle == INVALID_HANDLE_VALUE) {
    return bail("Invalid file handle");
  }

  LARGE_INTEGER fileSize{};
  if (!GetFileSizeEx(m_fileHandle, &fileSize)) {
    return bail("Failed to get file size");
  }
  m_size = fileSize.QuadPart;

  m_mappingHandle =
      CreateFileMappingA(m_fileHandle,
                         NULL,           // lpSecurityAttributes
                         PAGE_READONLY,  // flProtect
                         0,              // dwMaximumSizeHigh
                         0,              // dwMaximumSizeLow
                         NULL);          // lpName, but I shouldn't need to name this file mapping
  if (m_mappingHandle == NULL) {
    return bail("Failed to get file mapping");
  }

  m_mappingPtr = MapViewOfFile(m_mappingHandle,  // hFileMappingObject
                               FILE_MAP_READ,    // dwDesiredAccess // TODO: FILE_MAP_LARGE_PAGES?
                               0,                // dwFileOffsetHigh
                               0,                // dwFileOffsetlow
                               0);  // dwNumberOfBytesToMap // Could also use fileSize, but this
                                    // seems like it should work as well
  if (m_mappingPtr == NULL) {
    return bail("Failed to map view of file");
  }

  return true;
}

void File::resetMembers() {
  m_fileHandle    = INVALID_HANDLE_VALUE;
  m_mappingHandle = NULL;
  m_mappingPtr    = NULL;
  m_size          = 0;
}

#endif  // ndef _MSC_VER

}  // namespace mmapped

//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "DmaBufAllocator.hpp"
#include "QnnMem.h"
#include "QnnTypeMacros.hpp"

DmaBufferAllocator::DmaBufferAllocator(Qnn_ContextHandle_t contextHandle,
                                       QNN_INTERFACE_VER_TYPE* qnnInterface)
    : m_libDmaBufHeapHandle(nullptr),
      m_dmaBufCreate(nullptr),
      m_dmaBufAlloc(nullptr),
      m_dmaBufDeinit(nullptr),
      m_libIonMemHandle(nullptr),
      m_ionOpenFn(nullptr),
      m_ionAllocFn(nullptr),
      m_ionCloseFn(nullptr),
      m_qnnInterface(qnnInterface),
      m_contextHandle(contextHandle) {}

bool DmaBufferAllocator::initialize() {
  const char* dmaHeapLocation = "/dev/dma_heap";
  const char* ionLocation     = "/dev/ion";
  if (std::filesystem::exists(dmaHeapLocation)) {
    QNN_DEBUG("Using DmaBuf Allocator");
    m_useIonMemHandle = false;
  } else if (std::filesystem::exists(ionLocation)) {
    QNN_DEBUG("Using ION Allocator");
    m_useIonMemHandle = true;
  } else {
    QNN_ERROR("Zero Copy Memory Not Supported");
    return false;
  }

  if (m_useIonMemHandle) {
    // On Android, 32-bit and 64-bit libion.so can be found at /system/lib and /system/lib64
    //  respectively.
    const std::string defaultLibPaths[] = {"libion.so"};
    for (const auto& path : defaultLibPaths) {
      m_libIonMemHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (m_libIonMemHandle != nullptr) {
        break;
      }
    }
    if (nullptr == m_libIonMemHandle) {
      QNN_ERROR("Unable to load memory allocator. dlerror(): %s", dlerror());
      return false;
    }
    m_ionOpenFn = (IonOpenFn_t)dlsym(m_libIonMemHandle, "ion_open");
    if (nullptr == m_ionOpenFn) {
      QNN_ERROR("Unable to access symbols in libion. dlerror(): %s", dlerror());
      return false;
    }
    m_ionAllocFn = (IonAllocFd_t)dlsym(m_libIonMemHandle, "ion_alloc_fd");
    if (nullptr == m_ionAllocFn) {
      QNN_ERROR("Unable to access symbols in libion. dlerror(): %s", dlerror());
      return false;
    }
    m_ionCloseFn = (IonCloseFn_t)dlsym(m_libIonMemHandle, "ion_close");
    if (nullptr == m_ionCloseFn) {
      QNN_ERROR("Unable to access symbols in libion. dlerror(): %s", dlerror());
      return false;
    }
    return true;
  }

  // On Android, 32-bit and 64-bit libdmaBufheap.so can be found at /system/lib and /system/lib64
  //  respectively.
  const std::string defaultLibPaths[] = {"libdmabufheap.so", "libdmabufheap.so.0"};
  for (const auto& path : defaultLibPaths) {
    m_libDmaBufHeapHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (m_libDmaBufHeapHandle != nullptr) {
      break;
    }
  }
  if (nullptr == m_libDmaBufHeapHandle) {
    QNN_ERROR("Unable to load memory allocator. dlerror(): %s", dlerror());
    return false;
  }
  m_dmaBufCreate =
      (DmaBufCreateFn_t)dlsym(m_libDmaBufHeapHandle, "CreateDmabufHeapBufferAllocator");
  m_dmaBufAlloc  = (DmaBufAllocFn_t)dlsym(m_libDmaBufHeapHandle, "DmabufHeapAlloc");
  m_dmaBufDeinit = (DmaBufDeinitFn_t)dlsym(m_libDmaBufHeapHandle, "FreeDmabufHeapBufferAllocator");
  if (nullptr == m_dmaBufCreate || nullptr == m_dmaBufAlloc || nullptr == m_dmaBufDeinit) {
    QNN_ERROR("Unable to access symbols in libdmaBufheap. dlerror(): %s", dlerror());
    return false;
  }
  return true;
}

DmaBufferAllocator::~DmaBufferAllocator() {
  if (m_libDmaBufHeapHandle) {
    dlclose(m_libDmaBufHeapHandle);
    m_libDmaBufHeapHandle = nullptr;
  }
  if (m_libIonMemHandle) {
    dlclose(m_libIonMemHandle);
    m_libIonMemHandle = nullptr;
  }
}

DmaBufferData* DmaBufferAllocator::getDmaBufTensorData(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) return nullptr;
  return &m_tensorToDmaBufferData.at(tensor);
}

void* DmaBufferAllocator::getBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_WARN("DmaBufferAllocator: getBuffer: received a null pointer to a tensor");
    return nullptr;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    QNN_ERROR("DmaBufferAllocator: Tensor not found with address = %p", tensor);
    return nullptr;
  }
  DmaBufferData dmaBufferData = m_tensorToDmaBufferData[tensor];
  return dmaBufferData.memPointer;
}

int DmaBufferAllocator::getFd(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("DmaBufferAllocator: getFd : Couldn't find tensor %p", tensor);
    return -1;
  }
  return data->fd;
}

size_t DmaBufferAllocator::getOffset(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("DmaBufferAllocator: getOffset : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->offset;
}

size_t DmaBufferAllocator::getBufferSize(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("DmaBufferAllocator: getBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->totalBufferSize;
};

size_t DmaBufferAllocator::getTotalBufferSize(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    QNN_ERROR("DmaBufferAllocator: getTotalBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->totalBufferSize;
}

bool DmaBufferAllocator::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (!tensor) {
    QNN_ERROR("DmaBufferAllocator: Received nullptr for tensor");
    return false;
  }

  if (m_tensorToDmaBufferData.find(tensor) != m_tensorToDmaBufferData.end()) {
    QNN_ERROR("DmaBufferAllocator: Tensor already allocated");
    return false;
  }
  int fd                   = -1;
  int ionAllocatorFd       = -1;
  void* dmaBufferAllocator = nullptr;
  if (m_useIonMemHandle) {
    if (m_libIonMemHandle == nullptr) {
      QNN_ERROR("DmaBufferAllocator not initialized");
      return false;
    }

    ionAllocatorFd = m_ionOpenFn();
    if (ionAllocatorFd == -1) {
      QNN_ERROR("DmaBufferAllocator: nullptr returned for ion_open().");
      return false;
    }

    int errorCode = m_ionAllocFn(ionAllocatorFd,
                                 tensorDataSize,
                                 ION_CL_DEVICE_PAGE_SIZE_QCOM,
                                 ION_HEAP(ION_SYSTEM_HEAP_ID),
                                 0,
                                 &fd);
    if (errorCode < 0) {
      QNN_ERROR("DmaBufAlloc returned a invalid file descriptor = %d", fd);
      return false;
    }
  } else {
    if (m_libDmaBufHeapHandle == nullptr) {
      QNN_ERROR("DmaBufferAllocator not initialized");
      return false;
    }

    dmaBufferAllocator = m_dmaBufCreate();
    if (dmaBufferAllocator == nullptr) {
      QNN_ERROR("DmaBufferAllocator: nullptr returned for CreateDmabufHeapBufferAllocator().");
      return false;
    }

    fd = m_dmaBufAlloc(dmaBufferAllocator, "qcom,system", tensorDataSize, 0, 0);
    if (fd < 0) {
      QNN_ERROR("DmaBufAlloc returned a invalid file descriptor = %d", fd);
      return false;
    }
  }

  void* memPointer = mmap(nullptr, tensorDataSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (MAP_FAILED == memPointer) {
    QNN_ERROR("DmaBufferAllocator: Unable to open file returned by DmaBufAlloc with mmap");
    return false;
  }

  Qnn_MemDescriptor_t memDescriptor = {
      {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
      QNN_TENSOR_GET_DATA_TYPE(tensor),
      QNN_MEM_TYPE_DMA_BUF,
      {.dmaBufInfo = {fd, memPointer}}};
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);
  Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);

  if (QNN_SUCCESS !=
      m_qnnInterface->memRegister(m_contextHandle, &memDescriptor, 1, &(memHandle))) {
    QNN_ERROR("DmaBufferAllocator: Failure to register ion memory with the backend");
    return false;
  }
  QNN_DEBUG(
      "DmaBufferAllocator: Memregister successful with handle %p for DMA buffer with size: %zu and "
      "fd %d",
      memHandle,
      tensorDataSize,
      fd);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
  m_tensorToDmaBufferData.insert(
      {tensor, DmaBufferData(dmaBufferAllocator, ionAllocatorFd, fd, memPointer, tensorDataSize)});

  return true;
}

bool DmaBufferAllocator::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("DmaBufferAllocator: Received nullptr for tensor");
    return false;
  }
  if (m_sharedMemoryTensors.find(tensor) == m_sharedMemoryTensors.end()) {
    auto memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
    if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
      QNN_ERROR("DmaBufferAllocator: Failed to deregister custom memory handle with the backend");
      return false;
    }
    if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
      QNN_ERROR("DmaBufferAllocator: Tensor not found with address = %p", tensor);
      return false;
    }
    DmaBufferData dmaBufferData = m_tensorToDmaBufferData[tensor];
    if (!dmaBufferData.memPointer) {
      QNN_ERROR("DmaBufAllocator: Nullptr recieved for memory with fd :%d", dmaBufferData.fd);
    }
    if (munmap(dmaBufferData.memPointer, dmaBufferData.totalBufferSize)) {
      QNN_ERROR("DmaBufferAllocator: Unmap failed for memory with fd %d", dmaBufferData.fd);
      return false;
    }
    if (m_useIonMemHandle) {
      close(dmaBufferData.fd);
      m_ionCloseFn(dmaBufferData.ionAllocatorFd);
    } else {
      if (!m_dmaBufDeinit) {
        QNN_ERROR("DmaBufferAllocator: DmaBuf Deinit function pointer is null");
        return false;
      }
      m_dmaBufDeinit(dmaBufferData.dmaBufferAllocator);
    }
  }
  m_tensorToDmaBufferData.erase(tensor);
  return true;
}

void DmaBufferAllocator::freeFusedBuffers() {
  std::vector<Qnn_Tensor_t*> allocatedtensorToDmaBufferData;
  for (const auto& it : m_tensorToDmaBufferData) {
    if (m_sharedMemoryTensors.find(it.first) == m_sharedMemoryTensors.end()) {
      allocatedtensorToDmaBufferData.push_back(it.first);
    }
  }

  for (const auto& tensor : allocatedtensorToDmaBufferData) {
    if (!freeTensorBuffer(tensor)) {
      QNN_ERROR("DmaBufAllocator : Deallocation failed");
    }
  }
}

bool DmaBufferAllocator::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("DmaBufferAllocator: Received nullptr");
    return false;
  }
  if (m_tensorToDmaBufferData.find(src) == m_tensorToDmaBufferData.end()) {
    QNN_ERROR("DmaBufferAllocator: Src Tensor not found");
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToDmaBufferData.insert({dest, m_tensorToDmaBufferData[src]});
  m_sharedMemoryTensors.insert(dest);
  return true;
}

bool DmaBufferAllocator::beforeWriteToBuffer(Qnn_Tensor_t* tensor) {
  if (m_useIonMemHandle) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("beforeWriteToBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    QNN_ERROR("beforeWriteToBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "beforeWriteToBuffer: Error preparing the cache for buffer writes."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool DmaBufferAllocator::afterWriteToBuffer(Qnn_Tensor_t* tensor) {
  if (m_useIonMemHandle) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("afterWriteToBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    QNN_ERROR("afterWriteToBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "afterWriteToBuffer: Error close the cache after buffer writing."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool DmaBufferAllocator::beforeReadFromBuffer(Qnn_Tensor_t* tensor) {
  if (m_useIonMemHandle) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("beforeReadFromBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    QNN_ERROR("beforeReadFromBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "beforeReadFromBuffer: Error preparing the cache for buffer reading."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool DmaBufferAllocator::afterReadFromBuffer(Qnn_Tensor_t* tensor) {
  if (m_useIonMemHandle) {
    return true;
  }
  if (!tensor) {
    QNN_WARN("afterReadFromBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    QNN_ERROR("afterReadFromBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    QNN_ERROR(
        "afterReadFromBuffer: Error closing the cache after buffer reading."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

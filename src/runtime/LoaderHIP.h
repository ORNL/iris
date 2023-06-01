#ifndef IRIS_SRC_RT_LOADER_HIP_H
#define IRIS_SRC_RT_LOADER_HIP_H

#include "Loader.h"
#include <iris/hip/hip_runtime.h>
#include <pthread.h>

namespace iris {
namespace rt {

class LoaderHIP : public Loader {
public:
  LoaderHIP();
  ~LoaderHIP();

  const char* library() { return "libamdhip64.so"; }
//  const char* library() { return "libhip_hcc.so"; }
  int LoadFunctions();

  hipError_t (*hipInit)(unsigned int flags);
  hipError_t (*hipDriverGetVersion)(int* driverVersion);
  hipError_t (*hipSetDevice)(int deviceId);
  hipError_t (*hipGetDevice)(int* deviceId);
  hipError_t (*hipGetDeviceCount)(int* count);
  hipError_t (*hipDeviceGetAttribute)(int* pi, hipDeviceAttribute_t attr, int deviceId);
  hipError_t (*hipDeviceGet)(hipDevice_t* device, int ordinal);
  hipError_t (*hipDeviceGetName)(char* name, int len, hipDevice_t device);
  hipError_t (*hipCtxCreate)(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
  hipError_t (*hipCtxGetCurrent)(hipCtx_t* ctx);
  hipError_t (*hipCtxSetCurrent)(hipCtx_t ctx);
  hipError_t (*hipCtxSynchronize)(void);
  hipError_t (*hipModuleLoad)(hipModule_t* module, const char* fname);
  hipError_t (*hipModuleGetFunction)(hipFunction_t* function, hipModule_t module, const char* kname);
  hipError_t (*hipMalloc)(void** ptr, size_t size);
  hipError_t (*hipHostRegister)(void *ptr, size_t size, unsigned int flags);
  hipError_t (*hipMemset)(hipDeviceptr_t ptr, int init, size_t size);
  hipError_t (*hipDeviceEnablePeerAccess)(int peerDevice, unsigned int flags);
  hipError_t (*hipDeviceCanAccessPeer)(int *canaccess, int device, int peerDevice);
  hipError_t (*hipFree)(void* ptr);
  hipError_t (*hipGetDeviceProperties)(hipDeviceProp_t* prop, int deviceId);
  hipError_t (*hipMemcpy2D)(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
  hipError_t (*hipMemcpyDtoD)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
  hipError_t (*hipMemcpyHtoD)(hipDeviceptr_t dst, void* src, size_t sizeBytes);
  hipError_t (*hipMemcpyDtoH)(void* dst, hipDeviceptr_t src, size_t sizeBytes);
  hipError_t (*hipModuleLaunchKernel)(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra);
  hipError_t (*hipDeviceSynchronize)(void);

  void Lock();
  void Unlock();

private:
  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HIP_H */


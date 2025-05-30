#include "LoaderHIP.h"
#include "Debug.h"

namespace iris {
namespace rt {

LoaderHIP::LoaderHIP() {
  pthread_mutex_init(&mutex_, NULL);
}

void *LoaderHIP::GetSymbol(const char *name) {
  void *dptr = NULL;
  hipGetSymbolAddress(&dptr, name);
  return (void *)dptr;
}

LoaderHIP::~LoaderHIP() {
  pthread_mutex_destroy(&mutex_);
}

int LoaderHIP::LoadFunctions() {
  LOADFUNC(hipInit);
  LOADFUNC(hipDriverGetVersion);
  LOADFUNC(hipSetDevice);
  LOADFUNC(hipGetDevice);
  LOADFUNC(hipGetDeviceCount);
  LOADFUNC(hipPointerGetAttribute);
  LOADFUNC(hipDeviceGetAttribute);
  LOADFUNC(hipDeviceGet);
  LOADFUNC(hipDeviceGetName);
  LOADFUNC(hipCtxCreate);
  LOADFUNC(hipCtxDestroy);
  LOADFUNC(hipDeviceReset);
  LOADFUNC(hipCtxGetCurrent);
  LOADFUNC(hipCtxSetCurrent);
  LOADFUNC(hipCtxSynchronize);
  LOADFUNC(hipModuleLoad);
  LOADFUNC(hipModuleGetFunction);
  LOADFUNC(hipMallocAsync);
  LOADFUNC(hipMalloc);
  LOADFUNC(hipHostRegister);
  LOADFUNC(hipHostUnregister);
  LOADFUNC(hipMemset);
  LOADFUNC(hipMemsetAsync);
  LOADFUNC(hipFree);
  LOADFUNC(hipFreeAsync);
  LOADFUNC(hipGetDeviceProperties);
  LOADFUNC(hipDeviceCanAccessPeer);
  LOADFUNC(hipCtxEnablePeerAccess);
  LOADFUNC(hipDeviceEnablePeerAccess);
  LOADFUNC(hipStreamCreate);
  LOADFUNC(hipStreamDestroy);
  LOADFUNC(hipMemcpy2D);
  LOADFUNC(hipHostGetDevicePointer);
  LOADFUNC(hipSetDeviceFlags);
  LOADFUNC(hipMemcpy2DAsync);
  LOADFUNC(hipMemcpyDtoD);
  LOADFUNC(hipMemcpyDtoDAsync);
  LOADFUNC(hipMemcpyHtoD);
  LOADFUNC(hipMemcpyHtoDAsync);
  LOADFUNC(hipMemcpyDtoH);
  LOADFUNC(hipMemcpyDtoHAsync);
  LOADFUNC(hipModuleLaunchKernel);
  LOADFUNC(hipDeviceSynchronize);
  LOADFUNC(hipStreamWaitEvent);
  LOADFUNC(hipEventCreateWithFlags);
  LOADFUNC(hipEventCreate);
  LOADFUNC(hipEventRecord);
  LOADFUNC(hipEventDestroy);
  LOADFUNC(hipEventSynchronize);
  LOADFUNC(hipEventElapsedTime);
  LOADFUNC(hipEventQuery);
  LOADFUNC(hipStreamAddCallback);
 
  return IRIS_SUCCESS;
}

void LoaderHIP::Lock() {
  pthread_mutex_lock(&mutex_);
}

void LoaderHIP::Unlock() {
  pthread_mutex_unlock(&mutex_);
}

} /* namespace rt */
} /* namespace iris */


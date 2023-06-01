#include "LoaderHIP.h"
#include "Debug.h"

namespace iris {
namespace rt {

LoaderHIP::LoaderHIP() {
  pthread_mutex_init(&mutex_, NULL);
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
  LOADFUNC(hipDeviceGetAttribute);
  LOADFUNC(hipDeviceGet);
  LOADFUNC(hipDeviceGetName);
  LOADFUNC(hipCtxCreate);
  LOADFUNC(hipCtxGetCurrent);
  LOADFUNC(hipCtxSetCurrent);
  LOADFUNC(hipCtxSynchronize);
  LOADFUNC(hipModuleLoad);
  LOADFUNC(hipModuleGetFunction);
  LOADFUNC(hipMalloc);
  LOADFUNC(hipHostRegister);
  LOADFUNC(hipMemset);
  LOADFUNC(hipFree);
  LOADFUNC(hipGetDeviceProperties);
  LOADFUNC(hipDeviceCanAccessPeer);
  LOADFUNC(hipDeviceEnablePeerAccess);
  LOADFUNC(hipMemcpy2D);
  LOADFUNC(hipMemcpyDtoD);
  LOADFUNC(hipMemcpyHtoD);
  LOADFUNC(hipMemcpyDtoH);
  LOADFUNC(hipModuleLaunchKernel);
  LOADFUNC(hipDeviceSynchronize);
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


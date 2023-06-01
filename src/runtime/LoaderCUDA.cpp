#include "LoaderCUDA.h"
#include "Debug.h"

namespace iris {
namespace rt {

LoaderCUDA::LoaderCUDA() {
}

LoaderCUDA::~LoaderCUDA() {
}

int LoaderCUDA::LoadFunctions() {
  LOADFUNC(cuInit);
  LOADFUNC(cuDriverGetVersion);
  LOADFUNC(cuDeviceGet);
  LOADFUNC(cuDeviceGetAttribute);
  LOADFUNC(cuDeviceGetCount);
  LOADFUNC(cuDeviceGetName);
  LOADFUNC(cuCtxGetCurrent);
  LOADFUNC(cuCtxSetCurrent);
  LOADFUNC(cuCtxEnablePeerAccess);
  LOADFUNCEXT(cudaSetDevice);
  LOADFUNCEXT(cudaDeviceCanAccessPeer);
  LOADFUNCEXT(cudaDeviceEnablePeerAccess);
  LOADFUNCSYM(cuCtxCreate, cuCtxCreate_v2);
  LOADFUNC(cuCtxSynchronize);
  LOADFUNC(cuStreamAddCallback);
  LOADFUNC(cuStreamCreate);
  LOADFUNC(cuStreamSynchronize);
  LOADFUNC(cuModuleGetFunction);
  LOADFUNC(cuModuleLoad);
  LOADFUNC(cuModuleGetTexRef);
  LOADFUNCSYM(cuTexRefSetAddress, cuTexRefSetAddress_v2);
  LOADFUNC(cuTexRefSetAddressMode);
  LOADFUNC(cuTexRefSetFilterMode);
  LOADFUNC(cuTexRefSetFlags);
  LOADFUNC(cuTexRefSetFormat);
  LOADFUNC(cuMemcpy2D);
  //LOADFUNC(cuMemset);
  LOADFUNC(cuMemcpyDtoD);
  LOADFUNCEXT(cudaMalloc);
  LOADFUNCEXT(cudaMemcpy);
  LOADFUNCEXT(cudaMemcpy2D);
  LOADFUNCEXT(cudaMemset);
  LOADFUNCEXT(cudaHostRegister);
  LOADFUNC(cuMemcpy2DUnaligned);
  LOADFUNC(cuMemcpy2DAsync);
  LOADFUNCSYM(cuMemAlloc, cuMemAlloc_v2);
  LOADFUNCSYM(cuMemFree, cuMemFree_v2);
  LOADFUNCSYM(cuMemcpyHtoD, cuMemcpyHtoD_v2);
  LOADFUNCSYM(cuMemcpyHtoDAsync, cuMemcpyHtoDAsync_v2);
  LOADFUNCSYM(cuMemcpyDtoH, cuMemcpyDtoH_v2);
  LOADFUNCSYM(cuMemcpyDtoHAsync, cuMemcpyDtoHAsync);
  LOADFUNC(cuLaunchKernel);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


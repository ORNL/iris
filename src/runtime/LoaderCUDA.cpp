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
  LOADFUNCSYM(cuMemAlloc, cuMemAlloc_v2);
  LOADFUNCSYM(cuMemFree, cuMemFree_v2);
  LOADFUNCSYM(cuMemcpyHtoD, cuMemcpyHtoD_v2);
  LOADFUNCSYM(cuMemcpyHtoDAsync, cuMemcpyHtoDAsync_v2);
  LOADFUNCSYM(cuMemcpyDtoH, cuMemcpyDtoH_v2);
  LOADFUNCSYM(cuMemcpyDtoHAsync, cuMemcpyDtoHAsync);
  LOADFUNC(cuLaunchKernel);
  return IRIS_OK;
}

} /* namespace rt */
} /* namespace iris */


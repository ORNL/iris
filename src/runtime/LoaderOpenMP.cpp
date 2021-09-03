#include "LoaderOpenMP.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

LoaderOpenMP::LoaderOpenMP() {
}

LoaderOpenMP::~LoaderOpenMP() {
}

const char* LoaderOpenMP::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("KERNEL_OPENMP", &path, NULL);
  return path;
}

int LoaderOpenMP::LoadFunctions() {
  LOADFUNC(brisbane_openmp_init);
  LOADFUNC(brisbane_openmp_finalize);
  LOADFUNC(brisbane_openmp_kernel);
  LOADFUNC(brisbane_openmp_setarg);
  LOADFUNC(brisbane_openmp_setmem);
  LOADFUNC(brisbane_openmp_launch);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


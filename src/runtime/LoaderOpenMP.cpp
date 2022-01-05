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
  Platform::GetPlatform()->EnvironmentGet("KERNEL_BIN_OPENMP", &path, NULL);
  return path;
}

int LoaderOpenMP::LoadFunctions() {
  /*
  LOADFUNC(brisbane_openmp_init);
  LOADFUNC(brisbane_openmp_finalize);
  LOADFUNC(brisbane_openmp_kernel);
  LOADFUNC(brisbane_openmp_setarg);
  LOADFUNC(brisbane_openmp_setmem);
  LOADFUNC(brisbane_openmp_launch);
  */
  LOADFUNCSYM(brisbane_openmp_init,     iris_openmp_init);
  LOADFUNCSYM(brisbane_openmp_finalize, iris_openmp_finalize);
  LOADFUNCSYM(brisbane_openmp_kernel,   iris_openmp_kernel);
  LOADFUNCSYM(brisbane_openmp_setarg,   iris_openmp_setarg);
  LOADFUNCSYM(brisbane_openmp_setmem,   iris_openmp_setmem);
  LOADFUNCSYM(brisbane_openmp_launch,   iris_openmp_launch);

  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


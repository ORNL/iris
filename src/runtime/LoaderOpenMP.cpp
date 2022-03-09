#include "LoaderOpenMP.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
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
  LOADFUNC(iris_openmp_init);
  LOADFUNC(iris_openmp_finalize);
  LOADFUNC(iris_openmp_kernel);
  LOADFUNC(iris_openmp_setarg);
  LOADFUNC(iris_openmp_setmem);
  LOADFUNC(iris_openmp_launch);
  */
  LOADFUNCSYM(iris_openmp_init,     iris_openmp_init);
  LOADFUNCSYM(iris_openmp_finalize, iris_openmp_finalize);
  LOADFUNCSYM(iris_openmp_kernel,   iris_openmp_kernel);
  LOADFUNCSYM(iris_openmp_setarg,   iris_openmp_setarg);
  LOADFUNCSYM(iris_openmp_setmem,   iris_openmp_setmem);
  LOADFUNCSYM(iris_openmp_launch,   iris_openmp_launch);

  return IRIS_OK;
}

} /* namespace rt */
} /* namespace iris */


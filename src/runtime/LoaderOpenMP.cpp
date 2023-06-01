#include "LoaderOpenMP.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderOpenMP::LoaderOpenMP() : Loader() {
    iris_openmp_init = NULL;
    iris_openmp_init_handles = NULL;
    iris_openmp_finalize_handles = NULL;
    iris_openmp_finalize = NULL;
    iris_openmp_kernel = NULL;
    iris_openmp_setarg = NULL;
    iris_openmp_setmem = NULL;
    iris_openmp_launch = NULL;
    iris_openmp_kernel_with_obj = NULL;
    iris_openmp_setarg_with_obj = NULL;
    iris_openmp_setmem_with_obj = NULL;
    iris_openmp_launch_with_obj = NULL;
}

LoaderOpenMP::~LoaderOpenMP() {
}

const char* LoaderOpenMP::library() {
  char* path = NULL;
  Platform::GetPlatform()->GetFilePath("KERNEL_BIN_OPENMP", &path, NULL);
  return path;
}

int LoaderOpenMP::LoadFunctions() {
  Loader::LoadFunctions();
  LOADFUNCSYM(iris_openmp_init,     iris_openmp_init);
  LOADFUNCSYM(iris_openmp_finalize, iris_openmp_finalize);
  LOADFUNCSYM_OPTIONAL(iris_openmp_init_handles,   iris_openmp_init_handles);
  LOADFUNCSYM_OPTIONAL(iris_openmp_finalize_handles,   iris_openmp_finalize_handles);
  LOADFUNCSYM_OPTIONAL(iris_openmp_kernel,   iris_openmp_kernel);
  LOADFUNCSYM_OPTIONAL(iris_openmp_setarg,   iris_openmp_setarg);
  LOADFUNCSYM_OPTIONAL(iris_openmp_setmem,   iris_openmp_setmem);
  LOADFUNCSYM_OPTIONAL(iris_openmp_launch,   iris_openmp_launch);
  LOADFUNCSYM_OPTIONAL(iris_openmp_kernel_with_obj,   iris_openmp_kernel_with_obj);
  LOADFUNCSYM_OPTIONAL(iris_openmp_setarg_with_obj,   iris_openmp_setarg_with_obj);
  LOADFUNCSYM_OPTIONAL(iris_openmp_setmem_with_obj,   iris_openmp_setmem_with_obj);
  LOADFUNCSYM_OPTIONAL(iris_openmp_launch_with_obj,   iris_openmp_launch_with_obj);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


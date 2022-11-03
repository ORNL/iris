#include "LoaderHost2CUDA.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2CUDA::LoaderHost2CUDA() : Loader() {
    iris_host2cuda_init = NULL;
    iris_host2cuda_init_handles = NULL;
    iris_host2cuda_finalize_handles = NULL;
    iris_host2cuda_finalize = NULL;
    iris_host2cuda_kernel = NULL;
    iris_host2cuda_setarg = NULL;
    iris_host2cuda_setmem = NULL;
    iris_host2cuda_launch = NULL;
    iris_host2cuda_kernel_with_obj = NULL;
    iris_host2cuda_setarg_with_obj = NULL;
    iris_host2cuda_setmem_with_obj = NULL;
    iris_host2cuda_launch_with_obj = NULL;
}

LoaderHost2CUDA::~LoaderHost2CUDA() {
}

const char* LoaderHost2CUDA::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("KERNEL_HOST2CUDA", &path, NULL);
  return path;
}

int LoaderHost2CUDA::LoadFunctions() {
  Loader::LoadFunctions();
  LOADFUNC_OPTIONAL(iris_host2cuda_init);
  LOADFUNC_OPTIONAL(iris_host2cuda_init_handles);
  LOADFUNC_OPTIONAL(iris_host2cuda_finalize_handles);
  LOADFUNC_OPTIONAL(iris_host2cuda_finalize);
  LOADFUNC_OPTIONAL(iris_host2cuda_kernel);
  LOADFUNC_OPTIONAL(iris_host2cuda_setarg);
  LOADFUNC_OPTIONAL(iris_host2cuda_setmem);
  LOADFUNC_OPTIONAL(iris_host2cuda_launch);
  LOADFUNC_OPTIONAL(iris_host2cuda_kernel_with_obj);
  LOADFUNC_OPTIONAL(iris_host2cuda_setarg_with_obj);
  LOADFUNC_OPTIONAL(iris_host2cuda_setmem_with_obj);
  LOADFUNC_OPTIONAL(iris_host2cuda_launch_with_obj);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


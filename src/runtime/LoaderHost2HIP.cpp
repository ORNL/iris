#include "LoaderHost2HIP.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2HIP::LoaderHost2HIP() : Loader() {
    iris_host2hip_init = NULL;
    iris_host2hip_init_handles = NULL;
    iris_host2hip_finalize_handles = NULL;
    iris_host2hip_finalize = NULL;
    iris_host2hip_kernel = NULL;
    iris_host2hip_setarg = NULL;
    iris_host2hip_setmem = NULL;
    iris_host2hip_launch = NULL;
    iris_host2hip_kernel_with_obj = NULL;
    iris_host2hip_setarg_with_obj = NULL;
    iris_host2hip_setmem_with_obj = NULL;
    iris_host2hip_launch_with_obj = NULL;
}

LoaderHost2HIP::~LoaderHost2HIP() {
}

const char* LoaderHost2HIP::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("KERNEL_HOST2HIP", &path, NULL);
  return path;
}

int LoaderHost2HIP::LoadFunctions() {
  Loader::LoadFunctions();
  LOADFUNC_OPTIONAL(iris_host2hip_init);
  LOADFUNC_OPTIONAL(iris_host2hip_init_handles);
  LOADFUNC_OPTIONAL(iris_host2hip_finalize_handles);
  LOADFUNC_OPTIONAL(iris_host2hip_finalize);
  LOADFUNC_OPTIONAL(iris_host2hip_kernel);
  LOADFUNC_OPTIONAL(iris_host2hip_setarg);
  LOADFUNC_OPTIONAL(iris_host2hip_setmem);
  LOADFUNC_OPTIONAL(iris_host2hip_launch);
  LOADFUNC_OPTIONAL(iris_host2hip_kernel_with_obj);
  LOADFUNC_OPTIONAL(iris_host2hip_setarg_with_obj);
  LOADFUNC_OPTIONAL(iris_host2hip_setmem_with_obj);
  LOADFUNC_OPTIONAL(iris_host2hip_launch_with_obj);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


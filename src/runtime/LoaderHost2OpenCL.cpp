#include "LoaderHost2OpenCL.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2OpenCL::LoaderHost2OpenCL(const char *suffix) : Loader() {
    iris_host2opencl_init = NULL;
    iris_host2opencl_init_handles = NULL;
    iris_host2opencl_finalize_handles = NULL;
    iris_host2opencl_finalize = NULL;
    iris_host2opencl_kernel = NULL;
    iris_host2opencl_setarg = NULL;
    iris_host2opencl_setmem = NULL;
    iris_host2opencl_launch = NULL;
    iris_host2opencl_set_queue = NULL;
    iris_host2opencl_set_queue_with_obj = NULL;
    iris_host2opencl_kernel_with_obj = NULL;
    iris_host2opencl_setarg_with_obj = NULL;
    iris_host2opencl_setmem_with_obj = NULL;
    iris_host2opencl_launch_with_obj = NULL;
    int len = strlen(suffix)+1;
    suffix_ = new char[len];
    for (int i=0; i<len; i++) {
      suffix_[i] = toupper(suffix[i]);
    }
}

LoaderHost2OpenCL::~LoaderHost2OpenCL() {
    delete [] suffix_;
}

const char* LoaderHost2OpenCL::library() {
  char* path = NULL;
  if (strcmp("CL", suffix_)==0)
      Platform::GetPlatform()->EnvironmentGet("KERNEL_HOST2OPENCL", &path, NULL);
  else {
      char filename[250];
      sprintf(filename, "KERNEL_HOST2OPENCL_%s", suffix_);
      Platform::GetPlatform()->EnvironmentGet(filename, &path, NULL);
  }
  //_trace("Path to loader Host2OpenCL: %s suffix_:%s", path, suffix_);
  return path;
}

int LoaderHost2OpenCL::LoadFunctions() {
  Loader::LoadFunctions();
  LOADFUNC_OPTIONAL(iris_host2opencl_init);
  LOADFUNC_OPTIONAL(iris_host2opencl_init_handles);
  LOADFUNC_OPTIONAL(iris_host2opencl_finalize_handles);
  LOADFUNC_OPTIONAL(iris_host2opencl_finalize);
  LOADFUNC_OPTIONAL(iris_host2opencl_kernel);
  LOADFUNC_OPTIONAL(iris_host2opencl_setarg);
  LOADFUNC_OPTIONAL(iris_host2opencl_setmem);
  LOADFUNC_OPTIONAL(iris_host2opencl_launch);
  LOADFUNC(iris_host2opencl_set_queue);
  LOADFUNC_OPTIONAL(iris_host2opencl_set_queue_with_obj);
  LOADFUNC_OPTIONAL(iris_host2opencl_kernel_with_obj);
  LOADFUNC_OPTIONAL(iris_host2opencl_setarg_with_obj);
  LOADFUNC_OPTIONAL(iris_host2opencl_setmem_with_obj);
  LOADFUNC_OPTIONAL(iris_host2opencl_launch_with_obj);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


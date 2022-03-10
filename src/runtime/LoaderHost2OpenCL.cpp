#include "LoaderHost2OpenCL.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2OpenCL::LoaderHost2OpenCL(const char *suffix) {
    iris_host2opencl_init = NULL;
    iris_host2opencl_finalize = NULL;
    iris_host2opencl_kernel = NULL;
    iris_host2opencl_setarg = NULL;
    iris_host2opencl_setmem = NULL;
    iris_host2opencl_launch = NULL;
    iris_host2opencl_set_handle = NULL;
    iris_host2opencl_get_handle = NULL;
    int len = strlen(suffix_)+1;
    suffix_ = new char[len];
    for (int i=0; i<len; i++) {
      suffix_[i] = toupper(suffix[i]);
    }
}

LoaderHost2OpenCL::~LoaderHost2OpenCL() {
    delete suffix_;
}

const char* LoaderHost2OpenCL::library() {
  char* path = NULL;
  if (strcmp("cl", suffix_)==0)
      Platform::GetPlatform()->EnvironmentGet("KERNEL_HOST2OPENCL", &path, NULL);
  else {
      char filename[250];
      sprintf(filename, "KERNEL_HOST2OPENCL_%x", suffix_);
      Platform::GetPlatform()->EnvironmentGet(filename, &path, NULL);
  }
  //_trace("Path to loader Host2OpenCL: %s suffix_:%s", path, suffix_);
  return path;
}

int LoaderHost2OpenCL::LoadFunctions() {
  LOADFUNC(iris_host2opencl_init);
  LOADFUNC(iris_host2opencl_finalize);
  LOADFUNC(iris_host2opencl_kernel);
  LOADFUNC(iris_host2opencl_setarg);
  LOADFUNC(iris_host2opencl_setmem);
  LOADFUNC(iris_host2opencl_launch);
  LOADFUNC(iris_host2opencl_set_handle);
  LOADFUNC(iris_host2opencl_get_handle);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


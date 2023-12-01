#include "LoaderHost2OpenCL.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2OpenCL::LoaderHost2OpenCL(const char *suffix) : HostInterfaceClass("KERNEL_HOST2OPENCL") {
    enable_strict_handle_check();
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
  HostInterfaceClass::LoadFunctions();
  REGISTER_HOST_WRAPPER(iris_host_init,             iris_host2opencl_init             );
  REGISTER_HOST_WRAPPER(iris_host_init_handles,     iris_host2opencl_init_handles     );
  REGISTER_HOST_WRAPPER(iris_host_finalize_handles, iris_host2opencl_finalize_handles );
  REGISTER_HOST_WRAPPER(iris_host_finalize,         iris_host2opencl_finalize         );
  REGISTER_HOST_WRAPPER(iris_host_kernel,           iris_host2opencl_kernel           );
  REGISTER_HOST_WRAPPER(iris_host_setarg,           iris_host2opencl_setarg           );
  REGISTER_HOST_WRAPPER(iris_host_setmem,           iris_host2opencl_setmem           );
  REGISTER_HOST_WRAPPER(iris_host_launch,           iris_host2opencl_launch           );
  REGISTER_HOST_WRAPPER(iris_host_kernel_with_obj,  iris_host2opencl_kernel_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setarg_with_obj,  iris_host2opencl_setarg_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setmem_with_obj,  iris_host2opencl_setmem_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_launch_with_obj,  iris_host2opencl_launch_with_obj  );
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


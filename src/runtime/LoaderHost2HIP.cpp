#include "LoaderHost2HIP.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2HIP::LoaderHost2HIP() : HostInterfaceClass("KERNEL_HOST2HIP") {
    enable_strict_handle_check();
}

LoaderHost2HIP::~LoaderHost2HIP() {
}

int LoaderHost2HIP::LoadFunctions() {
  HostInterfaceClass::LoadFunctions();
  REGISTER_HOST_WRAPPER(iris_host_init,             iris_host2hip_init             );
  REGISTER_HOST_WRAPPER(iris_host_init_handles,     iris_host2hip_init_handles     );
  REGISTER_HOST_WRAPPER(iris_host_finalize_handles, iris_host2hip_finalize_handles );
  REGISTER_HOST_WRAPPER(iris_host_finalize,         iris_host2hip_finalize         );
  REGISTER_HOST_WRAPPER(iris_host_kernel,           iris_host2hip_kernel           );
  REGISTER_HOST_WRAPPER(iris_host_setarg,           iris_host2hip_setarg           );
  REGISTER_HOST_WRAPPER(iris_host_setmem,           iris_host2hip_setmem           );
  REGISTER_HOST_WRAPPER(iris_host_launch,           iris_host2hip_launch           );
  REGISTER_HOST_WRAPPER(iris_host_kernel_with_obj,  iris_host2hip_kernel_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setarg_with_obj,  iris_host2hip_setarg_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setmem_with_obj,  iris_host2hip_setmem_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_launch_with_obj,  iris_host2hip_launch_with_obj  );
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */



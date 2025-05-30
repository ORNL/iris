#include "LoaderHost2CUDA.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {


LoaderHost2CUDA::LoaderHost2CUDA() : HostInterfaceClass("KERNEL_HOST2CUDA") {
    enable_strict_handle_check();
}

LoaderHost2CUDA::~LoaderHost2CUDA() {
}

int LoaderHost2CUDA::LoadFunctions() {
  HostInterfaceClass::LoadFunctions();
  REGISTER_HOST_WRAPPER(iris_host_init,             iris_host2cuda_init             );
  REGISTER_HOST_WRAPPER(iris_host_init_handles,     iris_host2cuda_init_handles     );
  REGISTER_HOST_WRAPPER(iris_host_finalize_handles, iris_host2cuda_finalize_handles );
  REGISTER_HOST_WRAPPER(iris_host_finalize,         iris_host2cuda_finalize         );
  REGISTER_HOST_WRAPPER(iris_host_kernel,           iris_host2cuda_kernel           );
  REGISTER_HOST_WRAPPER(iris_host_setarg,           iris_host2cuda_setarg           );
  REGISTER_HOST_WRAPPER(iris_host_setmem,           iris_host2cuda_setmem           );
  REGISTER_HOST_WRAPPER(iris_host_launch,           iris_host2cuda_launch           );
  REGISTER_HOST_WRAPPER(iris_host_kernel_with_obj,  iris_host2cuda_kernel_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setarg_with_obj,  iris_host2cuda_setarg_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setmem_with_obj,  iris_host2cuda_setmem_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_launch_with_obj,  iris_host2cuda_launch_with_obj  );
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


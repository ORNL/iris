#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

#include "LoaderOpenMP.h"

namespace CLinkage {
    extern "C" {
    int iris_openmp_init();
    int iris_openmp_init_handles(int devno);
    int iris_openmp_finalize_handles(int devno);
    int iris_openmp_finalize();
    int iris_openmp_kernel(const char* name);
    int iris_openmp_setarg(int idx, size_t size, void* value);
    int iris_openmp_setmem(int idx, void* mem);
    int iris_openmp_launch(int dim, size_t off, size_t gws);
    void *iris_openmp_get_kernel_ptr(const char* name);
    int iris_openmp_kernel_with_obj(void *obj, const char* name);
    int iris_openmp_setarg_with_obj(void *obj, int idx, size_t size, void* value);
    int iris_openmp_setmem_with_obj(void *obj, int idx, void* mem);
    int iris_openmp_launch_with_obj(void *stream, void *obj, int devno, int dim, size_t off, size_t gws);
    }
}
namespace iris {
namespace rt {

LoaderOpenMP::LoaderOpenMP() : HostInterfaceClass("KERNEL_BIN_OPENMP") {
}

LoaderOpenMP::~LoaderOpenMP() {
}

int LoaderOpenMP::LoadFunctions() {
  HostInterfaceClass::LoadFunctions();
  REGISTER_HOST_WRAPPER(iris_host_init,             iris_openmp_init             );
  REGISTER_HOST_WRAPPER(iris_host_init_handles,     iris_openmp_init_handles     );
  REGISTER_HOST_WRAPPER(iris_host_finalize_handles, iris_openmp_finalize_handles );
  REGISTER_HOST_WRAPPER(iris_host_finalize,         iris_openmp_finalize         );
  REGISTER_HOST_WRAPPER(iris_host_kernel,           iris_openmp_kernel           );
  REGISTER_HOST_WRAPPER(iris_host_setarg,           iris_openmp_setarg           );
  REGISTER_HOST_WRAPPER(iris_host_setmem,           iris_openmp_setmem           );
  REGISTER_HOST_WRAPPER(iris_host_launch,           iris_openmp_launch           );
  REGISTER_HOST_WRAPPER(iris_host_kernel_with_obj,  iris_openmp_kernel_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setarg_with_obj,  iris_openmp_setarg_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_setmem_with_obj,  iris_openmp_setmem_with_obj  );
  REGISTER_HOST_WRAPPER(iris_host_launch_with_obj,  iris_openmp_launch_with_obj  );
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


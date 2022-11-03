#ifndef IRIS_SRC_RT_LOADER_HOST2OpenCL_H
#define IRIS_SRC_RT_LOADER_HOST2OpenCL_H

#include "Loader.h"

namespace iris {
namespace rt {

class LoaderHost2OpenCL : public Loader {
public:
  LoaderHost2OpenCL(const char *suffix);
  ~LoaderHost2OpenCL();

  const char* library();
  int LoadFunctions();

  int (*iris_host2opencl_init)();
  int (*iris_host2opencl_init_handles)(int devno);
  int (*iris_host2opencl_finalize_handles)(int devno);
  int (*iris_host2opencl_finalize)();
  void *(*iris_host2opencl_set_queue)(void *);
  void *(*iris_host2opencl_set_queue_with_obj)(void *, void *);
  int (*iris_host2opencl_kernel)(const char* name);
  int (*iris_host2opencl_setarg)(int idx, size_t size, void* value);
  int (*iris_host2opencl_setmem)(int idx, void* mem);
  int (*iris_host2opencl_launch)(int dim, size_t off, size_t gws);
  int (*iris_host2opencl_kernel_with_obj)(void *obj, const char* name);
  int (*iris_host2opencl_setarg_with_obj)(void *obj, int idx, size_t size, void* value);
  int (*iris_host2opencl_setmem_with_obj)(void *obj, int idx, void* mem);
  int (*iris_host2opencl_launch_with_obj)(void *obj, int devno, int dim, size_t off, size_t gws);
  char *suffix_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2OpenCL_H */


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
  int (*iris_host2opencl_finalize)();
  void *(*iris_host2opencl_set_handle)(void *);
  void *(*iris_host2opencl_get_handle)();
  int (*iris_host2opencl_kernel)(const char* name);
  int (*iris_host2opencl_setarg)(int idx, size_t size, void* value);
  int (*iris_host2opencl_setmem)(int idx, void* mem);
  int (*iris_host2opencl_launch)(int dim, size_t off, size_t gws);
  char *suffix_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2OpenCL_H */


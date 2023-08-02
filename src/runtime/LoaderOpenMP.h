#ifndef IRIS_SRC_RT_LOADER_OPENMP_H
#define IRIS_SRC_RT_LOADER_OPENMP_H

#include "Loader.h"

namespace iris {
namespace rt {

class LoaderOpenMP : public Loader {
public:
  LoaderOpenMP();
  ~LoaderOpenMP();

  const char* library();
  int LoadFunctions();
#ifdef DISABLE_DYNAMIC_LINKING
  bool IsFunctionExists(const char *name);
  void *GetFunctionPtr(const char *kernel_name);
  int SetKernelPtr(void *obj, char *kernel_name);
#endif
  int (*iris_openmp_init)();
  int (*iris_openmp_init_handles)(int devno);
  int (*iris_openmp_finalize_handles)(int devno);
  int (*iris_openmp_finalize)();
  int (*iris_openmp_kernel)(const char* name);
  int (*iris_openmp_setarg)(int idx, size_t size, void* value);
  int (*iris_openmp_setmem)(int idx, void* mem);
  int (*iris_openmp_launch)(int dim, size_t off, size_t gws);
  void *(*iris_openmp_get_kernel_ptr)(const char* name);
  int (*iris_openmp_kernel_with_obj)(void *obj, const char* name);
  int (*iris_openmp_setarg_with_obj)(void *obj, int idx, size_t size, void* value);
  int (*iris_openmp_setmem_with_obj)(void *obj, int idx, void* mem);
  int (*iris_openmp_launch_with_obj)(void *stream, void *obj, int devno, int dim, size_t off, size_t gws);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_OPENMP_H */


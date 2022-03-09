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

  int (*iris_openmp_init)();
  int (*iris_openmp_finalize)();
  int (*iris_openmp_kernel)(const char* name);
  int (*iris_openmp_setarg)(int idx, size_t size, void* value);
  int (*iris_openmp_setmem)(int idx, void* mem);
  int (*iris_openmp_launch)(int dim, size_t off, size_t gws);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_OPENMP_H */


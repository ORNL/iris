#ifndef IRIS_SRC_RT_LOADER_HOST2CUDA_H
#define IRIS_SRC_RT_LOADER_HOST2CUDA_H

#include "Loader.h"

namespace iris {
namespace rt {

class LoaderHost2CUDA : public Loader {
public:
  LoaderHost2CUDA();
  ~LoaderHost2CUDA();

  const char* library();
  int LoadFunctions();

  int (*iris_host2cuda_init)();
  int (*iris_host2cuda_finalize)();
  int (*iris_host2cuda_kernel)(const char* name);
  int (*iris_host2cuda_setarg)(int idx, size_t size, void* value);
  int (*iris_host2cuda_setmem)(int idx, void* mem);
  int (*iris_host2cuda_launch)(int dim, size_t off, size_t gws);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2CUDA_H */


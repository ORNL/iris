#ifndef IRIS_SRC_RT_LOADER_HOST2HIP_H
#define IRIS_SRC_RT_LOADER_HOST2HIP_H

#include "Loader.h"

namespace iris {
namespace rt {

class LoaderHost2HIP : public Loader {
public:
  LoaderHost2HIP();
  ~LoaderHost2HIP();

  const char* library();
  int LoadFunctions();

  int (*iris_host2hip_init)();
  int (*iris_host2hip_finalize)();
  int (*iris_host2hip_kernel)(const char* name);
  int (*iris_host2hip_setarg)(int idx, size_t size, void* value);
  int (*iris_host2hip_setmem)(int idx, void* mem);
  int (*iris_host2hip_launch)(int dim, size_t off, size_t gws);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2HIP_H */


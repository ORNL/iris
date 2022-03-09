#ifndef IRIS_SRC_RT_LOADER_HEXAGON_H
#define IRIS_SRC_RT_LOADER_HEXAGON_H

#include "Loader.h"
#include <stdint.h>

namespace iris {
namespace rt {

class LoaderHexagon : public Loader {
public:
  LoaderHexagon();
  ~LoaderHexagon();

  const char* library();
  int LoadFunctions();

  int (*iris_hexagon_init)();
  int (*iris_hexagon_finalize)();
  int (*iris_hexagon_kernel)(const char* name);
  int (*iris_hexagon_setarg)(int idx, size_t size, void* value);
  int (*iris_hexagon_setmem)(int idx, void* mem, int size);
  int (*iris_hexagon_launch)(int dim, size_t off, size_t gws);
  
  void* (*iris_hexagon_rpcmem_alloc)(int heapid, uint32_t flags, int size);
  void (*iris_hexagon_rpcmem_free)(void* po);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HEXAGON_H */


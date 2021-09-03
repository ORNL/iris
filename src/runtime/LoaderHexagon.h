#ifndef BRISBANE_SRC_RT_LOADER_HEXAGON_H
#define BRISBANE_SRC_RT_LOADER_HEXAGON_H

#include "Loader.h"
#include <stdint.h>

namespace brisbane {
namespace rt {

class LoaderHexagon : public Loader {
public:
  LoaderHexagon();
  ~LoaderHexagon();

  const char* library();
  int LoadFunctions();

  int (*brisbane_hexagon_init)();
  int (*brisbane_hexagon_finalize)();
  int (*brisbane_hexagon_kernel)(const char* name);
  int (*brisbane_hexagon_setarg)(int idx, size_t size, void* value);
  int (*brisbane_hexagon_setmem)(int idx, void* mem, int size);
  int (*brisbane_hexagon_launch)(int dim, size_t off, size_t gws);
  
  void* (*brisbane_hexagon_rpcmem_alloc)(int heapid, uint32_t flags, int size);
  void (*brisbane_hexagon_rpcmem_free)(void* po);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_LOADER_HEXAGON_H */


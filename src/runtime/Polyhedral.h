#ifndef BRISBANE_SRC_RT_POLYHEDRAL_H
#define BRISBANE_SRC_RT_POLYHEDRAL_H

#include <iris/brisbane.h>
#include <iris/brisbane_poly_types.h>
#include "Loader.h"

namespace brisbane {
namespace rt {

class Polyhedral : public Loader {
public:
  Polyhedral();
  ~Polyhedral();

  const char* library() { return "kernel.poly.so"; }

  int LoadFunctions();

  int Kernel(const char* name);
  int SetArg(int idx, size_t size, void* value);
  int Launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws);
  int GetMem(int idx, brisbane_poly_mem* plmem);

private:
  int (*brisbane_poly_init)();
  int (*brisbane_poly_finalize)();
  int (*brisbane_poly_kernel)(const char* name);
  int (*brisbane_poly_setarg)(int idx, size_t size, void* value);
  int (*brisbane_poly_launch)(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws);
  int (*brisbane_poly_getmem)(int idx, brisbane_poly_mem* plmem);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLYHEDRAL_H */

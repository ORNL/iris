#ifndef IRIS_SRC_RT_POLYHEDRAL_H
#define IRIS_SRC_RT_POLYHEDRAL_H

#include <iris/iris.h>
#include <iris/iris_poly_types.h>
#include "Loader.h"

namespace iris {
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
  int GetMem(int idx, iris_poly_mem* plmem);

private:
  int (*iris_poly_init)();
  int (*iris_poly_finalize)();
  int (*iris_poly_kernel)(const char* name);
  int (*iris_poly_setarg)(int idx, size_t size, void* value);
  int (*iris_poly_launch)(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws);
  int (*iris_poly_getmem)(int idx, iris_poly_mem* plmem);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLYHEDRAL_H */

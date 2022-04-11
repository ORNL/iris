#include "Polyhedral.h"
#include "Debug.h"
#include "Loader.h"
#include <dlfcn.h>

namespace iris {
namespace rt {

Polyhedral::Polyhedral() {
}

Polyhedral::~Polyhedral() {
  if (handle_) iris_poly_finalize();
}

int Polyhedral::LoadFunctions() {
  LOADFUNC(iris_poly_init);
  LOADFUNC(iris_poly_finalize);
  LOADFUNC(iris_poly_kernel);
  LOADFUNC(iris_poly_setarg);
  LOADFUNC(iris_poly_launch);
  LOADFUNC(iris_poly_getmem);

  iris_poly_init();

  return IRIS_SUCCESS;
}

int Polyhedral::Kernel(const char* name) {
  return iris_poly_kernel(name);
}

int Polyhedral::SetArg(int idx, size_t size, void* value) {
  return iris_poly_setarg(idx, size, value);
}

int Polyhedral::Launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws) {
  return iris_poly_launch(dim, wgo, wgs, gws, lws);
}

int Polyhedral::GetMem(int idx, iris_poly_mem* plmem) {
  return iris_poly_getmem(idx, plmem);
}

} /* namespace rt */
} /* namespace iris */


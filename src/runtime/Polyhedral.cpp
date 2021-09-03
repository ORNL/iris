#include "Polyhedral.h"
#include "Debug.h"
#include "Loader.h"
#include <dlfcn.h>

namespace brisbane {
namespace rt {

Polyhedral::Polyhedral() {
}

Polyhedral::~Polyhedral() {
  if (handle_) brisbane_poly_finalize();
}

int Polyhedral::LoadFunctions() {
  LOADFUNC(brisbane_poly_init);
  LOADFUNC(brisbane_poly_finalize);
  LOADFUNC(brisbane_poly_kernel);
  LOADFUNC(brisbane_poly_setarg);
  LOADFUNC(brisbane_poly_launch);
  LOADFUNC(brisbane_poly_getmem);

  brisbane_poly_init();

  return BRISBANE_OK;
}

int Polyhedral::Kernel(const char* name) {
  return brisbane_poly_kernel(name);
}

int Polyhedral::SetArg(int idx, size_t size, void* value) {
  return brisbane_poly_setarg(idx, size, value);
}

int Polyhedral::Launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws) {
  return brisbane_poly_launch(dim, wgo, wgs, gws, lws);
}

int Polyhedral::GetMem(int idx, brisbane_poly_mem* plmem) {
  return brisbane_poly_getmem(idx, plmem);
}

} /* namespace rt */
} /* namespace brisbane */


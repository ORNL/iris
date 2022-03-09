#include "LoaderHexagon.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>
#include <unistd.h>

namespace iris {
namespace rt {

LoaderHexagon::LoaderHexagon() {
}

LoaderHexagon::~LoaderHexagon() {
}

const char* LoaderHexagon::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("KERNEL_BIN_HEXAGON", &path, NULL);
  return path;
}


int LoaderHexagon::LoadFunctions() {
  LOADFUNC(iris_hexagon_init);
  LOADFUNC(iris_hexagon_finalize);
  LOADFUNC(iris_hexagon_kernel);
  LOADFUNC(iris_hexagon_setarg);
  LOADFUNC(iris_hexagon_setmem);
  LOADFUNC(iris_hexagon_launch);

  LOADFUNC(iris_hexagon_rpcmem_alloc);
  LOADFUNC(iris_hexagon_rpcmem_free);

  return IRIS_OK;
}

} /* namespace rt */
} /* namespace iris */


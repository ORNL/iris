#include "LoaderHexagon.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>
#include <unistd.h>

namespace brisbane {
namespace rt {

LoaderHexagon::LoaderHexagon() {
}

LoaderHexagon::~LoaderHexagon() {
}

const char* LoaderHexagon::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("KERNEL_HEXAGON", &path, NULL);
  return path;
}


int LoaderHexagon::LoadFunctions() {
  LOADFUNC(brisbane_hexagon_init);
  LOADFUNC(brisbane_hexagon_finalize);
  LOADFUNC(brisbane_hexagon_kernel);
  LOADFUNC(brisbane_hexagon_setarg);
  LOADFUNC(brisbane_hexagon_setmem);
  LOADFUNC(brisbane_hexagon_launch);

  LOADFUNC(brisbane_hexagon_rpcmem_alloc);
  LOADFUNC(brisbane_hexagon_rpcmem_free);

  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


#include "LoaderHexagon.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>
#include <unistd.h>

namespace iris {
namespace rt {

LoaderHexagon::LoaderHexagon() {
    iris_hexagon_init = NULL;
    iris_hexagon_finalize = NULL;
    iris_hexagon_kernel = NULL;
    iris_hexagon_setarg= NULL;
    iris_hexagon_setmem = NULL;
    iris_hexagon_launch = NULL;
    iris_hexagon_kernel_with_obj = NULL;
    iris_hexagon_setarg_with_obj= NULL;
    iris_hexagon_setmem_with_obj = NULL;
    iris_hexagon_launch_with_obj = NULL;
    iris_get_kernel_names = NULL;
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
  LOADFUNC_OPTIONAL(iris_hexagon_kernel);
  LOADFUNC_OPTIONAL(iris_hexagon_setarg);
  LOADFUNC_OPTIONAL(iris_hexagon_setmem);
  LOADFUNC_OPTIONAL(iris_hexagon_launch);
  LOADFUNC_OPTIONAL(iris_hexagon_kernel_with_obj);
  LOADFUNC_OPTIONAL(iris_hexagon_setarg_with_obj);
  LOADFUNC_OPTIONAL(iris_hexagon_setmem_with_obj);
  LOADFUNC_OPTIONAL(iris_hexagon_launch_with_obj);
  LOADFUNC_OPTIONAL(iris_get_kernel_names);

  LOADFUNC(iris_hexagon_rpcmem_alloc);
  LOADFUNC(iris_hexagon_rpcmem_free);

  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


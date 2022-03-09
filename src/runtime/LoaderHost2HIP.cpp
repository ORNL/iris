#include "LoaderHost2HIP.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2HIP::LoaderHost2HIP() {
    iris_host2hip_init = NULL;
    iris_host2hip_finalize = NULL;
    iris_host2hip_kernel = NULL;
    iris_host2hip_setarg = NULL;
    iris_host2hip_setmem = NULL;
    iris_host2hip_launch = NULL;
}

LoaderHost2HIP::~LoaderHost2HIP() {
}

const char* LoaderHost2HIP::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("kernel.host2hip", &path, NULL);
  return path;
}

int LoaderHost2HIP::LoadFunctions() {
  LOADFUNC(iris_host2hip_init);
  LOADFUNC(iris_host2hip_finalize);
  LOADFUNC(iris_host2hip_kernel);
  LOADFUNC(iris_host2hip_setarg);
  LOADFUNC(iris_host2hip_setmem);
  LOADFUNC(iris_host2hip_launch);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


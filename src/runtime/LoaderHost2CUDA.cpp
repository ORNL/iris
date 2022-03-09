#include "LoaderHost2CUDA.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {

LoaderHost2CUDA::LoaderHost2CUDA() {
    iris_host2cuda_init = NULL;
    iris_host2cuda_finalize = NULL;
    iris_host2cuda_kernel = NULL;
    iris_host2cuda_setarg = NULL;
    iris_host2cuda_setmem = NULL;
    iris_host2cuda_launch = NULL;
}

LoaderHost2CUDA::~LoaderHost2CUDA() {
}

const char* LoaderHost2CUDA::library() {
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("kernel.host2cuda", &path, NULL);
  return path;
}

int LoaderHost2CUDA::LoadFunctions() {
  LOADFUNC(iris_host2cuda_init);
  LOADFUNC(iris_host2cuda_finalize);
  LOADFUNC(iris_host2cuda_kernel);
  LOADFUNC(iris_host2cuda_setarg);
  LOADFUNC(iris_host2cuda_setmem);
  LOADFUNC(iris_host2cuda_launch);
  return IRIS_OK;
}

} /* namespace rt */
} /* namespace iris */


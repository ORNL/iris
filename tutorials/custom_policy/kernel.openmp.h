#include <iris/iris_openmp.h>

static void setid(int* mem, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(mem) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  mem[i] = i;
  IRIS_OPENMP_KERNEL_END
}


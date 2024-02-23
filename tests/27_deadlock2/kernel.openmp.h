#include <iris/iris_openmp.h>

void copy(int* dst, int *src, IRIS_OPENMP_KERNEL_ARGS) {
  size_t i;
#pragma omp parallel for shared(dst, src) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  dst[i] = src[i];
  IRIS_OPENMP_KERNEL_END
}


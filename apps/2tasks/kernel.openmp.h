#include <iris/iris_openmp.h>

static void kernel0(float* dst, float* src, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(dst, src) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  dst[i] = src[i];
  IRIS_OPENMP_KERNEL_END
}

static void kernel1(float* dst, float* src, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(dst, src) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  dst[i] += src[i];
  IRIS_OPENMP_KERNEL_END
}


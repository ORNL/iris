#include <iris/iris_openmp.h>

static void saxpy(float* Z, float A, float* X, float* Y, IRIS_OPENMP_KERNEL_ARGS) {
  size_t i;
#pragma omp parallel for shared(Z, A, X, Y) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  Z[i] = A * X[i] + Y[i];
  IRIS_OPENMP_KERNEL_END
}


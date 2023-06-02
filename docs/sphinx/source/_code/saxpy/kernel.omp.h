#include <iris/iris_openmp.h>

static void saxpy(float* S, float A, float* X, float* Y, IRIS_OPENMP_KERNEL_ARGS) {
  int i = 0;
#pragma omp parallel for shared(S, A, X, Y) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  S[i] = A * X[i] + Y[i];
  IRIS_OPENMP_KERNEL_END
}

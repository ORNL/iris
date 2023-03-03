#include <brisbane/brisbane_openmp.h>

static void saxpy(float* Z, float A, float* X, float* Y, BRISBANE_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(Z, A, X, Y) private(_id)
  BRISBANE_OPENMP_KERNEL_BEGIN
  Z[_id] = A * X[_id] + Y[_id];
  BRISBANE_OPENMP_KERNEL_END
}


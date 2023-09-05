#include <iris/iris_openmp.h>

void saxpy(int* Z, int* X, int* Y, int A, IRIS_OPENMP_KERNEL_ARGS) {
  size_t id;
#pragma omp parallel for shared(Z, X, Y) private(_id)
  IRIS_OPENMP_KERNEL_BEGIN(id)
  Z[id] = A * X[id] + Y[id];
  IRIS_OPENMP_KERNEL_END
}


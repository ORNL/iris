#include <iris/iris_openmp.h>

static void saxpy(float* Z, float A, float* X, float* Y, IRIS_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(Z, A, X, Y) private(_id)
  IRIS_OPENMP_KERNEL_BEGIN
  Z[_id] = A * X[_id] + Y[_id];
  IRIS_OPENMP_KERNEL_END
}

static void ijk(double* C, double* A, double* B, IRIS_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(C, A, B) private(_id)
  IRIS_OPENMP_KERNEL_BEGIN
  size_t SIZE = _ndr;
  size_t j, k;
  for (size_t j = 0; j < SIZE; j++) {
    double sum = 0.0;
    for (size_t k = 0; k < SIZE; k++) {
      sum += A[_id * SIZE + k] * B[k * SIZE + j];
    }
    C[_id * SIZE + j] = sum;
  }
  IRIS_OPENMP_KERNEL_END
}


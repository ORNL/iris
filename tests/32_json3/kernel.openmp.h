#include <iris/iris_openmp.h>

void process(int* A, IRIS_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(A) private(_id)
  IRIS_OPENMP_KERNEL_BEGIN(_id)
  A[_id] *= 100;
  IRIS_OPENMP_KERNEL_END
}

void ijk(double* C, double* A, double* B, size_t *_off, size_t *_ndr) {
  const size_t SIZE = _ndr[0];
#pragma omp parallel for collapse(2) shared(A,B,C) private(SIZE)
  //IRIS_OPENMP_KERNEL_BEGIN(i)
  for (size_t i = _off[0]; i < _off[0] + _ndr[0]; i++) {
    for (size_t j = 0; j < SIZE; j++){
      double sum = 0.0;
      for (size_t k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;
    }
  }
  //IRIS_OPENMP_KERNEL_END
}

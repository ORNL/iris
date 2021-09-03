#include <brisbane/brisbane_openmp.h>

static void ijk(float* C, float* A, float* B, BRISBANE_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C, A, B) private(i)
  BRISBANE_OPENMP_KERNEL_BEGIN (i)
  for (int j = 0; j < _ndr; j++) {
    float sum = 0.0;
    for (int k = 0; k < _ndr; k++) {
      sum += A[i * _ndr + k] * B[k * _ndr + j];
    }
    C[i * _ndr + j] = sum;
  }
  BRISBANE_OPENMP_KERNEL_END
}


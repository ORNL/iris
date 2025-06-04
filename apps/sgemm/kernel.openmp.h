#include <iris/iris_openmp.h>

static void ijk(float* C, float* A, float* B, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C, A, B) private(i)
  IRIS_OPENMP_KERNEL_BEGIN (i)
  for (int j = 0; j < *_bws; j++) {
    float sum = 0.0;
    for (int k = 0; k < *_bws; k++) {
      sum += A[i * (*_bws) + k] * B[k * (*_bws) + j];
    }
    C[i * (*_bws) + j] = sum;
  }
  IRIS_OPENMP_KERNEL_END
}


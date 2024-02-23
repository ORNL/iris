#include <iris/iris_openmp.h>

void vecadd(int* C, int* A, int* B, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C, A, B) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  C[i] += A[i] + B[i];
  IRIS_OPENMP_KERNEL_END
}


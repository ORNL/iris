#include <iris/iris_openmp.h>

static void loop0(int* A, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(A) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  A[i] *= 2;
  IRIS_OPENMP_KERNEL_END
}

static void loop1(int* B, int* A, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(B,A) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  B[i] += A[i];
  IRIS_OPENMP_KERNEL_END
}

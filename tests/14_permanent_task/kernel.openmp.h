#include <iris/iris_openmp.h>

static void process(int* A, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(A) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  A[i]++;
  IRIS_OPENMP_KERNEL_END
}


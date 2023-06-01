#include <iris/iris_openmp.h>

static void kernel_A(int* AB, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(AB) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  AB[i] = i;
  IRIS_OPENMP_KERNEL_END
}

static void kernel_B(int* AB, int* BC, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(AB, BC) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  BC[i] = AB[i] * 10;
  IRIS_OPENMP_KERNEL_END
}

static void kernel_C(int* BC, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(BC) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  BC[i] = BC[i] * 2;
  IRIS_OPENMP_KERNEL_END
}


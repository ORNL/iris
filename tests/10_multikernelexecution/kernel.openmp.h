#include <iris/iris_openmp.h>

void kernel0(int* C, int loop, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  for (int k = 0; k < loop; k++) {
  for (int j = 0; j < loop; j++) {
    C[i] += i;
  }
  }
  IRIS_OPENMP_KERNEL_END
}

void kernel1(int* C, int loop, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  for (int k = 0; k < loop; k++) {
  for (int j = 0; j < loop; j++) {
    C[i] += i;
  }
  }
  IRIS_OPENMP_KERNEL_END
}

void kernel2(int* C, int loop, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  for (int k = 0; k < loop; k++) {
  for (int j = 0; j < loop; j++) {
    C[i] += i;
  }
  }
  IRIS_OPENMP_KERNEL_END
}

void kernel3(int* C, int loop, IRIS_OPENMP_KERNEL_ARGS) {
  int i;
#pragma omp parallel for shared(C) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  for (int k = 0; k < loop; k++) {
  for (int j = 0; j < loop; j++) {
    C[i] += i;
  }
  }
  IRIS_OPENMP_KERNEL_END
}


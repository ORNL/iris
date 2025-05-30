#include <iris/iris_openmp.h>

void vecadd(int* A, int* B, int* C, IRIS_OPENMP_KERNEL_ARGS) {
  size_t i;
#pragma omp parallel for shared(C, A, B) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  C[i] = A[i] + B[i];
  IRIS_OPENMP_KERNEL_END
}
void blockadd(int* A, int* B, int* C, size_t SIZE, IRIS_OPENMP_KERNEL_ARGS) {
  size_t i, j;
  //size_t SIZE=16;
#pragma omp parallel for shared(C, A, B) private(i, j)
  IRIS_OPENMP_KERNEL_BEGIN2D(i, j)
  C[i*SIZE+j] = A[i*SIZE+j] + B[i*SIZE+j];
  IRIS_OPENMP_KERNEL_END2D
}


#include <iris/iris_openmp.h>
#include <stdio.h>

static void saxpy(float* Z, float A, float* X, float* Y, IRIS_OPENMP_KERNEL_ARGS) {
  size_t i;
  int SIZE = 8;
  printf("Before input first = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", X[i]);
  printf("]\n");

  printf("Before input second = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Y[i]);
  printf("]\n");

  printf("Before output third = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");


#pragma omp parallel for shared(Z, A, X, Y) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  Z[i] = A * X[i] + Y[i];
  IRIS_OPENMP_KERNEL_END


  printf("After input first = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", X[i]);
  printf("]\n");

  printf("After input second = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Y[i]);
  printf("]\n");

  printf("After output third = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n\n\n\n");



}


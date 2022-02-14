#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B, *C, *D, *E;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));
  D = (int*) malloc(SIZE * sizeof(int));
  E = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
  }

#pragma acc parallel loop copyin(A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma omp target teams distribute parallel for map(to:A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma brisbane kernel h2d(A[0:SIZE], B[0:SIZE]) alloc(C[0:SIZE]) device(gpu)
  for (int i = 0; i < SIZE; i++) {
    C[i] = A[i] + B[i];
  }

#pragma acc parallel loop present(C[0:SIZE]) device(cpu)
#pragma omp target teams distribute parallel for device(cpu)
#pragma brisbane kernel present(C[0:SIZE]) device(cpu)
  for (int i = 0; i < SIZE; i++) {
    D[i] = C[i] * 10;
  }

#pragma acc parallel loop present(D[0:SIZE]) device(data)
#pragma omp target teams distribute parallel for map(from:E[0:SIZE]) device(data)
#pragma brisbane kernel d2h(E[0:SIZE]) present(D[0:SIZE]) device(data)
  for (int i = 0; i < SIZE; i++) {
    E[i] = D[i] * 2;
  }

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8d = (%8d + %8d) * %d\n", i, E[i], A[i], B[i], 20);
    if (E[i] != (A[i] + B[i]) * 20) ERROR++;
  }
  printf("ERROR[%d]\n", ERROR);

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B, *C;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%d]\n", SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i;
    C[i] = 0;
  }

#pragma acc parallel loop copyin(A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma omp target teams distribute parallel for map(to:A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma iris kernel h2d(A[0:SIZE], B[0:SIZE]) alloc(C[0:SIZE]) device(gpu)
  for (int i = 0; i < SIZE; i++) {
    C[i] = A[i] + B[i];
  }

  for (int i = 0; i < SIZE; i++) {
    printf("C[%d] = %d\n", i, C[i]);
    if (C[i] != (A[i] + B[i])) ERROR++;
  }
  printf("ERROR[%d]\n", ERROR);

  free(A);
  free(B);
  free(C);

  return ERROR;
}

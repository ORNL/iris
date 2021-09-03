#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, 1);

  size_t SIZE;
  int TARGET;
  int VERBOSE;
  float *X, *Y, *Z;
  float A = 10;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  VERBOSE = argc > 3 ? atol(argv[3]) : 0;

  printf("[%s:%d] SIZE[%zu] TARGET[%d] VERBOSE[%d]\n", __FILE__, __LINE__, SIZE, TARGET, VERBOSE);

  X = (float*) malloc(SIZE * sizeof(float));
  Y = (float*) malloc(SIZE * sizeof(float));
  Z = (float*) malloc(SIZE * sizeof(float));

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    X[i] = i;
    Y[i] = i;
  }

  printf("X [");
  for (int i = 0; i < SIZE; i++) printf(" %2.0f.", X[i]);
  printf("]\n");
  printf("Y [");
  for (int i = 0; i < SIZE; i++) printf(" %2.0f.", Y[i]);
  printf("]\n");

  }

  brisbane_mem mem_X;
  brisbane_mem mem_Y;
  brisbane_mem mem_Z;
  brisbane_mem_create(SIZE * sizeof(float), &mem_X);
  brisbane_mem_create(SIZE * sizeof(float), &mem_Y);
  brisbane_mem_create(SIZE * sizeof(float), &mem_Z);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_X, X);
  brisbane_task_h2d_full(task0, mem_Y, Y);
  void* saxpy_params[4] = { mem_Z, &A, mem_X, mem_Y };
  int saxpy_params_info[4] = { brisbane_w, sizeof(A), brisbane_r, brisbane_r };
  brisbane_task_kernel(task0, "saxpy", 1, NULL, &SIZE, NULL, 4, saxpy_params, saxpy_params_info);
  brisbane_task_d2h_full(task0, mem_Z, Z);
  brisbane_task_submit(task0, TARGET, NULL, 1);

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    //printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");

  }

  brisbane_mem_release(mem_X);
  brisbane_mem_release(mem_Y);
  brisbane_mem_release(mem_Z);

  free(X);
  free(Y);
  free(Z);

  brisbane_task_release(task0);

  brisbane_finalize();

  return 0;
}

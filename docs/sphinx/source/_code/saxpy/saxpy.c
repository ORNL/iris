#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE;
  int TARGET;
  int VERBOSE;
  float *X, *Y, *Z;
  float A = 10;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  VERBOSE = argc > 3 ? atol(argv[3]) : 1;

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

  iris_mem mem_X;
  iris_mem mem_Y;
  iris_mem mem_Z;
  iris_mem_create(SIZE * sizeof(float), &mem_X);
  iris_mem_create(SIZE * sizeof(float), &mem_Y);
  iris_mem_create(SIZE * sizeof(float), &mem_Z);

  iris_task task0;
  iris_task_create(&task0);
  iris_task_h2d_full(task0, mem_X, X);
  iris_task_h2d_full(task0, mem_Y, Y);
  void* saxpy_params[4] = { mem_Z, &A, mem_X, mem_Y };
  int saxpy_params_info[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task0, "saxpy", 1, NULL, &SIZE, NULL, 4, saxpy_params, saxpy_params_info);
  iris_task_d2h_full(task0, mem_Z, Z);
  iris_task_submit(task0, TARGET, NULL, 1);

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");

  }

  iris_mem_release(mem_X);
  iris_mem_release(mem_Y);
  iris_mem_release(mem_Z);

  free(X);
  free(Y);
  free(Z);

  iris_task_release(task0);

  iris_finalize();

  return 0;
}

#include <iris/iris.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  size_t SIZE;
  float *X, *Y, *Z;
  float A = 10;
  int ERROR = 0;

  int nteams = 8;
  int chunk_size = SIZE / nteams;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  X = (float*) malloc(SIZE * sizeof(float));
  Y = (float*) malloc(SIZE * sizeof(float));
  Z = (float*) malloc(SIZE * sizeof(float));

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

  iris::Mem mem_X(SIZE * sizeof(float));
  iris::Mem mem_Y(SIZE * sizeof(float));
  iris::Mem mem_Z(SIZE * sizeof(float));

  iris::Task task;
  task.h2d_full(&mem_X, X);
  task.h2d_full(&mem_Y, Y);
  void* params0[4] = { &mem_Z, &A, &mem_X, &mem_Y };
  int pinfo0[4] = { iris_w, sizeof(A), iris_r, iris_r };
  task.kernel("saxpy", 1, NULL, &SIZE, NULL, 4, params0, pinfo0);
  task.d2h_full(&mem_Z, Z);
  task.submit(1, NULL, 1);

  for (int i = 0; i < SIZE; i++) {
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");

  free(X);
  free(Y);
  free(Z);

  platform.finalize();

  return 0;
}

#include <iris/iris.h>
#include <stdio.h>
#include <malloc.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE = 8;
  float *X, *Y, *Z;

  X = (float*) malloc(SIZE * sizeof(float));
  Y = (float*) malloc(SIZE * sizeof(float));
  Z = (float*) malloc(SIZE * sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    X[i] = i;
    Y[i] = i * 10;
  }

  printf("X [");
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", X[i]);
  printf("]\n");
  printf("Y [");
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Y[i]);
  printf("]\n");

  iris_mem mem_X;
  iris_mem mem_Y;
  iris_mem mem_Z;
  iris_mem_create(SIZE * sizeof(float), &mem_X);
  iris_mem_create(SIZE * sizeof(float), &mem_Y);
  iris_mem_create(SIZE * sizeof(float), &mem_Z);

  iris_task task0;
  iris_task_create(&task0);
  iris_task_h2d_full(task0, mem_X, X);
  void* task0_params[2] = { &mem_Z, &mem_X };
  int task0_params_info[2] = { iris_w, iris_r };
  iris_task_kernel(task0, "kernel0", 1, NULL, &SIZE, NULL, 2, task0_params, task0_params_info);
  iris_task_submit(task0, iris_gpu, NULL, 1);

  iris_task task1;
  iris_task_create(&task1);
  iris_task_h2d_full(task1, mem_Y, Y);
  void* task1_params[2] = { &mem_Z, &mem_Y };
  int task1_params_info[2] = { iris_rw, iris_r };
  iris_task_kernel(task1, "kernel1", 1, NULL, &SIZE, NULL, 2, task1_params, task1_params_info);
  iris_task_d2h_full(task1, mem_Z, Z);
  iris_task_submit(task1, iris_cpu, NULL, 1);

  printf("Z [");
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");

  iris_task_release(task0);
  iris_task_release(task1);

  iris_mem_release(mem_X);
  iris_mem_release(mem_Y);
  iris_mem_release(mem_Z);

  iris_finalize();
  return 0;
}


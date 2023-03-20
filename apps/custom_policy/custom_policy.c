#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE = argc > 1 ? atol(argv[1]) : 8;
  int* A = (int*) malloc(SIZE * sizeof(int));

  iris_mem memA;
  iris_mem_create(SIZE * sizeof(int), &memA);

  iris_register_policy("libPolicyGWS.so", "custom_gws", (void*) 16);

  void* params[1] = { &memA };
  int params_info[1] = { iris_w };
  iris_task task;
  iris_task_create(&task);
  iris_task_kernel(task, "setid", 1, NULL, &SIZE, NULL, 1, params, params_info);
  iris_task_d2h_full(task, memA, A);
  iris_task_submit(task, iris_custom, "custom_gws", 1);

  printf("A[");
  for (int i = 0; i < SIZE; i++) printf("%3d", A[i]);
  printf("]\n");

  iris_finalize();

  return 0;
}


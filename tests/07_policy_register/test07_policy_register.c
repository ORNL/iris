#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  size_t SIZE = argc > 1 ? atol(argv[1]) : 4;

  int* A = (int*) malloc(SIZE * sizeof(int));

  iris_register_policy("libPolicyGWS.so", "policy_gws", (void*) 8);
  iris_register_policy("libPolicyGWSHook.so", "policy_gws_hook", (void*) 8);

  iris_mem memA;
  iris_mem_create(SIZE * sizeof(int), &memA);

  void* params[1] = { &memA };
  int params_info[1] = { iris_w };
  iris_task task;
  iris_task_create(&task);
  iris_task_kernel(task, "process", 1, NULL, &SIZE, NULL, 1, params, params_info);
  iris_task_d2h_full(task, memA, A);
  iris_task_submit(task, iris_custom, "policy_gws_hook", 1);

  for (int i = 0; i < SIZE; i++) printf("[%3d] %8d\n", i, A[i]);

  iris_finalize();

  return iris_error_count();
}


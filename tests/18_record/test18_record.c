#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  size_t SIZE;
  int *A, *B;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) A[i] = i;

  iris_record_start();

  iris_mem mem;
  iris_mem_create(SIZE * sizeof(int), &mem);

  iris_task task1;
  iris_task_create(&task1);

  iris_task_h2d_full(task1, mem, A);
  iris_task_set_name(task1, "named_memory_transfer");
  iris_task_submit(task1, iris_default, NULL, 1);

  void* params[1] = { &mem };
  int params_info[1] = { iris_rw };
  iris_task task2;
  iris_task task2_dep[] = { task1 };
  iris_task_create(&task2);
  iris_task_depend(task2, 1, task2_dep);
  iris_task_kernel(task2, "process", 1, NULL, &SIZE, NULL, 1, params, params_info);
  iris_task_submit(task2, iris_default, NULL, 1);

  iris_task task3;
  iris_task task3_dep[] = { task2 };
  iris_task_create(&task3);
  iris_task_depend(task3, 1, task3_dep);
  iris_task_d2h_full(task3, mem, B);
  iris_task_submit(task3, iris_data, NULL, 1);

  iris_record_stop();

  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  iris_finalize();

  return iris_error_count();
}

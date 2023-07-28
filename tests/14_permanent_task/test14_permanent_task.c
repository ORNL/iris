#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  size_t SIZE;
  int LOOP;
  int *A, *B;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  LOOP = argc > 2 ? atoi(argv[2]) : 10;

  printf("[%s:%d] SIZE[%lu] LOOP[%d]\n", __FILE__, __LINE__, SIZE, LOOP);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) A[i] = i;

  iris_mem mem;
  iris_mem_create(SIZE * sizeof(int), &mem);

  iris_task task1;
  iris_task_create(&task1);

  iris_task_h2d_full(task1, mem, A);
  iris_task_submit(task1, iris_any, NULL, true);

  void* params[1] = { &mem };
  int params_info[1] = { iris_rw };
  iris_task task2;
#if 1
  iris_task_create_perm(&task2);
#else
  iris_task_create(&task2);
  iris_task_retain(task2, true);
#endif
  iris_task_kernel(task2, "process", 1, NULL, &SIZE, NULL, 1, params, params_info);
  for (int i = 0; i < LOOP; i++) iris_task_submit(task2, iris_any, NULL, true);

  iris_task task3;
  iris_task_create(&task3);
  iris_task_d2h_full(task3, mem, B);
  iris_task_submit(task3, iris_any, NULL, true);

  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  iris_finalize();


  printf("Number of errors:%d\n", iris_error_count());
  return iris_error_count();
}


#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

size_t SIZE;
int NTASKS;
int PERCENT;
int *A, *B, *C;

iris_task* tasks;

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  time_t t;
  srand((unsigned int) time(&t));

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  NTASKS = argc > 2 ? atoi(argv[2]) : 10;
  PERCENT = argc > 3 ? atoi(argv[3]) : 100;

  printf("[%s:%d] SIZE[%zu] NTASKS[%d] PERCENT[%d]\n", __FILE__, __LINE__, SIZE, NTASKS, PERCENT);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));
  tasks = (iris_task*) malloc(NTASKS * sizeof(iris_task));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 100;
  }

  printf("[%s:%d] A [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");
  printf("[%s:%d] B [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", B[i]);
  printf("]\n");

  iris_mem mem_A, mem_B, mem_C;
  iris_mem_create(SIZE * sizeof(int), &mem_A);
  iris_mem_create(SIZE * sizeof(int), &mem_B);
  iris_mem_create(SIZE * sizeof(int), &mem_C);

  iris_task task0;
  iris_task_create(&task0);
  iris_task_h2d_full(task0, mem_A, A);
  iris_task_submit(task0, iris_roundrobin, NULL, 1);

  iris_task task1;
  iris_task_create(&task1);
  iris_task_h2d_full(task1, mem_B, B);
  iris_task_submit(task1, iris_roundrobin, NULL, 1);

  for (int i = 0; i < NTASKS; i++) {
    iris_task_create(tasks + i);
    void* params[3] = { mem_C, mem_A, mem_B };
    int params_info[3] = { iris_xw, iris_xr, iris_xr };
    iris_task_kernel(tasks[i], "vecadd", 1, NULL, &SIZE, NULL, 3, params, params_info);
    for (int j = 0; j < i; j++) {
      if (rand() % 100 < PERCENT) iris_task_depend(tasks[i], 1, tasks + j);
    }
    iris_task_submit(tasks[i], iris_roundrobin, NULL, 0);
  }

  iris_synchronize();

  iris_task task2;
  iris_task_create(&task2);
  iris_task_d2h_full(task2, mem_C, C);
  iris_task_submit(task2, iris_roundrobin, NULL, 1);

  printf("[%s:%d] C [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", C[i]);
  printf("]\n");

  iris_mem_release(mem_A);
  iris_mem_release(mem_B);
  iris_mem_release(mem_C);

  iris_finalize();

  free(A);
  free(B);
  free(C);
  free(tasks);

  return 0;
}

#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

size_t SIZE;
int NTASKS;
int PERCENT;
int *A;

iris_task* tasks;

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);
  setenv("IRIS_ARCHS", "cuda", 1);

  time_t t;
  srand((unsigned int) time(&t));

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  NTASKS = argc > 2 ? atoi(argv[2]) : 10;
  PERCENT = argc > 3 ? atoi(argv[3]) : 100;

  printf("[%s:%d] SIZE[%zu] NTASKS[%d] PERCENT[%d]\n", __FILE__, __LINE__, SIZE, NTASKS, PERCENT);

  A = (int*) malloc(SIZE * sizeof(int));
  tasks = (iris_task*) malloc(NTASKS * sizeof(iris_task));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }

  printf("[%s:%d] A [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");

  iris_mem mem_A;
  iris_mem_create(SIZE * sizeof(int), &mem_A);

  iris_task task0;
  iris_task_create_name("task0", &task0);
  iris_task_h2d_full(task0, mem_A, A);
  iris_task_submit(task0, iris_random, NULL, 1);

  for (int i = 0; i < NTASKS; i++) {
    iris_task_create_name("task-i", tasks + i);
    void* params[1] = { &mem_A };
    int params_info[1] = { iris_w };
    iris_task_kernel(tasks[i], "add1", 1, NULL, &SIZE, NULL, 1, params, params_info);
    for (int j = 0; j < i; j++) {
      if (rand() % 100 < PERCENT) iris_task_depend(tasks[i], 1, tasks + j);
    }
    iris_task_depend(tasks[i], 1, &task0);
    iris_task_submit(tasks[i], iris_random, NULL, 0);
  }

  iris_synchronize();

  iris_task task1;
  iris_task_create(&task1);
  iris_task_d2h_full(task1, mem_A, A);
  iris_task_submit(task1, iris_random, NULL, 1);

  printf("[%s:%d] A [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");

  iris_mem_release(mem_A);

  iris_finalize();

  free(A);
  free(tasks);
  printf("IRIS error count:%d\n", iris_error_count());
  return iris_error_count();
}

#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE;
  int *A, *B, *C, *D;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  printf("[%s:%d] SIZE[%zu]\n", __FILE__, __LINE__, SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));
  D = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 10;
    C[i] = 0;
    D[i] = 0;
  }

  printf("A [");
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");
  printf("B [");
  for (int i = 0; i < SIZE; i++) printf(" %4d", B[i]);
  printf("]\n");

  iris_mem mem_A;
  iris_mem mem_B;
  iris_mem mem_C;
  iris_mem mem_D;
  iris_mem_create(SIZE * sizeof(int), &mem_A);
  iris_mem_create(SIZE * sizeof(int), &mem_B);
  iris_mem_create(SIZE * sizeof(int), &mem_C);
  iris_mem_create(SIZE * sizeof(int), &mem_D);

  iris_task task0;
  iris_task task1;
  iris_task task2;
  iris_task task3;

  iris_task_create(&task0);
  iris_task_create(&task1);
  iris_task_create(&task2);
  iris_task_create(&task3);

  iris_task_h2d_full(task0, mem_A, A);
  iris_task_submit(task0, 0, NULL, 1);

  iris_task_h2d_full(task1, mem_B, B);
  iris_task_submit(task1, 1, NULL, 1);

  void* params2[2] = { &mem_C, &mem_B };
  int params_info2[2] = { iris_w, iris_r };
  iris_task_kernel(task2, "copy", 1, NULL, &SIZE, NULL, 2, params2, params_info2);
  iris_task_d2h_full(task2, mem_C, C);
  iris_task_submit(task2, 0, NULL, 0);

  void* params3[2] = { &mem_D, &mem_A };
  int params_info3[2] = { iris_w, iris_r };
  iris_task_kernel(task3, "copy", 1, NULL, &SIZE, NULL, 2, params3, params_info3);
  iris_task_d2h_full(task3, mem_D, D);
  iris_task_submit(task3, 1, NULL, 0);

  iris_synchronize();

  printf("C [");
  for (int i = 0; i < SIZE; i++) printf(" %4d", C[i]);
  printf("]\n");
  printf("D [");
  for (int i = 0; i < SIZE; i++) printf(" %4d", D[i]);
  printf("]\n");

  iris_finalize();

  free(A);

  printf("IRIS error count:%d\n", iris_error_count());
  return iris_error_count();
}

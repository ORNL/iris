#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *SRC, *SINK;

  iris_init(&argc, &argv, 1);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%lu]\n", SIZE);

  int* AB = (int*) malloc(SIZE * sizeof(int));
  int* BC = (int*) malloc(SIZE * sizeof(int));

  iris_mem mem_AB;
  iris_mem mem_BC;

  iris_mem_create(SIZE * sizeof(int), &mem_AB);
  iris_mem_create(SIZE * sizeof(int), &mem_BC);

  iris_kernel kernel_A;
  iris_kernel_create("kernel_A", &kernel_A);
  iris_kernel_setmem(kernel_A, 0, mem_AB, iris_w);

  iris_kernel kernel_B;
  iris_kernel_create("kernel_B", &kernel_B);
  iris_kernel_setmem(kernel_B, 0, mem_AB, iris_r);
  iris_kernel_setmem(kernel_B, 1, mem_BC, iris_w);

  iris_kernel kernel_C;
  iris_kernel_create("kernel_C", &kernel_C);
  iris_kernel_setmem(kernel_C, 0, mem_BC, iris_r);

  iris_task task_A;
  iris_task_create(&task_A);
  iris_task_h2d_full(task_A, mem_AB, AB);
  iris_task_kernel_object(task_A, kernel_A, 1, NULL, &SIZE, NULL);

  iris_task task_B;
  iris_task_create(&task_B);
  iris_task_kernel_object(task_B, kernel_B, 1, NULL, &SIZE, NULL);
  iris_task_depend(task_B, 1, &task_A);

  iris_task task_C;
  iris_task_create(&task_C);
  iris_task_kernel_object(task_C, kernel_C, 1, NULL, &SIZE, NULL);
  iris_task_d2h_full(task_C, mem_BC, BC);
  iris_task_depend(task_C, 1, &task_B);

  iris_task_submit(task_A, iris_cpu, NULL, false);
  iris_task_submit(task_B, iris_cpu, NULL, false);
  iris_task_submit(task_C, iris_cpu, NULL, false);

  iris_synchronize();

  for (int i = 0; i < SIZE; i++) {
    printf("[%3d] %10d\n", i, BC[i]);
  }

  iris_finalize();

  return 0;
}

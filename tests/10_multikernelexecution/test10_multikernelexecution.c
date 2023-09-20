#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int LOOP;
  double t0, t1;

  iris_init(&argc, &argv, 1);

  SIZE = argc > 1 ? atol(argv[1]) : 128;
  LOOP = argc > 2 ? atoi(argv[2]) : 100;
  printf("SIZE[%lu] LOOP[%d]\n", SIZE, LOOP);

  iris_mem mem0, mem1, mem2, mem3;
  iris_mem_create(SIZE * sizeof(int), &mem0);
  iris_mem_create(SIZE * sizeof(int), &mem1);
  iris_mem_create(SIZE * sizeof(int), &mem2);
  iris_mem_create(SIZE * sizeof(int), &mem3);

  iris_kernel kernel0, kernel1, kernel2, kernel3;
  iris_kernel_create("kernel0", &kernel0);
  iris_kernel_create("kernel1", &kernel1);
  iris_kernel_create("kernel2", &kernel2);
  iris_kernel_create("kernel3", &kernel3);

  iris_kernel_setmem(kernel0, 0, mem0, iris_w);
  iris_kernel_setmem(kernel1, 0, mem1, iris_w);
  iris_kernel_setmem(kernel2, 0, mem2, iris_w);
  iris_kernel_setmem(kernel3, 0, mem3, iris_w);

  iris_kernel_setarg(kernel0, 1, sizeof(int), &LOOP);
  iris_kernel_setarg(kernel1, 1, sizeof(int), &LOOP);
  iris_kernel_setarg(kernel2, 1, sizeof(int), &LOOP);
  iris_kernel_setarg(kernel3, 1, sizeof(int), &LOOP);

  iris_task task0, task1, task2, task3;
  iris_task_create(&task0);
  iris_task_create(&task1);
  iris_task_create(&task2);
  iris_task_create(&task3);
  iris_task_kernel_object(task0, kernel0, 1, NULL, &SIZE, NULL);
  iris_task_kernel_object(task1, kernel1, 1, NULL, &SIZE, NULL);
  iris_task_kernel_object(task2, kernel2, 1, NULL, &SIZE, NULL);
  iris_task_kernel_object(task3, kernel3, 1, NULL, &SIZE, NULL);

  iris_task_submit(task0, iris_sdq, NULL, false);
  iris_task_submit(task1, iris_sdq, NULL, false);
  iris_task_submit(task2, iris_sdq, NULL, false);
  iris_task_submit(task3, iris_sdq, NULL, false);

  printf("IRIS Synchronize\n");
  iris_synchronize();

  printf("IRIS Finalize\n");
  iris_finalize();
  printf("Number of errors:%d\n", iris_error_count());
  return iris_error_count();
}

#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE, nbytes;
  double *A, *B, *C;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  nbytes = SIZE * sizeof(double);

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  iris_mem memA, memB, memC;
  iris_mem_create(nbytes, &memA);
  iris_mem_create(nbytes, &memB);
  iris_mem_create(nbytes, &memC);

  iris_task task;
  iris_task_create(&task);
  iris_task_malloc(task, memA);
  iris_task_malloc(task, memB);
  iris_task_malloc(task, memC);
  iris_task_submit(task, 0, NULL, 1);

  iris_mem_arch(memA, 0, (void**) &A);
  iris_mem_arch(memB, 0, (void**) &B);
  iris_mem_arch(memC, 0, (void**) &C);

  printf("[%s:%d] A[%p] B[%p] C[%p]\n", __FILE__, __LINE__, A, B, C);

  iris_mem_release(memA);
  iris_mem_release(memB);
  iris_mem_release(memC);

  iris_finalize();

  return iris_error_count();
}

#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int ndevs;
  int *A, *B, *C, *D, *E;
  int ERROR = 0;

  iris_init(&argc, &argv, true);

  iris_device_count(&ndevs);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  ndevs= argc > 2 ? atoi(argv[2]) : ndevs;

  printf("SIZE[%lu] ndevs[%d]\n", SIZE, ndevs);

  A = (int*) valloc(SIZE * ndevs * sizeof(int));
  B = (int*) valloc(SIZE * ndevs * sizeof(int));
  C = (int*) valloc(SIZE * ndevs * sizeof(int));

  for (int i = 0; i < SIZE * ndevs; i++) {
    A[i] = i;
    B[i] = i * 1000;
  }

  iris_mem mem_A;
  iris_mem mem_B;
  iris_mem mem_C;
  iris_mem_create(SIZE * ndevs * sizeof(int), &mem_A);
  iris_mem_create(SIZE * ndevs * sizeof(int), &mem_B);
  iris_mem_create(SIZE * ndevs * sizeof(int), &mem_C);

  for (int i = 0; i < ndevs; i++) {
    iris_task task;
    iris_task_create(&task);
    iris_task_h2d(task, mem_A, SIZE * i * sizeof(int), SIZE * sizeof(int), A + SIZE * i);
    iris_task_h2d(task, mem_B, SIZE * i * sizeof(int), SIZE * sizeof(int), B + SIZE * i);
    int off_x = SIZE * i;
    void* params[4] = { mem_C, mem_A, mem_B, &off_x };
    int pinfo[4] = { iris_w, iris_r, iris_r, sizeof(int) };
    size_t memranges[8] = {
      i * SIZE * sizeof(int), SIZE * sizeof(int),
      i * SIZE * sizeof(int), SIZE * sizeof(int),
      i * SIZE * sizeof(int), SIZE * sizeof(int),
      0, 0
    };
    iris_task_kernel_v3(task, "vecadd", 1, NULL, &SIZE, NULL, 4, params, NULL, pinfo, memranges);
    iris_task_d2h(task, mem_C, SIZE * i * sizeof(int), SIZE * sizeof(int), C + SIZE * i);
    iris_task_submit(task, i, NULL, 0);
    iris_task_release(task);
  }

  iris_synchronize();

  for (int i = 0; i < SIZE * ndevs; i++) {
    printf("[%8d] %8d = %8d + %8d\n", i, C[i], A[i], B[i]);
    if (C[i] != A[i] + B[i]) ERROR++;
  }
  printf("CORRECT[%d] ERROR[%d]\n", SIZE * ndevs - ERROR, ERROR);

  iris_mem_release(mem_A);
  iris_mem_release(mem_B);
  iris_mem_release(mem_C);

  free(A);
  free(B);
  free(C);

  iris_finalize();

  return 0;
}

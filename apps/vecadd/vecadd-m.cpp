#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int ndevs;
  int *A, *B, *C, *D, *E;
  int ERROR = 0;

  brisbane_init(&argc, &argv, true);

  brisbane_device_count(&ndevs);

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

  brisbane_mem mem_A;
  brisbane_mem mem_B;
  brisbane_mem mem_C;
  brisbane_mem_create(SIZE * ndevs * sizeof(int), &mem_A);
  brisbane_mem_create(SIZE * ndevs * sizeof(int), &mem_B);
  brisbane_mem_create(SIZE * ndevs * sizeof(int), &mem_C);

  for (int i = 0; i < ndevs; i++) {
    brisbane_task task;
    brisbane_task_create(&task);
    brisbane_task_h2d(task, mem_A, SIZE * i * sizeof(int), SIZE * sizeof(int), A + SIZE * i);
    brisbane_task_h2d(task, mem_B, SIZE * i * sizeof(int), SIZE * sizeof(int), B + SIZE * i);
    int off_x = SIZE * i;
    void* params[4] = { mem_C, mem_A, mem_B, &off_x };
    int pinfo[4] = { brisbane_w, brisbane_r, brisbane_r, sizeof(int) };
    size_t memranges[8] = {
      i * SIZE * sizeof(int), SIZE * sizeof(int),
      i * SIZE * sizeof(int), SIZE * sizeof(int),
      i * SIZE * sizeof(int), SIZE * sizeof(int),
      0, 0
    };
    brisbane_task_kernel_v3(task, "vecadd", 1, NULL, &SIZE, NULL, 4, params, NULL, pinfo, memranges);
    brisbane_task_d2h(task, mem_C, SIZE * i * sizeof(int), SIZE * sizeof(int), C + SIZE * i);
    brisbane_task_submit(task, i, NULL, 0);
    brisbane_task_release(task);
  }

  brisbane_synchronize();

  for (int i = 0; i < SIZE * ndevs; i++) {
    printf("[%8d] %8d = %8d + %8d\n", i, C[i], A[i], B[i]);
    if (C[i] != A[i] + B[i]) ERROR++;
  }
  printf("CORRECT[%d] ERROR[%d]\n", SIZE * ndevs - ERROR, ERROR);

  brisbane_mem_release(mem_A);
  brisbane_mem_release(mem_B);
  brisbane_mem_release(mem_C);

  free(A);
  free(B);
  free(C);

  brisbane_finalize();

  return 0;
}

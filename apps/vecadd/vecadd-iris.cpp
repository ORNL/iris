#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B, *C, *D, *E;
  int ERROR = 0;

  iris_init(&argc, &argv, true);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%lu]\n", SIZE);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));
  C = (int*) valloc(SIZE * sizeof(int));
  D = (int*) valloc(SIZE * sizeof(int));
  E = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
  }

  iris_mem mem_A;
  iris_mem mem_B;
  iris_mem mem_C;
  iris_mem_create(SIZE * sizeof(int), &mem_A);
  iris_mem_create(SIZE * sizeof(int), &mem_B);
  iris_mem_create(SIZE * sizeof(int), &mem_C);

  iris_task task0;
  iris_task_create(&task0);
  iris_task_h2d_full(task0, mem_A, A);
  iris_task_h2d_full(task0, mem_B, B);
  void* params0[3] = { mem_C, mem_A, mem_B };
  int pinfo0[3] = { iris_w, iris_r, iris_r };
  iris_task_kernel(task0, "loop0", 1, NULL, &SIZE, NULL, 3, params0, pinfo0);
  iris_task_submit(task0, iris_gpu, NULL, true);
/*
#pragma acc parallel loop copyin(A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma omp target teams distribute parallel for map(to:A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma brisbane kernel h2d(A[0:SIZE], B[0:SIZE]) alloc(C[0:SIZE]) device(gpu)
  for (int i = 0; i < SIZE; i++) {
    C[i] = A[i] + B[i];
  }
*/

  iris_mem mem_D;
  iris_mem_create(SIZE * sizeof(int), &mem_D);

  iris_task task1;
  iris_task_create(&task1);
  void* params1[2] = { mem_D, mem_C };
  int pinfo1[2] = { iris_w, iris_r };
  iris_task_kernel(task1, "loop1", 1, NULL, &SIZE, NULL, 2, params1, pinfo1);

  iris_task_submit(task1, iris_gpu, NULL, true);
/*
  #pragma acc parallel loop present(C[0:SIZE]) device(cpu)
  #pragma omp target teams distribute parallel for device(cpu)
  #pragma brisbane kernel present(C[0:SIZE]) device(cpu)
  for (int i = 0; i < SIZE; i++) {
    D[i] = C[i] * 10;
  }
*/

  iris_mem mem_E;
  iris_mem_create(SIZE * sizeof(int), &mem_E);

  iris_task task2;
  iris_task_create(&task2);
  void* params2[2] = { mem_E, mem_D };
  int pinfo2[2] = { iris_w, iris_r };
  iris_task_kernel(task2, "loop2", 1, NULL, &SIZE, NULL, 2, params2, pinfo2);
  iris_task_d2h_full(task2, mem_E, E);
  iris_task_submit(task2, iris_data, NULL, true);

/*
#pragma acc parallel loop present(D[0:SIZE]) device(data)
#pragma omp target teams distribute parallel for map(from:E[0:SIZE]) device(data)
#pragma brisbane kernel d2h(E[0:SIZE]) present(D[0:SIZE]) device(data)
  for (int i = 0; i < SIZE; i++) {
    E[i] = D[i] * 2;
  }
*/

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] %8d = (%8d + %8d) * %d\n", i, E[i], A[i], B[i], 20);
    if (E[i] != (A[i] + B[i]) * 20) ERROR++;
  }
  printf("ERROR[%d]\n", ERROR);

  iris_task_release(task0);
  iris_task_release(task1);
  iris_task_release(task2);
  iris_mem_release(mem_A);
  iris_mem_release(mem_B);
  iris_mem_release(mem_C);
  iris_mem_release(mem_D);
  iris_mem_release(mem_E);

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);

  iris_finalize();

  return 0;
}

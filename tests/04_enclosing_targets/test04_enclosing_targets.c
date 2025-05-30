#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  printf("Inside main\n");
  size_t SIZE;
  int *A, *B;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("INIT SIZE[%d]\n", SIZE);
  //setenv("IRIS_ARCHS", "opencl", 1);
  iris_init(&argc, &argv, true);

  printf("SIZE[%d]\n", SIZE);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 1000;
  }

  iris_mem mem_A;
  iris_mem mem_B;
  /*
  iris_mem_map(mem_A, A, SIZE * sizeof(int));
  iris_mem_map(mem_B, B, SIZE * sizeof(int));
  */
  iris_mem_create(SIZE * sizeof(int), &mem_A);
  iris_mem_create(SIZE * sizeof(int), &mem_B);
#pragma omp target data map(A, B)
  {


  iris_task task0;
  iris_task_create(&task0);
  iris_task_h2d_full(task0, mem_A, A);
  size_t kernel_loop0_off[1] = { 0 };
  size_t kernel_loop0_idx[1] = { SIZE };
  void* loop0_params[1] = { &mem_A };
  int loop0_params_info[1] = { iris_rw };
  iris_task_kernel(task0, "loop0", 1, kernel_loop0_off, kernel_loop0_idx, NULL, 1, loop0_params, loop0_params_info);
  iris_task_submit(task0, iris_default, NULL, true);
  iris_task_release(task0);
#if 0
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
    A[i] *= 2;
  }
#endif

  iris_task task1;
  iris_task_create(&task1);
  iris_task_retain(task1, true);
  iris_task_h2d_full(task1, mem_B, B);
  void* loop1_params[2] = { &mem_B, &mem_A };
  int loop1_params_info[2] = { iris_rw, iris_r };
  size_t kernel_loop1_off[1] = { 0 };
  size_t kernel_loop1_idx[1] = { SIZE };
  iris_task_kernel(task1, "loop1", 1, kernel_loop1_off, kernel_loop1_idx, NULL, 2, loop1_params, loop1_params_info);
  iris_task_submit(task1, iris_depend, NULL, true);
  iris_task_d2h_full(task1, mem_A, A);
  iris_task_d2h_full(task1, mem_B, B);
  iris_task_submit(task1, iris_depend, NULL, true);
  iris_task_release(task1);
  
  iris_synchronize();
#if 0
#pragma omp parallel for
  for (int i = 0; i < SIZE; i++) {
    B[i] += A[i];
  }
#endif

  for (int i = 0; i < SIZE; i++) {
    printf("[%8d] A[%8d] B[%8d]\n", i, A[i], B[i]);
  }
  }

  iris_mem_release(mem_A);
  iris_mem_release(mem_B);

  free(A);
  free(B);

  iris_finalize();
  return iris_error_count();
}

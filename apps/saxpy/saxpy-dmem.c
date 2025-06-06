#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <sys/mman.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE;
  int TARGET;
  int VERBOSE;
  float *X, *Y, *Z;
  float A = 10;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : iris_default;
  VERBOSE = argc > 3 ? atol(argv[3]) : 1;

  printf("[%s:%d] SIZE[%zu] TARGET[%d] VERBOSE[%d]\n", __FILE__, __LINE__, SIZE, TARGET, VERBOSE);

  int target = iris_default;
  if (TARGET == 0) target = iris_cpu;
  else if (TARGET == 1) target = iris_cuda;
  else if (TARGET == 2) target = iris_hip;
  else target= TARGET;
  size_t alignment = 4096;
  int result;
  size_t size = SIZE*sizeof(float);
  posix_memalign(&X, alignment, SIZE*sizeof(float));
  posix_memalign(&Y, alignment, SIZE*sizeof(float));
  posix_memalign(&Z, alignment, SIZE*sizeof(float));
  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    X[i] = i;
    Y[i] = i;
  }

  printf("X [");
  for (int i = 0; i < SIZE; i++) printf(" %2.0f.", X[i]);
  printf("]\n");
  printf("Y [");
  for (int i = 0; i < SIZE; i++) printf(" %2.0f.", Y[i]);
  printf("]\n");

  }

  clock_t start = clock();
#if 1
  if (mlock(X, size) != 0) {
      perror("mlock failed");
      free(X);
      return 1;
  }
  if (mlock(Y, size) != 0) {
      perror("mlock failed");
      free(Y);
      return 1;
  }
  if (mlock(Z, size) != 0) {
      perror("mlock failed");
      free(Z);
      return 1;
  }
#endif

  iris_mem mem_X;
  iris_mem mem_Y;
  iris_mem mem_Z;
  iris_data_mem_create(&mem_X, X, SIZE * sizeof(float));
  iris_data_mem_create(&mem_Y, Y, SIZE * sizeof(float));
  iris_data_mem_create(&mem_Z, Z, SIZE * sizeof(float));

  iris_task task0;
  iris_task_create(&task0);
  void* saxpy_params[4]      = { &mem_Z, &A,        &mem_X, &mem_Y };
  int   saxpy_params_info[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task0, "saxpy", 1, NULL, &SIZE, NULL, 4, saxpy_params, saxpy_params_info);
  iris_task_dmem_flush_out(task0, mem_Z);
  iris_task_submit(task0, target, NULL, 1);
  clock_t end = clock();
  double     cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time spent in the for loop: %f seconds\n", cpu_time_used);

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    //printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");

  }

  iris_mem_release(mem_X);
  iris_mem_release(mem_Y);
  iris_mem_release(mem_Z);

  free(X);
  free(Y);
  free(Z);

  //iris_task_release(task0);

  iris_finalize();

  return 0;
}

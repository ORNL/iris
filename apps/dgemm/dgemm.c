#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

size_t SIZE, UNIT;
int VERBOSE = 1;
int TARGET;
double *A, *B, *C;
double t0, t1, t2, t3;

void ijk() {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      double sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;
    }
  }
}

void kij() {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      double a = A[i * SIZE + k];
      for (int j = 0; j < SIZE; j++) {
        C[i * SIZE + j] += a * B[k * SIZE + j];
      }
    }
  }
}

int main(int argc, char** argv) {
  int ERROR = 0;

  iris_init(&argc, &argv, 1);

  iris_timer_now(&t0);

  SIZE = argc > 1 ? atol(argv[1]) : 64;
  TARGET = argc > 2 ? atoi(argv[2]) : 0;
  VERBOSE = argc > 3 ? atol(argv[3]) : 1;

  int target_dev = TARGET == 0 ? iris_cpu : TARGET == 1 ? iris_gpu : TARGET == 2 ? iris_dsp : TARGET == 3 ? iris_nvidia : TARGET == 4 ? iris_amd : iris_fpga;
  printf("SIZE[%d] MATRIX_SIZE[%u]MB VERBOSE[%d] TARGET[%d]\n", SIZE, SIZE * SIZE * sizeof(double) / 1024 / 1024, VERBOSE, TARGET);

  A = (double*) malloc(SIZE * SIZE * sizeof(double));
  B = (double*) malloc(SIZE * SIZE * sizeof(double));
  C = (double*) malloc(SIZE * SIZE * sizeof(double));

  for (int i = 0; i < SIZE * SIZE; i++) {
    A[i] = (double)i+1;
    B[i] = (double)((i+1) * 10);
    C[i] = (double)0;
  }

  iris_mem mem_A;
  iris_mem mem_B;
  iris_mem mem_C;
  iris_mem_create(SIZE * SIZE * sizeof(double), &mem_A);
  iris_mem_create(SIZE * SIZE * sizeof(double), &mem_B);
  iris_mem_create(SIZE * SIZE * sizeof(double), &mem_C);

  iris_timer_now(&t1);

  iris_task task;
  iris_task_create(&task);
  iris_task_h2d(task, mem_A, 0, SIZE * SIZE * sizeof(double), A);
  iris_task_h2d(task, mem_B, 0, SIZE * SIZE * sizeof(double), B);
  size_t ijk_idx[2] = { SIZE, SIZE };
  size_t lws_size = (SIZE > 16 ) ? 16 : SIZE;
  size_t ijk_lws[2] = { lws_size, lws_size };
  void* params[3] = { &mem_C, &mem_A, &mem_B };
  int pinfo[3] = { iris_w, iris_r, iris_r };
  iris_task_kernel(task, "ijk", 2, NULL, ijk_idx, ijk_lws, 3, params, pinfo);
  iris_task_d2h(task, mem_C, 0, SIZE * SIZE * sizeof(double), C);
  iris_task_submit(task, target_dev, NULL, 1);

  iris_timer_now(&t2);

  if (VERBOSE) {

  int print_size = (SIZE > 8) ? 8: SIZE;
  printf("[[ A ]]\n");
  for (int i = 0; i < print_size; i++) {
    for (int j = 0; j < print_size; j++) {
      printf("%5.0lf ", A[i * SIZE + j]);
    }
    printf("\n");
  }

  printf("[[ B ]]\n");
  for (int i = 0; i < print_size; i++) {
    for (int j = 0; j < print_size; j++) {
      printf("%5.0lf ", B[i * SIZE + j]);
    }
    printf("\n");
  }

  printf("[[ C ]]\n");
  for (int i = 0; i < print_size; i++) {
    for (int j = 0; j < print_size; j++) {
      printf("%5.0lf ", C[i * SIZE + j]);
    }
    printf("\n");
  }

  int error_check = 0;
  if (error_check) {
  printf("Checking errors\n");
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      double sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      if (sum != C[i * SIZE + j]) ERROR++;
    }
  }
  }

  }

  iris_timer_now(&t3);

  printf("ERROR[%d] TIME[%lf,%lf]\n", ERROR, t3 - t0, t2 - t1);

  iris_task_release(task);
  iris_mem_release(mem_A);
  iris_mem_release(mem_B);
  iris_mem_release(mem_C);

  free(A);
  free(B);
  free(C);

  iris_finalize();

  return 0;
}

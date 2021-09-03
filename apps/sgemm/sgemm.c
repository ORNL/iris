#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

size_t SIZE, UNIT;
int VERBOSE;
int TARGET;
float *A, *B, *C;
double t0, t1, t2, t3;

void ijk() {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      float sum = 0.0;
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
      float a = A[i * SIZE + k];
      for (int j = 0; j < SIZE; j++) {
        C[i * SIZE + j] += a * B[k * SIZE + j];
      }
    }
  }
}

int main(int argc, char** argv) {
  int ERROR = 0;

  brisbane_init(&argc, &argv, 1);

  brisbane_timer_now(&t0);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  TARGET = argc > 2 ? atoi(argv[2]) : 0;
  VERBOSE = argc > 3 ? atol(argv[3]) : 0;

  printf("SIZE[%d] MATRIX_SIZE[%u]MB VERBOSE[%d] TARGET[%d]\n", SIZE, SIZE * SIZE * sizeof(float) / 1024 / 1024, VERBOSE, TARGET);

  A = (float*) malloc(SIZE * SIZE * sizeof(float));
  B = (float*) malloc(SIZE * SIZE * sizeof(float));
  C = (float*) malloc(SIZE * SIZE * sizeof(float));

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      A[i * SIZE + j] = i + j;
      B[i * SIZE + j] = i * j;
      C[i * SIZE + j] = 0.0;
    }
  }

  }

  brisbane_mem mem_A;
  brisbane_mem mem_B;
  brisbane_mem mem_C;
  brisbane_mem_create(SIZE * SIZE * sizeof(float), &mem_A);
  brisbane_mem_create(SIZE * SIZE * sizeof(float), &mem_B);
  brisbane_mem_create(SIZE * SIZE * sizeof(float), &mem_C);

  brisbane_timer_now(&t1);

  brisbane_task task;
  brisbane_task_create(&task);
  brisbane_task_h2d(task, mem_A, 0, SIZE * SIZE * sizeof(float), A);
  brisbane_task_h2d(task, mem_B, 0, SIZE * SIZE * sizeof(float), B);
  size_t ijk_idx[2] = { SIZE, SIZE };
  size_t ijk_lws[2] = { 32, 32 };
  void* params[3] = { mem_C, mem_A, mem_B };
  int pinfo[3] = { brisbane_w, brisbane_r, brisbane_r };
  brisbane_task_kernel(task, "ijk", 2, NULL, ijk_idx, ijk_lws, 3, params, pinfo);
  brisbane_task_d2h(task, mem_C, 0, SIZE * SIZE * sizeof(float), C);
  brisbane_task_submit(task, TARGET, NULL, 1);

  brisbane_timer_now(&t2);

  if (VERBOSE) {

  printf("[[ A ]]\n");
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      printf("%5.0lf ", A[i * SIZE + j]);
    }
    printf("\n");
  }

  printf("[[ B ]]\n");
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      printf("%5.0lf ", B[i * SIZE + j]);
    }
    printf("\n");
  }

  printf("[[ C ]]\n");
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      printf("%5.0lf ", C[i * SIZE + j]);
    }
    printf("\n");
  }

  printf("Checking errors\n");
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      float sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      if (sum != C[i * SIZE + j]) ERROR++;
    }
  }

  }

  brisbane_timer_now(&t3);

  printf("ERROR[%d] TIME[%lf,%lf]\n", ERROR, t3 - t0, t2 - t1);

  brisbane_task_release(task);
  brisbane_mem_release(mem_A);
  brisbane_mem_release(mem_B);
  brisbane_mem_release(mem_C);

  free(A);
  free(B);
  free(C);

  brisbane_finalize();

  return 0;
}

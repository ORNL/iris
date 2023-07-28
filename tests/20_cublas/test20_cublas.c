#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda.h>

typedef struct {
  size_t SIZE;
  iris_mem memA, memB, memC;
} cublas0_params;

int cublas0(void* param, const int* dev) {
  cublas0_params* p = (cublas0_params*) param;
  size_t SIZE = p->SIZE;
  iris_mem memA = p->memA;
  iris_mem memB = p->memB;
  iris_mem memC = p->memC;
  double alpha = 1.0;
  double beta = 0.0;

  int dev_type;
  char dev_name[256];

  iris_device_info(*dev, iris_type, &dev_type, NULL);
  iris_device_info(*dev, iris_name, dev_name, NULL);

  double *A, *B, *C;
  iris_mem_arch(memA, *dev, (void**) &A);
  iris_mem_arch(memB, *dev, (void**) &B);
  iris_mem_arch(memC, *dev, (void**) &C);

  printf("[%s:%d] dev[%d:0x%x:%s] SIZE[%zu] A[%p] B[%p] C[%p]\n", __FILE__, __LINE__, *dev, dev_type, dev_name, SIZE, A, B, C);

  printf("[%s:%d] CUBLAS_STATUS_SUCCESS[%d] CUBLAS_STATUS_NOT_INITIALIZED[%d]\n", __FILE__, __LINE__, CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED);

  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  printf("[%s:%d] %d\n", __FILE__, __LINE__, stat);

  stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE, SIZE, &alpha, A, SIZE, B, SIZE, &beta, C, SIZE);
  printf("[%s:%d] %d\n", __FILE__, __LINE__, stat);

  stat = cublasDestroy(handle);
  printf("[%s:%d] %d\n", __FILE__, __LINE__, stat);

  return IRIS_SUCCESS;
}

int main(int argc, char** argv) {
  //setenv("IRIS_ARCHS", "cuda", 1);
  iris_init(&argc, &argv, 1);

  int ndevs;
  bool is_nvidia_device=0;
  iris_device_count(&ndevs);
  for (int d = 0; d < ndevs; d++){
    int backend_worker;
    iris_device_info(d, iris_backend, &backend_worker, NULL);
    if (backend_worker == iris_cuda) is_nvidia_device = 1;
  }
  if(!is_nvidia_device){
    printf("Skipping this test because it is only designed to test NVIDIA GPUs with CUDA.\n");
    return 0;
  }

  size_t SIZE, nbytes;
  double *A, *B, *C;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  nbytes = SIZE * SIZE * sizeof(double);

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  A = (double*) malloc(nbytes);
  B = (double*) malloc(nbytes);
  C = (double*) malloc(nbytes);

  for (int i = 0; i < SIZE * SIZE; i++) {
    A[i] = i;
    B[i] = i * 10;
    C[i] = 0;
  }

  iris_mem memA, memB, memC;
  iris_mem_create(nbytes, &memA);
  iris_mem_create(nbytes, &memB);
  iris_mem_create(nbytes, &memC);

  iris_task task0;
  iris_task_create_name("h2d", &task0);
  iris_task_h2d_full(task0, memA, A);
  iris_task_h2d_full(task0, memB, B);
  iris_task_h2d_full(task0, memC, C);
  iris_task_submit(task0, iris_nvidia, NULL, 1);

  cublas0_params task1_params;
  task1_params.SIZE = SIZE;
  task1_params.memA = memA;
  task1_params.memB = memB;
  task1_params.memC = memC;

  iris_task task1;
  iris_task_create_name("cublas0_task", &task1);
  iris_task_host(task1, cublas0, &task1_params);
  iris_task_depend(task1, 1, &task0);
  iris_task_submit(task1, iris_nvidia, NULL, 1);

  iris_task task9;
  iris_task_create_name("d2h", &task9);
  iris_task_d2h_full(task9, memC, C);
  iris_task_depend(task9, 1, &task1);
  iris_task_submit(task9, iris_nvidia, NULL, 1);

  for (int i = 0; i < SIZE * SIZE; i++) {
    printf("%10.1lf", C[i]);
    if (i % SIZE == SIZE - 1) printf("\n");
  }

  iris_mem_release(memA);
  iris_mem_release(memB);
  iris_mem_release(memC);

  iris_finalize();

  printf("Errors:%d\n", iris_error_count());
  return iris_error_count();
}

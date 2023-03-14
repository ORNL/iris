#include <iris/iris.h>
#include <iris/rt/Command.h>
#include <iris/rt/Device.h>
#include <iris/rt/Kernel.h>
#include <iris/rt/Task.h>
#include <stdio.h>
#include <stdlib.h>

int my_kernel_selector(iris_task task, void* params, char* kernel_name) {
  size_t threshold = *((size_t*) params);
  iris::rt::Task* t = task.class_obj;
  iris::rt::Device* d = t->dev();
  iris::rt::Command* c = t->cmd_kernel();
  iris::rt::Kernel* k = c->kernel();
  size_t ws = c->ws();
  char* name = k->name();
  printf("[%s:%d] kernel[%s] ws[%lu] threshold[%lu]\n", __FILE__, __LINE__, name, ws, threshold);
  if (d->model() == iris_opencl && ws > threshold) {
    sprintf(kernel_name, "%s_v2", name);
    printf("[%s:%d] new kernel[%s]\n", __FILE__, __LINE__, kernel_name);
  }
  return IRIS_SUCCESS;
}

int main(int argc, char** argv) {
  iris_env_set("ARCHS", "opencl:cuda");
  iris_env_set("KERNEL_CUDA",   "kernel.ptx");
  iris_env_set("KERNEL_OPENCL", "kernel-negative.cl");
  iris_env_set("KERNEL_CUDA",   "kernel-negative.ptx");

  iris_init(&argc, &argv, 1);

  size_t SIZE;
  int TARGET;
  int* A;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : 0;

  printf("[%s:%d] SIZE[%zu] TARGET[%d]\n", __FILE__, __LINE__, SIZE, TARGET);

  A = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }

  printf("A [");
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");

  iris_mem mem_A;
  iris_mem_create(SIZE * sizeof(int), &mem_A);

  iris_task task;
  iris_task_create(&task);
  iris_task_h2d_full(task, mem_A, A);
  void* params[1] = { &mem_A };
  int params_info[1] = { iris_rw };
  size_t threshold = 16;
  iris_task_kernel(task, "add1", 1, NULL, &SIZE, NULL, 1, params, params_info);
  iris_task_kernel_selector(task, my_kernel_selector, &threshold, sizeof(threshold));
  iris_task_d2h_full(task, mem_A, A);
  iris_task_submit(task, TARGET, NULL, 1);

  printf("A [");
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");

  iris_mem_release(mem_A);
  iris_task_release(task);

  iris_finalize();

  free(A);

  return iris_error_count();
}


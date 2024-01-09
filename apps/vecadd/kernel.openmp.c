#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global int * A;
  __global int * B;
  __global int * C;
} iris_openmp_vecadd_args;
iris_openmp_vecadd_args vecadd_args;

static int iris_openmp_vecadd_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_vecadd_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: vecadd_args.A = (__global int *) mem; break;
    case 1: vecadd_args.B = (__global int *) mem; break;
    case 2: vecadd_args.C = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

typedef struct {
  __global int * A;
  __global int * B;
  __global int * C;
} iris_openmp_kernel1_args;
iris_openmp_kernel1_args kernel1_args;

static int iris_openmp_kernel1_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel1_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel1_args.A = (__global int *) mem; break;
    case 1: kernel1_args.B = (__global int *) mem; break;
    case 2: kernel1_args.C = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

#include "kernel.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "vecadd") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "empty") == 0) {
    iris_openmp_kernel_idx = 1;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_vecadd_setarg(idx, size, value);
    case 1: return iris_openmp_kernel1_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_vecadd_setmem(idx, mem);
    case 1: return iris_openmp_kernel1_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t *off, size_t *ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: vecadd(vecadd_args.A, vecadd_args.B, vecadd_args.C, off, ndr); break;
    case 1: empty(kernel1_args.A, kernel1_args.B, kernel1_args.C, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


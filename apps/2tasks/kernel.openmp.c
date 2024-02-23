#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float * dst;
  __global float * src;
} iris_openmp_kernel0_args;
iris_openmp_kernel0_args kernel0_args;

static int iris_openmp_kernel0_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel0_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel0_args.dst = (__global float *) mem; break;
    case 1: kernel0_args.src = (__global float *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}
typedef struct {
  __global float * dst;
  __global float * src;
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
    case 0: kernel1_args.dst = (__global float *) mem; break;
    case 1: kernel1_args.src = (__global float *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

#include "kernel.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "kernel0") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "kernel1") == 0) {
    iris_openmp_kernel_idx = 1;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_kernel0_setarg(idx, size, value);
    case 1: return iris_openmp_kernel1_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_kernel0_setmem(idx, mem);
    case 1: return iris_openmp_kernel1_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t *off, size_t *ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: kernel0(kernel0_args.dst, kernel0_args.src, off, ndr); break;
    case 1: kernel1(kernel1_args.dst, kernel1_args.src, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


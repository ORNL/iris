#include <brisbane/brisbane_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float * Z;
  float A;
  __global float * X;
  __global float * Y;
} brisbane_openmp_saxpy_args;
brisbane_openmp_saxpy_args saxpy_args;

static int brisbane_openmp_saxpy_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&saxpy_args.A, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_saxpy_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy_args.Z = (__global float *__restrict) mem; break;
    case 2: saxpy_args.X = (__global float *__restrict) mem; break;
    case 3: saxpy_args.Y = (__global float *__restrict) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

#include "kernel.openmp.h"

int brisbane_openmp_kernel(const char* name) {
  brisbane_openmp_lock();
  if (strcmp(name, "saxpy") == 0) {
    brisbane_openmp_kernel_idx = 0;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setarg(int idx, size_t size, void* value) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_saxpy_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setmem(int idx, void* mem) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_saxpy_setmem(idx, mem);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: saxpy(saxpy_args.Z, saxpy_args.A, saxpy_args.X, saxpy_args.Y, off, ndr); break;
  }
  brisbane_openmp_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


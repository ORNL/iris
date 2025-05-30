#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global int * Z;
  __global int * X;
  __global int * Y;
  int A;
} iris_openmp_saxpy_args;
iris_openmp_saxpy_args saxpy_args;

static int iris_openmp_saxpy_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 3: memcpy(&saxpy_args.A, value, size); break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_saxpy_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy_args.Z = (__global int*__restrict) mem; break;
    case 1: saxpy_args.X = (__global int*__restrict) mem; break;
    case 2: saxpy_args.Y = (__global int*__restrict) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

#include "kernel.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "saxpy") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_saxpy_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_saxpy_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t *off, size_t *ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: saxpy(saxpy_args.Z, saxpy_args.X, saxpy_args.Y, saxpy_args.A, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


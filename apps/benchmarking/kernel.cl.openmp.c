#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float * Z;
  float A;
  __global float * X;
  __global float * Y;
} iris_openmp_saxpy_args;
iris_openmp_saxpy_args saxpy_args;

static int iris_openmp_saxpy_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&saxpy_args.A, value, size); break;
    default: return IRIS_ERR;
  }
  return IRIS_OK;
}

static int iris_openmp_saxpy_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy_args.Z = (__global float *__restrict) mem; break;
    case 2: saxpy_args.X = (__global float *__restrict) mem; break;
    case 3: saxpy_args.Y = (__global float *__restrict) mem; break;
    default: return IRIS_ERR;
  }
  return IRIS_OK;
}

typedef struct {
  __global double * C;
  __global double * A;
  __global double * B;
} iris_openmp_ijk_args;
iris_openmp_ijk_args ijk_args;

static int iris_openmp_ijk_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERR;
  }
  return IRIS_OK;
}

static int iris_openmp_ijk_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: ijk_args.C = (__global double *__restrict) mem; break;
    case 1: ijk_args.A = (__global double *__restrict) mem; break;
    case 2: ijk_args.B = (__global double *__restrict) mem; break;
    default: return IRIS_ERR;
  }
  return IRIS_OK;
}


#include "kernel.cl.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "saxpy") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_OK;
  }
  if (strcmp(name, "ijk") == 0) {
    iris_openmp_kernel_idx = 1;
    return IRIS_OK;
  }
  return IRIS_ERR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_saxpy_setarg(idx, size, value);
    case 1: return iris_openmp_ijk_setarg(idx, size, value);
  }
  return IRIS_ERR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_saxpy_setmem(idx, mem);
    case 1: return iris_openmp_ijk_setmem(idx, mem);
  }
  return IRIS_ERR;
}

int iris_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: saxpy(saxpy_args.Z, saxpy_args.A, saxpy_args.X, saxpy_args.Y, off, ndr); break;
    case 1: ijk(ijk_args.Z, ijk_args.A, ijk_args.X, ijk_args.Y, off, ndr); break;

  }
  iris_openmp_unlock();
  return IRIS_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


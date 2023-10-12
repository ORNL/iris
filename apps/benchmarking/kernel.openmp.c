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
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_saxpy_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: saxpy_args.Z = (__global float *__restrict) mem; break;
    case 2: saxpy_args.X = (__global float *__restrict) mem; break;
    case 3: saxpy_args.Y = (__global float *__restrict) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

typedef struct {
  __global int * A;
} iris_openmp_add_id_args;
iris_openmp_add_id_args add_id_args;

static int iris_openmp_add_id_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_add_id_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: add_id_args.A = (__global float *__restrict) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

typedef struct {
  __global double * C;
  __global double * A;
  __global double * B;
} iris_openmp_ijk_args;
iris_openmp_ijk_args ijk_args;

static int iris_openmp_ijk_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_ijk_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: ijk_args.C = (__global double *__restrict) mem; break;
    case 1: ijk_args.A = (__global double *__restrict) mem; break;
    case 2: ijk_args.B = (__global double *__restrict) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

typedef struct {
  __global int * A;
} iris_openmp_nothing_args;
iris_openmp_nothing_args nothing_args;

static int iris_openmp_nothing_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_nothing_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: nothing_args.A = (__global float *__restrict) mem; break;
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
  if (strcmp(name, "ijk") == 0) {
    iris_openmp_kernel_idx = 1;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "add_id") == 0) {
    iris_openmp_kernel_idx = 2;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "nothing") == 0) {
    iris_openmp_kernel_idx = 3;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_saxpy_setarg(idx, size, value);
    case 1: return iris_openmp_ijk_setarg(idx, size, value);
    case 2: return iris_openmp_add_id_setarg(idx, size, value);
    case 3: return iris_openmp_nothing_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_saxpy_setmem(idx, mem);
    case 1: return iris_openmp_ijk_setmem(idx, mem);
    case 2: return iris_openmp_add_id_setmem(idx, mem);
    case 3: return iris_openmp_nothing_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t *off, size_t *ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: saxpy(saxpy_args.Z, saxpy_args.A, saxpy_args.X, saxpy_args.Y, off, ndr); break;
    case 1: ijk(ijk_args.C, ijk_args.A, ijk_args.B, off, ndr); break;
    case 2: add_id(add_id_args.A, off, ndr); break;
    case 3: nothing(nothing_args.A, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


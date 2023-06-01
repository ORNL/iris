#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#include "kernel.cl.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "ijk") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_ijk_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_ijk_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: ijk(ijk_args.C, ijk_args.A, ijk_args.B, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


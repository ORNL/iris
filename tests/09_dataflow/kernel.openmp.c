#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global int * AB;
} iris_openmp_kernel0_args;
iris_openmp_kernel0_args kernel0_args;

typedef struct {
  __global int * AB;
  __global int * BC;
} iris_openmp_kernel1_args;
iris_openmp_kernel1_args kernel1_args;

typedef struct {
  __global int * BC;
} iris_openmp_kernel2_args;
iris_openmp_kernel2_args kernel2_args;


static int iris_openmp_kernel0_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel1_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel2_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel0_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel0_args.AB = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel1_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel1_args.AB = (__global int *) mem; break;
    case 1: kernel1_args.BC = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel2_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel2_args.BC = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

#include "kernel.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "kernel_A") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "kernel_B") == 0) {
    iris_openmp_kernel_idx = 1;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "kernel_C") == 0) {
    iris_openmp_kernel_idx = 2;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_kernel0_setarg(idx, size, value);
    case 1: return iris_openmp_kernel1_setarg(idx, size, value);
    case 2: return iris_openmp_kernel2_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_kernel0_setmem(idx, mem);
    case 1: return iris_openmp_kernel1_setmem(idx, mem);
    case 2: return iris_openmp_kernel2_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: kernel_A(kernel0_args.AB, off, ndr); break;
    case 1: kernel_B(kernel1_args.AB, kernel1_args.BC, off, ndr); break;
    case 2: kernel_C(kernel2_args.BC, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


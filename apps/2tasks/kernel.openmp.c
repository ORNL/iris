#include <iris/brisbane_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float * dst;
  __global float * src;
} brisbane_openmp_kernel0_args;
brisbane_openmp_kernel0_args kernel0_args;

static int brisbane_openmp_kernel0_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_kernel0_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel0_args.dst = (__global float *) mem; break;
    case 1: kernel0_args.src = (__global float *) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}
typedef struct {
  __global float * dst;
  __global float * src;
} brisbane_openmp_kernel1_args;
brisbane_openmp_kernel1_args kernel1_args;

static int brisbane_openmp_kernel1_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_kernel1_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel1_args.dst = (__global float *) mem; break;
    case 1: kernel1_args.src = (__global float *) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

#include "kernel.openmp.h"

int brisbane_openmp_kernel(const char* name) {
  brisbane_openmp_lock();
  if (strcmp(name, "kernel0") == 0) {
    brisbane_openmp_kernel_idx = 0;
    return BRISBANE_OK;
  }
  if (strcmp(name, "kernel1") == 0) {
    brisbane_openmp_kernel_idx = 1;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setarg(int idx, size_t size, void* value) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_kernel0_setarg(idx, size, value);
    case 1: return brisbane_openmp_kernel1_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setmem(int idx, void* mem) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_kernel0_setmem(idx, mem);
    case 1: return brisbane_openmp_kernel1_setmem(idx, mem);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: kernel0(kernel0_args.dst, kernel0_args.src, off, ndr); break;
    case 1: kernel1(kernel1_args.dst, kernel1_args.src, off, ndr); break;
  }
  brisbane_openmp_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

//kernel0
typedef struct {
  __global int * C;
  __global int loop;
} iris_openmp_kernel0_args;
iris_openmp_kernel0_args kernel0_args;

static int iris_openmp_kernel0_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&kernel0_args.loop, value, size); break; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel0_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel0_args.C = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

//kernel1
typedef struct {
  __global int * C;
  __global int loop;
} iris_openmp_kernel1_args;
iris_openmp_kernel1_args kernel1_args;

static int iris_openmp_kernel1_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&kernel1_args.loop, value, size); break; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel1_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel1_args.C = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

//kernel2
typedef struct {
  __global int * C;
  __global int loop;
} iris_openmp_kernel2_args;
iris_openmp_kernel2_args kernel2_args;

static int iris_openmp_kernel2_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&kernel2_args.loop, value, size); break; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel2_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel2_args.C = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

//kernel3
typedef struct {
  __global int * C;
  __global int loop;
} iris_openmp_kernel3_args;
iris_openmp_kernel3_args kernel3_args;

static int iris_openmp_kernel3_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&kernel3_args.loop, value, size); break; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_kernel3_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: kernel3_args.C = (__global int *) mem; break;
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
  if (strcmp(name, "kernel2") == 0) {
    iris_openmp_kernel_idx = 2;
    return IRIS_SUCCESS;
  }
  if (strcmp(name, "kernel3") == 0) {
    iris_openmp_kernel_idx = 3;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_kernel0_setarg(idx, size, value);
    case 1: return iris_openmp_kernel1_setarg(idx, size, value);
    case 2: return iris_openmp_kernel2_setarg(idx, size, value);
    case 3: return iris_openmp_kernel3_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_kernel0_setmem(idx, mem);
    case 1: return iris_openmp_kernel1_setmem(idx, mem);
    case 2: return iris_openmp_kernel2_setmem(idx, mem);
    case 3: return iris_openmp_kernel3_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: kernel0(kernel0_args.C, kernel0_args.loop, off, ndr); break;
    case 1: kernel1(kernel1_args.C, kernel1_args.loop, off, ndr); break;
    case 2: kernel2(kernel2_args.C, kernel2_args.loop, off, ndr); break;
    case 3: kernel3(kernel3_args.C, kernel3_args.loop, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


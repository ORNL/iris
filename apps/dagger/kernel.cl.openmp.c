#include <brisbane/brisbane_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global int * A;
} brisbane_openmp_process_args;
brisbane_openmp_process_args process_args;

static int brisbane_openmp_process_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_process_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: process_args.A = (__global int *) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

#include "kernel.cl.openmp.h"

int brisbane_openmp_kernel(const char* name) {
  brisbane_openmp_lock();
  if (strcmp(name, "process") == 0) {
    brisbane_openmp_kernel_idx = 0;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setarg(int idx, size_t size, void* value) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_process_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setmem(int idx, void* mem) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_process_setmem(idx, mem);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: process(process_args.A, off, ndr); break;
  }
  brisbane_openmp_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


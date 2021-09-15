#include <brisbane/brisbane_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global char * b;
  __global char * a;
} brisbane_openmp_uppercase_args;
brisbane_openmp_uppercase_args uppercase_args;

static int brisbane_openmp_uppercase_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_openmp_uppercase_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: uppercase_args.b = (__global char *) mem; break;
    case 1: uppercase_args.a = (__global char *) mem; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

#include "kernel.openmp.h"

int brisbane_openmp_kernel(const char* name) {
  brisbane_openmp_lock();
  if (strcmp(name, "uppercase") == 0) {
    brisbane_openmp_kernel_idx = 0;
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setarg(int idx, size_t size, void* value) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_uppercase_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_setmem(int idx, void* mem) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: return brisbane_openmp_uppercase_setmem(idx, mem);
  }
  return BRISBANE_ERR;
}

int brisbane_openmp_launch(int dim, size_t off, size_t ndr) {
  switch (brisbane_openmp_kernel_idx) {
    case 0: uppercase(uppercase_args.b, uppercase_args.a, off, ndr); break;
  }
  brisbane_openmp_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global int * mem;
} iris_openmp_setid_args;
iris_openmp_setid_args setid_args;

static int iris_openmp_setid_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

static int iris_openmp_setid_setmem(int idx, void* mem) {
  switch (idx) {
    case 0: setid_args.mem = (__global int *) mem; break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

#include "kernel.openmp.h"

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "setid") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_setid_setarg(idx, size, value);
  }
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  switch (iris_openmp_kernel_idx) {
    case 0: return iris_openmp_setid_setmem(idx, mem);
  }
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t *off, size_t *ndr) {
  switch (iris_openmp_kernel_idx) {
    case 0: setid(setid_args.mem, off, ndr); break;
  }
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


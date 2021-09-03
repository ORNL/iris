#include <brisbane/brisbane_poly.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  brisbane_poly_mem Z;
  float A;
  brisbane_poly_mem X;
  brisbane_poly_mem Y;
} brisbane_poly_saxpy_args;
brisbane_poly_saxpy_args saxpy_args;

int brisbane_poly_saxpy_init() {
  brisbane_poly_mem_init(&saxpy_args.Z);
  brisbane_poly_mem_init(&saxpy_args.X);
  brisbane_poly_mem_init(&saxpy_args.Y);
  return BRISBANE_OK;
}

static int brisbane_poly_saxpy_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&saxpy_args.A, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_poly_saxpy_getmem(int idx, brisbane_poly_mem* mem) {
  switch (idx) {
    case 0: memcpy(mem, &saxpy_args.Z, sizeof(brisbane_poly_mem)); break;
    case 2: memcpy(mem, &saxpy_args.X, sizeof(brisbane_poly_mem)); break;
    case 3: memcpy(mem, &saxpy_args.Y, sizeof(brisbane_poly_mem)); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

#include "kernel.cl.poly.h"

int brisbane_poly_kernel(const char* name) {
  brisbane_poly_lock();
  if (strcmp(name, "saxpy") == 0) {
    brisbane_poly_kernel_idx = 0;
    return saxpy_poly_available();
  }
  return BRISBANE_ERR;
}

int brisbane_poly_setarg(int idx, size_t size, void* value) {
  switch (brisbane_poly_kernel_idx) {
    case 0: return brisbane_poly_saxpy_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_poly_launch(int dim, size_t* wgo, size_t* wgs, size_t* gws, size_t* lws) {
  int ret = BRISBANE_OK;
  switch (brisbane_poly_kernel_idx) {
    case 0: brisbane_poly_saxpy_init(); ret = saxpy(wgo[0], wgo[1], wgo[2], wgs[0], wgs[1], wgs[2], gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]); break;
  }
  brisbane_poly_unlock();
  return ret;
}

int brisbane_poly_getmem(int idx, brisbane_poly_mem* mem) {
  switch (brisbane_poly_kernel_idx) {
    case 0: return brisbane_poly_saxpy_getmem(idx, mem);
  }
  return BRISBANE_ERR;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


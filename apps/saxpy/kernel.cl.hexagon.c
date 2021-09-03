#include <brisbane/brisbane_hexagon.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float *__restrict Z;
  int Zlen;
  float A;
  __global float *__restrict X;
  int Xlen;
  __global float *__restrict Y;
  int Ylen;
} brisbane_hexagon_saxpy_args;
brisbane_hexagon_saxpy_args saxpy_args;

static int brisbane_hexagon_saxpy_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    case 1: memcpy(&saxpy_args.A, value, size); break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_hexagon_saxpy_setmem(int idx, void* mem, int size) {
  switch (idx) {
    case 0: saxpy_args.Z = (__global float *__restrict) mem; saxpy_args.Zlen = size; break;
    case 2: saxpy_args.X = (__global float *__restrict) mem; saxpy_args.Xlen = size; break;
    case 3: saxpy_args.Y = (__global float *__restrict) mem; saxpy_args.Ylen = size; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}
static int (*saxpy)(__global float *__restrict, int, float, __global float *__restrict, int, __global float *__restrict, int, int, int);

int brisbane_hexagon_kernel(const char* name) {
  brisbane_hexagon_lock();
  if (strcmp(name, "saxpy") == 0) {
    brisbane_hexagon_kernel_idx = 0;
    if (!saxpy) saxpy = (int (*)(__global float *__restrict, int, float, __global float *__restrict, int, __global float *__restrict, int, int, int)) dlsym(brisbane_hexagon_dl, "brisbanehxg_saxpy");
    if (!saxpy) printf("%s", dlerror());
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_hexagon_setarg(int idx, size_t size, void* value) {
  switch (brisbane_hexagon_kernel_idx) {
    case 0: return brisbane_hexagon_saxpy_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_hexagon_setmem(int idx, void* mem, int size) {
  switch (brisbane_hexagon_kernel_idx) {
    case 0: return brisbane_hexagon_saxpy_setmem(idx, mem, size);
  }
  return BRISBANE_ERR;
}

int brisbane_hexagon_launch(int dim, size_t off, size_t ndr) {
  switch (brisbane_hexagon_kernel_idx) {
    case 0: (*saxpy)(saxpy_args.Z, saxpy_args.Zlen, saxpy_args.A, saxpy_args.X, saxpy_args.Xlen, saxpy_args.Y, saxpy_args.Ylen, off, ndr); break;
  }
  brisbane_hexagon_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


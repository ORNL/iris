#include <brisbane/brisbane_hexagon.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  __global float *__restrict C;
  int Clen;
  __global float *__restrict A;
  int Alen;
  __global float *__restrict B;
  int Blen;
} brisbane_hexagon_ijk_args;
brisbane_hexagon_ijk_args ijk_args;

static int brisbane_hexagon_ijk_setarg(int idx, size_t size, void* value) {
  switch (idx) {
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

static int brisbane_hexagon_ijk_setmem(int idx, void* mem, int size) {
  switch (idx) {
    case 0: ijk_args.C = (__global float *__restrict) mem; ijk_args.Clen = size; break;
    case 1: ijk_args.A = (__global float *__restrict) mem; ijk_args.Alen = size; break;
    case 2: ijk_args.B = (__global float *__restrict) mem; ijk_args.Blen = size; break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}
static int (*ijk)(__global float *__restrict, int, __global float *__restrict, int, __global float *__restrict, int, int, int);

int brisbane_hexagon_kernel(const char* name) {
  brisbane_hexagon_lock();
  if (strcmp(name, "ijk") == 0) {
    brisbane_hexagon_kernel_idx = 0;
    if (!ijk) ijk = (int (*)(__global float *__restrict, int, __global float *__restrict, int, __global float *__restrict, int, int, int)) dlsym(brisbane_hexagon_dl, "brisbanehxg_ijk");
    if (!ijk) printf("%s", dlerror());
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int brisbane_hexagon_setarg(int idx, size_t size, void* value) {
  switch (brisbane_hexagon_kernel_idx) {
    case 0: return brisbane_hexagon_ijk_setarg(idx, size, value);
  }
  return BRISBANE_ERR;
}

int brisbane_hexagon_setmem(int idx, void* mem, int size) {
  switch (brisbane_hexagon_kernel_idx) {
    case 0: return brisbane_hexagon_ijk_setmem(idx, mem, size);
  }
  return BRISBANE_ERR;
}

int brisbane_hexagon_launch(int dim, size_t off, size_t ndr) {
  int ret = -1;
  switch (brisbane_hexagon_kernel_idx) {
    case 0: ret = (*ijk)(ijk_args.C, ijk_args.Clen, ijk_args.A, ijk_args.Alen, ijk_args.B, ijk_args.Blen, off, ndr); break;
  }
  printf("[%s:%d] ret[%d]\n", __FILE__, __LINE__, ret);
  brisbane_hexagon_unlock();
  return BRISBANE_OK;
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif


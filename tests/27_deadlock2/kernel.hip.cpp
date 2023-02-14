#include <hip/hip_runtime.h>

extern "C" __global__
void copy(int* dst, int *src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dst[i] = src[i];
}


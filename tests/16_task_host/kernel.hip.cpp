#include <hip/hip_runtime.h>

extern "C" __global__ void process(int* A, int* factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = i * factor[0];
}


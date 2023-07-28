#include <hip/hip_runtime.h>

extern "C" __global__ void add1(int* A) {
  size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
  A[idx]++;
}


#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(int* A, int* B, int* C) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}


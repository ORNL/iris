#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(int* C, int* A, int* B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] += A[i] + B[i];
}


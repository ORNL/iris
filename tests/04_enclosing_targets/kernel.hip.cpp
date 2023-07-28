#include <hip/hip_runtime.h>

extern "C" __global__ void loop0(int* A) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  A[id] *= 2;
}

extern "C" __global__ void loop1(int* B, int* A) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  B[id] += A[id];
}

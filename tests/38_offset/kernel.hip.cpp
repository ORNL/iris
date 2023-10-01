#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(int* A, int* B, int* C) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}
extern "C" __global__ void vecadd_with_offsets(int* A, int* B, int* C, size_t blockOff_x, size_t blockOff_y, size_t blockOff_z) {
  size_t id = (blockOff_x + blockIdx.x) * blockDim.x + threadIdx.x;
  //int id = blockIdx.x * blockDim.x + threadIdx.x;
  C[id] = A[id] + B[id];
}


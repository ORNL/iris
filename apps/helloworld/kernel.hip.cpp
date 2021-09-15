#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy(float* Z, float A, float* X, float* Y) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  Z[id] = A * X[id] + Y[id];
}

extern "C" __global__ void saxpy_with_offsets(float* Z, float A, float* X, float* Y, size_t blockOff_x, size_t blockOff_y, size_t blockOff_z) {
  size_t id = (blockOff_x + blockIdx.x) * blockDim.x + threadIdx.x;
  Z[id] = A * X[id] + Y[id];
}


#include <hip/hip_runtime.h>
extern "C" __global__ void saxpy(int* Z, int* X, int* Y, int A) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  Z[id] = A * X[id] + Y[id];
}


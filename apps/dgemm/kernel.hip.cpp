#include <hip/hip_runtime.h>

extern "C" __global__ void ijk(double* C, double* A, double* B) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  size_t SIZE = gridDim.y * blockDim.y;

  double sum = 0.0;
  for (size_t k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;
}


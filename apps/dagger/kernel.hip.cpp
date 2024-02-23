#include <hip/hip_runtime.h>

extern "C" __global__ void process(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] *= 100;
}

extern "C" __global__ void ijk(double* C, double* A, double* B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t SIZE = gridDim.x * blockDim.x;

  double sum = 0.0;
  for (size_t k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;
}

extern "C" __global__ void special_ijk(double* C, double* A, double* B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t SIZE = gridDim.x * blockDim.x;

  double sum = 0.0;
  for (size_t k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;
  ///quick value-swap test
  double tmp = A[i];
  A[i] = B[i];
  B[i] = tmp;
}

extern "C" __global__ void bigk(double* C, double* A, double* B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t SIZE = gridDim.x * blockDim.x;

  for (size_t j = 0; j < SIZE; j++) {
    double sum = 0.0;
    for (size_t l = 0; l < 10; l++) {
      for (size_t k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
    }
    C[i * SIZE + j] += sum;
  }
}



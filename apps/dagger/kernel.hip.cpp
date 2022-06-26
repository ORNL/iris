#include <hip/hip_runtime.h>

extern "C" __global__ void process(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] *= 100;
}

extern "C" __global__ void ijk(double* C, double* A, double* B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t SIZE = gridDim.x * blockDim.x;

  //logic to span over local area as separate threads
  for (;i < SIZE;i++){
    for (;j < SIZE;j++){

      double sum = 0.0;
      for (size_t k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;

    }
  }
}

//the bigk kernel is the ijk task with an added for-loop to increase the kernel running time
extern "C" __global__ void bigk(double* C, double* A, double* B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t SIZE = gridDim.x * blockDim.x;

  //logic to span over local area as separate threads
  for (;i < SIZE;i++){
    for (;j < SIZE;j++){

      double sum = 0.0;
      for (size_t l = 0; l < SIZE; l++)
      for (size_t k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;

    }
  }
}

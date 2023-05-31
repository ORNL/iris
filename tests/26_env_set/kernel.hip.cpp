
#include <hip/hip_runtime.h>
extern "C" __global__
void add1(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = A[i] + 1;
}

extern "C" __global__
void add1_v2(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int a = A[i];
  a++;
  a++;
  A[i] = a;
}

extern "C" __global__
void add2(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = A[i] + 2;
}

extern "C" __global__
void add2_v2(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int a = A[i];
  a += 2;
  A[i] = a;
}


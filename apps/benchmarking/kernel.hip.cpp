#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy(float* Z, float A, float* X, float* Y) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  Z[id] = A * X[id] + Y[id];
}

extern "C" __global__ void mem1(int l, float* m1) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i;
  }
}

extern "C" __global__ void mem2(int l, float* m1, float* m2) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i];
  }
}

extern "C" __global__ void mem3(int l, float* m1, float* m2, float* m3) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i];
  }
}

extern "C" __global__ void mem4(int l, float* m1, float* m2, float* m3, float* m4) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i];
  }
}

extern "C" __global__ void mem5(int l, float* m1, float* m2, float* m3, float* m4, float* m5) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i] + m5[i];
  }
}

extern "C" __global__ void mem6(int l, float* m1, float* m2, float* m3, float* m4, float* m5, float* m6) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i] + m5[i] + m6[i];
  }
}

extern "C" __global__ void mem7(int l, float* m1, float* m2, float* m3, float* m4, float* m5, float* m6, float* m7) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i] + m5[i] + m6[i] + m7[i];
  }
}

extern "C" __global__ void mem8(int l, float* m1, float* m2, float* m3, float* m4, float* m5, float* m6, float* m7, float* m8) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i] + m5[i] + m6[i] + m7[i] + m8[i];
  }
}

extern "C" __global__ void mem9(int l, float* m1, float* m2, float* m3, float* m4, float* m5, float* m6, float* m7, float* m8, float* m9) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i] + m5[i] + m6[i] + m7[i] + m8[i] + m9[i];
  }
}

extern "C" __global__ void mem10(int l, float* m1, float* m2, float* m3, float* m4, float* m5, float* m6, float* m7, float* m8, float* m9, float* m10) {
  for (int loop = 0; loop < l; loop++) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  m1[i] = i + m2[i] + m3[i] + m4[i] + m5[i] + m6[i] + m7[i] + m8[i] + m9[i] + m10[i];
  }
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

extern "C" __global__ void nothing(int* A) {
}

extern "C" __global__ void add_id(int* A) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = A[i] + i;
}


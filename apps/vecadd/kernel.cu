extern "C" __global__ void vecadd(int* A, int* B, int* C) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  C[id] = A[id] + B[id];
}

extern "C" __global__ void empty(int* A, int* B, int* C) {
}

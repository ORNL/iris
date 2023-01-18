extern "C" __global__ void ijk(int* C, int* A, int *B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] += A[i] + B[i];
}


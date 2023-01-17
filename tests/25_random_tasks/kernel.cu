extern "C" __global__ void add1(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i]++;
}


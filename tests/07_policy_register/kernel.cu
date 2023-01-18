extern "C" __global__ void process(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = i;
}


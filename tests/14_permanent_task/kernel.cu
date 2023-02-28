extern "C" __global__ void process_task(int* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i]++;
}


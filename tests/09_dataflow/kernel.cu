extern "C" __global__
void kernel_A(int* AB) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  AB[i] = i;
}

extern "C" __global__
void kernel_B(int* AB, int* BC) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  BC[i] = AB[i] * 10;
}

extern "C" __global__
void kernel_C(int* BC) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  BC[i] = BC[i] * 2;
}


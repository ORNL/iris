extern "C" __global__ void kernel0(int* C, int loop) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}

extern "C" __global__ void kernel1(int* C, int loop) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}

extern "C" __global__ void kernel2(int* C, int loop) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}

extern "C" __global__ void kernel3(int* C, int loop) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}


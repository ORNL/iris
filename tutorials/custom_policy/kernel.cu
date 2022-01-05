extern "C" __global__ void setid(int* mem) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  mem[id] = id;
}


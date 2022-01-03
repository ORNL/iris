extern "C" __global__ void kernel0(float* dst, float* src) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  dst[id] = src[id];
}

extern "C" __global__ void kernel1(float* dst, float* src) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  dst[id] += src[id];
}


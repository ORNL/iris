extern "C" __global__ void saxpy(double* Z, double* X, double* Y, double A) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  Z[id] = A * X[id] + Y[id];
}


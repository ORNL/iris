extern "C" __global__ void saxpy(float* S, float A, float* X, float* Y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  S[i] = A * X[i] + Y[i];
}

extern "C" __global__ void vecadd(int* A, int* B, int* C) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  C[id] = A[id] + B[id];
}
extern "C" __global__ void vecadd_with_offsets(int* A, int* B, int* C, size_t blockOff_x, size_t blockOff_y, size_t blockOff_z) {
  size_t id = blockOff_x + blockIdx.x * blockDim.x + threadIdx.x;
  //size_t id = (blockOff_x + blockIdx.x) * blockDim.x + threadIdx.x;
  //printf("block offset in vecadd : %i\n",blockOff_x);
  C[id] = A[id] + B[id];
}
extern "C" __global__ void blockadd(int* A, int* B, int* C, size_t SIZE) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  C[j * SIZE + i] = A[j * SIZE + i] + B[j * SIZE + i];
}
extern "C" __global__ void blockadd_with_offsets(int* A, int* B, int* C, size_t SIZE, size_t blockOff_x, size_t blockOff_y, size_t blockOff_z) {
  size_t i = blockOff_x + blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockOff_y + blockIdx.y * blockDim.y + threadIdx.y;
  //size_t X_SIZE = blockOff_x + gridDim.x;
  //size_t SIZE = 16;
  //size_t SIZE = gridDim.y * blockDim.y;
  //printf("using ij:(%lu, %lu) blockOff:(%lu, %lu), blockDim:(%i, %i), blockIdx:(%i, %i), threadIdx:(%i, %i), gridDim:(%i, %i) SIZE:%lu:%lu \n", i, j, blockOff_x, blockOff_y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gridDim.x, gridDim.y, X_SIZE, Y_SIZE);
  C[j * SIZE + i] = A[j * SIZE + i] + B[j * SIZE + i];
}
#if 0
extern "C" __global__ void blockadd_with_offsets(double* A, double* B, double* C, size_t blockOff_x, size_t blockOff_y, size_t blockOff_z) {
  size_t i = (blockOff_x + blockIdx.x) * blockDim.x + threadIdx.x;
  size_t j = (blockOff_y + blockIdx.y) * blockDim.y + threadIdx.y;
  size_t SIZE = gridDim.y * blockDim.y;
  //printf("using global size = %i offset: %i\n",SIZE,blockOff_x);
  C[j * SIZE + i] = A[j * SIZE + i] + B[j * SIZE + i];
}
#endif

__kernel void saxpy(__global float* restrict Z, float A, __global float* restrict X, __global float* restrict Y) {
  size_t id = get_global_id(0);
  Z[id] = A * X[id] + Y[id];
}

__kernel void ijk(__global double* restrict C, __global double* restrict A, __global double* restrict B) {
  size_t i = get_global_id(1);
  size_t j = get_global_id(0);
  size_t SIZE = get_global_size(0);

  double sum = 0.0;
  for (size_t k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;
}

__kernel__ void nothing(__global int* A) {
}

__kernel__ void add_id(__global int* A) {
  size_t i = get_global_id(0);
  A[i] = A[i] + i;
}


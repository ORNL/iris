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


__kernel void loop0(__global int* restrict A) {
  size_t i = get_global_id(0);
  A[i] *= 2;
}

__kernel void loop1(__global int* restrict B, __global int* restrict A) {
  size_t i = get_global_id(0);
  B[i] += A[i];
}


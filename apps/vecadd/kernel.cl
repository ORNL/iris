__kernel void vecadd(__global int* restrict A, __global int* restrict B, __global int* restrict C) {
  size_t id = get_global_id(0);
  C[id] = A[id] + B[id];
}

__kernel void empty(__global int* restrict A, __global int* restrict B, __global int* restrict C) {
}

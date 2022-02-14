__kernel void loop0(__global int* restrict C, __global int* restrict A, __global int* restrict B) {
  size_t id = get_global_id(0);
  C[id] = A[id] + B[id];
}

__kernel void loop1(__global int* restrict D, __global int* restrict C) {
  size_t id = get_global_id(0);
  D[id] = C[id] * 10;
}

__kernel void loop2(__global int* restrict E, __global int* restrict D) {
  size_t id = get_global_id(0);
  E[id] = D[id] * 2;
}

__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B) {
  size_t id = get_global_id(0);
  C[id] = A[id] + B[id];
}


__kernel void vecadd(__global int* restrict A, __global int* restrict B, __global int* restrict C) {
  size_t id = get_global_id(0);
  C[id] = A[id] + B[id];
}
__kernel  void blockadd(__global int* restrict A, __global int* restrict B, __global int* restrict C, unsigned long SIZE) {
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  C[y*SIZE+x] = A[y*SIZE+x] + B[y*SIZE+x];
}

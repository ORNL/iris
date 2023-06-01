__kernel void vecadd(__global int* restrict C, __global int* restrict A, __global int* restrict B) {
  size_t id = get_global_id(0);
  printf("A[%i] = %i \n", id, A[id]);
  C[id] += A[id] + B[id];
}

__kernel void process(__global int* restrict A, __global int* restrict factor) {
  size_t id = get_global_id(0);
  A[id] = id * factor[0];
}


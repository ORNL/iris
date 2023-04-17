__kernel void process(__global int* A) {
  size_t i = get_global_id(0);
  A[i] = i;
}


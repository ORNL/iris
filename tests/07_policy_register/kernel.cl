__kernel void process(__global int* A) {
  int i = get_global_id(0);
  A[i] = i;
}


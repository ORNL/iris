__kernel void saxpy(__global int* restrict Z, __global int* restrict X, __global int* restrict Y, int SIZE, int A) {
  size_t id = get_global_id(0);
  Z[id] = A * X[id] + Y[id];
}


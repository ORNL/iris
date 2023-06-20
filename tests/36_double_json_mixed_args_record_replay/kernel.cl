__kernel void saxpy(__global double* restrict Z, __global double* restrict X, __global double* restrict Y, double A) {
  size_t id = get_global_id(0);
  Z[id] = A * X[id] + Y[id];
}


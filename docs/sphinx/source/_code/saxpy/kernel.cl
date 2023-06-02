__kernel void saxpy(__global float* S, float A, __global float* X, __global float* Y) {
  int i = get_global_id(0);
  S[i] = A * X[i] + Y[i];
}

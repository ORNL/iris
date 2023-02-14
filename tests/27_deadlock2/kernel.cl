__kernel void copy(__global int* dst, __global int* src) {
  size_t i = get_global_id(0);
  dst[i] = src[i];
}


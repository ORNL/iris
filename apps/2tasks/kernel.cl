__kernel void kernel0(__global float* dst, __global float* src) {
  int id = get_global_id(0);
  dst[id] = src[id];
}

__kernel void kernel1(__global float* dst, __global float* src) {
  int id = get_global_id(0);
  dst[id] += src[id];
}


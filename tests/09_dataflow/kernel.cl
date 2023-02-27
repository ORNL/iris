__kernel void kernel_A(__global int* restrict AB) {
  size_t i = get_global_id(0);
  AB[i] = i;
}

__kernel void kernel_B(__global int* restrict AB, __global int* restrict BC) {
  size_t i = get_global_id(0);
  BC[i] = AB[i] * 10;
}

__kernel void kernel_C(__global int* restrict BC) {
  size_t i = get_global_id(0);
  BC[i] = BC[i] * 2;
}


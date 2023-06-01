__kernel void kernel0(__global int* C, int loop) {
  size_t id = get_global_id(0);
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}

__kernel void kernel1(__global int* C, int loop) {
  size_t id = get_global_id(0);
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}

__kernel void kernel2(__global int* C, int loop) {
  size_t id = get_global_id(0);
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}

__kernel void kernel3(__global int* C, int loop) {
  size_t id = get_global_id(0);
  for (int i = 0; i < loop; i++) {
  for (int j = 0; j < loop; j++) {
    C[id] += id;
  }
  }
}


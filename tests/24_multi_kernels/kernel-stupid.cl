__kernel void add1(__global int* A) {
  int i = get_global_id(0);
  A[i] = A[i] + 2;
  A[i] = A[i] - 2;
  A[i] = A[i] + 1;
}

__kernel void add1_v1(__global int* A) {
  int i = get_global_id(0);
  int a = A[i];
  a++;
  A[i] = a;
}

__kernel void add2(__global int* A) {
  int i = get_global_id(0);
  A[i] = A[i] + 5;
  A[i] = A[i] - 5;
  A[i] = A[i] + 2;
}

__kernel void add2_v2(__global int* A) {
  int i = get_global_id(0);
  int a = A[i];
  a += 2;
  A[i] = a;
}


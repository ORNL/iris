__kernel void process(__global int* A) {
  size_t i = get_global_id(0);
  A[i] *= 100;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void ijk(__global double* restrict C, __global double* restrict A, __global double* restrict B) {
  size_t i = get_global_id(1);
  size_t j = get_global_id(0);
  size_t SIZE = get_global_size(0);

  //logic to span over local area as separate threads
  for (;i < SIZE;i++){
    for (;j < SIZE;j++){

      double sum = 0.0;
      for (size_t k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;

    }
  }
}

//the bigk kernel is the ijk task with an added for-loop to increase the kernel running time
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void bigk(__global double* restrict C, __global double* restrict A, __global double* restrict B) {
  size_t i = get_global_id(1);
  size_t j = get_global_id(0);
  size_t SIZE = get_global_size(0);

  //logic to span over local area as separate threads
  for (;i < SIZE;i++){
    for (;j < SIZE;j++){

      double sum = 0.0;
      for (size_t l = 0; l < SIZE; l++)
      for (size_t k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;

    }
  }
}

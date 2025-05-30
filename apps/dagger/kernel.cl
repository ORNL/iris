__kernel void process(__global int* A) {
  size_t i = get_global_id(0);
  A[i] *= 100;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void ijk(__global double* restrict C, __global double* restrict A, __global double* restrict B) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t SIZE = get_global_size(0);

  double sum = 0.0;
  for (size_t k = 0; k < SIZE; k++) {
    sum += A[i * SIZE + k] * B[k * SIZE + j];
  }
  C[i * SIZE + j] = sum;

}
/*
#include "eth_crc32_lut.h"

__kernel void crc32_slice8(__global const uint* restrict data, uint length_bytes, const uint length_ints, __global uint* restrict res)
{
  __private uint crc;
  __private uchar* currentChar;
  __private uint one,two;
  __private size_t i,j,gid;

  crc = 0xFFFFFFFF;
  gid = get_global_id(0);
  i = gid * length_ints;

  while (length_bytes >= 8) // process eight bytes at once
  {
    one = data[i++] ^ crc;
    two = data[i++];
    crc = crc32Lookup[7][ one      & 0xFF] ^
      crc32Lookup[6][(one>> 8) & 0xFF] ^
      crc32Lookup[5][(one>>16) & 0xFF] ^
      crc32Lookup[4][ one>>24        ] ^
      crc32Lookup[3][ two      & 0xFF] ^
      crc32Lookup[2][(two>> 8) & 0xFF] ^
      crc32Lookup[1][(two>>16) & 0xFF] ^
      crc32Lookup[0][ two>>24        ];
    length_bytes -= 8;
  }

  while(length_bytes) // remaining 1 to 7 bytes
  {
    one = data[i++];
    currentChar = (__private unsigned char*) &one;
    j=0;
    while (length_bytes && j < 4)
    {
      length_bytes = length_bytes - 1;
      crc = (crc >> 8) ^ crc32Lookup[0][(crc & 0xFF) ^ currentChar[j]];
      j = j + 1;
    }
  }

  res[gid] = ~crc;
}
*/

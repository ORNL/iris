extern "C" __global__ void uppercase(char* b, char* a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (a[i] >= 'a' && a[i] <= 'z') b[i] = a[i] + 'A' - 'a';
  else b[i] = a[i];
}
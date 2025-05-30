// Kernel 1: add1
extern "C" __global__ void add1(int* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = A[i] + 1;
}

// Kernel 2: add1_v2
extern "C" __global__ void add1_v2(int* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int a = A[i];
    a++;
    a++;
    A[i] = a;
}

// Kernel 3: add2
extern "C" __global__ void add2(int* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = A[i] + 2;
}

// Kernel 4: add2_v2
extern "C" __global__ void add2_v2(int* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int a = A[i];
    a += 2;
    A[i] = a;
}

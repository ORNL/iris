#ifndef __DEFAULT_IRIS_GPU_REDUCTION_H_
#define __DEFAULT_IRIS_GPU_REDUCTION_H_
#pragma once

#include <limits>
#include "iris/iris_gpu.h"

// CUDA kernel for block-level reduction (generic operation)

// Custom functors
struct iris_op_max{
    template <typename T>
    __device__ __host__ T operator()(T a, T b) const { return a > b ? a : b; }
};

struct iris_op_min {
    template <typename T>
    __device__ __host__ T operator()(T a, T b) const { return a < b ? a : b; }
};

struct iris_op_sum {
    template <typename T>
    __device__ __host__ T operator()(T a, T b) const { return a + b; }
};

struct iris_op_prod {
    template <typename T>
    __device__ __host__ T operator()(T a, T b) const { return a * b; }
};
template <typename T, typename Op>
struct ReductionIdentity;

template <typename T>
struct ReductionIdentity<T, iris_op_max> {
    static __host__ __device__ T value() { return -std::numeric_limits<T>::infinity(); }
};

template <typename T>
struct ReductionIdentity<T, iris_op_min> {
    static __host__ __device__ T value() { return std::numeric_limits<T>::infinity(); }
};

template <typename T>
struct ReductionIdentity<T, iris_op_sum> {
    static __host__ __device__ T value() { return T(0); }
};

template <typename T>
struct ReductionIdentity<T, iris_op_prod> {
    static __host__ __device__ T value() { return T(1); }
};


template <typename T, typename BinaryOp>
__global__ void iris_reduction_gpu_block(T *output, const T* input, size_t N, BinaryOp op) {
    T init = ReductionIdentity<T, BinaryOp>::value();
    extern __shared__ __align__(sizeof(T)) unsigned char s[];
    T* sdata = reinterpret_cast<T*>(s);
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    T local = init;
    for (int idx = i; idx < N; idx += gridSize) {
        local = op(local, input[idx]);
    }

    sdata[tid] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
// Final kernel: reduce block results into one
template <typename T, typename BinaryOp>
__global__ void iris_reduction_gpu_final(T *output, const T* input, size_t N, BinaryOp op) {
    T init = ReductionIdentity<T, BinaryOp>::value();
    extern __shared__ __align__(sizeof(T)) unsigned char s[];
    T* gsdata = reinterpret_cast<T*>(s);
    unsigned int tid = threadIdx.x;

    T local = init;
    for (int idx = tid; idx < N; idx += blockDim.x) {
        local = op(local, input[idx]);
    }

    gsdata[tid] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            gsdata[tid] = op(gsdata[tid], gsdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = gsdata[0];
    }
}
// Top-level wrapper for stable device reduction
template <typename T, typename BinaryOp>
void iris_reduction_gpu(T* d_output, const T* d_input, size_t N, BinaryOp op, void *stream) {
    const int threadsPerBlock = 256;
    const int numBlocks = IRIS_MIN((N + threadsPerBlock - 1) / threadsPerBlock, 1024);
    T* d_block_results = d_output; // reuse output buffer
    if (stream != NULL) {
        gpuStream_t gpu_stream= reinterpret_cast<gpuStream_t>(((void **)stream)[0]);
        iris_reduction_gpu_block<T, BinaryOp><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(T), gpu_stream>>>(
        d_block_results, d_input, N, op);

        iris_reduction_gpu_final<T, BinaryOp><<<1, threadsPerBlock, threadsPerBlock * sizeof(T), gpu_stream>>>(
        d_block_results, d_block_results, numBlocks, op);
    }
    else {
        iris_reduction_gpu_block<T, BinaryOp><<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
                d_block_results, d_input, N, op);

        iris_reduction_gpu_final<T, BinaryOp><<<1, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
                d_block_results, d_block_results, numBlocks, op);
    }
}

#endif //__DEFAULT_IRIS_GPU_REDUCTION_H_

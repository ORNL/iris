// rope_cuda_raw.cu
#pragma once
#include <limits>
#include "iris/iris_gpu.h"
#include <math.h>

template <typename T> __device__ T __iris_cos(T x);
template <> __device__ float  __iris_cos(float x)  { return cosf(x); }
template <> __device__ double __iris_cos(double x) { return cos(x);  }

template <typename T> __device__ T __iris_pow(T x, T y);
template <> __device__ float  __iris_pow(float x, float y)   { return powf(x, y); }
template <> __device__ double __iris_pow(double x, double y) { return pow(x, y);  }

template <typename T> __device__ T __iris_sin(T x);
template <> __device__ float  __iris_sin(float x)  { return sinf(x); }
template <> __device__ double __iris_sin(double x) { return sin(x);  }

// -----------------------------------------------------------------------------
// CUDA kernel: operates on an interleaved complex buffer [re,im, re,im, …]
// Template parameter T can be float or double.
// -----------------------------------------------------------------------------
template<typename T>
__global__ void iris_rope_gpu_core(
    T*     x,            // interleaved complex buffer
    size_t cols,         // number of head indices
    size_t rows,         // number of i indices
    int    pos,
    T      theta_scale
) 
{
    size_t i    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t head = blockIdx.y * blockDim.y + threadIdx.y;
    if (head < rows && i < cols) {
        // flatten (i,head) into scalar index
        size_t idx  = i + head * cols;
        size_t idx2 = 2 * idx;

        // compute θ = (pos - 1) * theta_scale^(i - 1)
        T exponent = (i == 0 ? T(1) : __iris_pow(theta_scale, T(i)));
        T theta    = T(pos - 1) * exponent;

        // cis(θ) = cos(θ) + i·sin(θ)
        T c = __iris_cos(theta);
        T s = __iris_sin(theta);

        // load
        T a = x[idx2 + 0];  // real part
        T b = x[idx2 + 1];  // imag part

        // (a+ib)*(c+is) = (a*c - b*s) + i*(a*s + b*c)
        x[idx2 + 0] = a * c - b * s;
        x[idx2 + 1] = a * s + b * c;
    }
}

// -----------------------------------------------------------------------------
// Host‐side launcher for the above kernel
// -----------------------------------------------------------------------------
template<typename T>
void iris_rope_gpu(
    T*     d_x,
    size_t cols,
    size_t rows,
    int    pos,
    T      theta_scale, void *stream
) {
    // tune block size if you like
    dim3 block(16, 16);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(((void **)stream)[0]);
        iris_rope_gpu_core<T><<<grid, block, 0, gpu_stream>>>(d_x, cols, rows, pos, theta_scale);
    }
    else {
        iris_rope_gpu_core<T><<<grid, block>>>(d_x, cols, rows, pos, theta_scale);
    }
}


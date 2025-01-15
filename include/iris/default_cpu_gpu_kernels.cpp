#include <stdio.h>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>  // HIP header for HIPCC
#include <hiprand/hiprand_kernel.h>
#define gpuStream_t         hipStream_t
#define gpurandState        hiprandState_t
#define gpurandStateSobol32 hiprandStateSobol32_t
#define gpurand_init        hiprand_init
#define gpurand_uniform     hiprand_uniform
#define gpurand_normal      hiprand_normal
#define gpurand_log_normal  hiprand_log_normal
#define gpurand             hiprand
#define gpurandDirectionVectors32_t hiprandDirectionVectors32_t
#define IS_DEFAULT_GPU 1 
#endif // __HIPCC__

#ifdef __CUDACC__
#include <cuda.h>     // CUDA header for NVCC
#include <cuda_runtime.h>     // CUDA header for NVCC
#include <curand_kernel.h>
#define gpuStream_t         CUstream //cudaStream_t
#define gpurandState        curandState_t
#define gpurandStateSobol32 curandStateSobol32_t
#define gpurand_init        curand_init
#define gpurand_uniform     curand_uniform
#define gpurand_normal      curand_normal
#define gpurand_log_normal  curand_log_normal
#define gpurand             curand
#define gpurandDirectionVectors32_t curandDirectionVectors32_t
#define IS_DEFAULT_GPU 1 
#endif // __CUDACC__

#ifndef IS_DEFAULT_GPU
#include <math.h>
#include <omp.h>
#include <random>
#endif

#include <stdint.h>
#include <stdlib.h>

#ifdef IS_DEFAULT_GPU
template <typename T>
__global__ void iris_reset_core(T *arr, T value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = value;
    }
}

// Generic random number generation function template
template <typename T, typename RNGState>
__device__ T generate_uniform_random(RNGState* state, T min, T max) {
    if constexpr (std::is_floating_point<T>::value) {
        T rand_val = gpurand_uniform(state);  // Uniform random float in (0, 1]
        return min + rand_val * (max - min); // Scale to [min, max]
    } else {
        unsigned int rand_int = gpurand(state);
        return min + (rand_int % (max - min + 1)); // Integer random number
    }
}

template <typename T, typename RNGState>
__device__ T generate_normal_random(RNGState* state, T mean, T stddev) {
    return gpurand_normal(state) * stddev + mean;  // Normal random (mean, stddev)
}

template <typename T, typename RNGState>
__device__ T generate_log_normal_random(RNGState* state, T mean, T stddev) {
    return gpurand_log_normal(state, mean, stddev);  // Log-normal random
}

template <typename T>
__global__ void  iris_random_uniform_seq_core(T *arr, unsigned long long seed, size_t size, T p1, T p2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gpurandState state;
        gpurand_init(seed, (unsigned long)idx, (unsigned long)0, &state);
        arr[idx] = generate_uniform_random<T>(&state, p1, p2);
    }
}

template <typename T>
__global__ void  iris_random_normal_seq_core(T *arr, unsigned long long seed, size_t size, T p1, T p2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gpurandState state;
        gpurand_init(seed, idx, 0, &state);
        arr[idx] = generate_normal_random<T>(&state, p1, p2);
    }
}

template <typename T>
__global__ void  iris_random_log_normal_seq_core(T *arr, unsigned long long seed, size_t size, T p1, T p2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gpurandState state;
        gpurand_init(seed, idx, 0, &state);
        arr[idx] = generate_log_normal_random<T>(&state, p1, p2);
    }
}

template <typename T>
__global__ void  iris_random_uniform_seq_sobol_core(T *arr, unsigned long long seed, size_t size, T p1, T p2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gpurandStateSobol32 state;
        gpurand_init(NULL, (unsigned int)idx, &state);
        arr[idx] = generate_uniform_random<T>(&state, p1, p2);
    }
}

template <typename T>
__global__ void  iris_random_normal_seq_sobol_core(T *arr, unsigned long long seed, size_t size, T p1, T p2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gpurandStateSobol32 state;
        gpurand_init(NULL, (unsigned int)idx, &state);
        arr[idx] = generate_normal_random<T>(&state, p1, p2);
    }
}

template <typename T>
__global__ void  iris_random_log_normal_seq_sobol_core(T *arr, unsigned long long seed, size_t size, T p1, T p2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gpurandStateSobol32 state;
        gpurand_init(NULL, (unsigned int)idx, &state);
        arr[idx] = generate_log_normal_random<T>(&state, p1, p2);
    }
}

template <typename T>
__global__ void  iris_arithmetic_seq_core(T *arr, T start, T increment, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = start + idx * increment;
    }
}

template <typename T>
__global__ void  iris_geometric_seq_core(T *arr, T start, T ratio, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_floating_point<T>::value) {
            arr[idx] = start * pow(ratio, idx);
        } else {
            T value = start;
            for(int i=0; i<idx; i++) {
                value *= ratio;
            }
            arr[idx] = value;
        }
    }
}

template <typename T>
void iris_reset(T *arr, T value, size_t size, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_reset_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, value, size);
    }
    else {
        iris_reset_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, value, size);
    }
}

template <typename T>
void iris_arithmetic_seq(T *arr, T start, T increment, size_t size, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_arithmetic_seq_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, start, increment, size);
    }
    else {
        iris_arithmetic_seq_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, start, increment, size);
    }
}

template <typename T>
void iris_geometric_seq(T *arr, T start, T ratio, size_t size, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_geometric_seq_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, start, ratio, size);
    }
    else {
        iris_geometric_seq_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, start, ratio, size);
    }
}

template <typename T>
void iris_random_uniform_sobol_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_random_uniform_seq_sobol_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, seed, size, p1, p2);
    }
    else {
        iris_random_uniform_seq_sobol_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, seed, size, p1, p2);
    }
}

template <typename T>
void iris_random_normal_sobol_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_random_normal_seq_sobol_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, seed, size, p1, p2);
    }
    else {
        iris_random_normal_seq_sobol_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, seed, size, p1, p2);
    }
}

template <typename T>
void iris_random_log_normal_sobol_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_random_log_normal_seq_sobol_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, seed, size, p1, p2);
    }
    else {
        iris_random_log_normal_seq_sobol_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, seed, size, p1, p2);
    }
}

template <typename T>
void iris_random_uniform_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    //printf("Size: %lu seed:%ld p1:%f p2:%f stream:%p\n", size, seed, (float)p1, (float)p2, stream);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_random_uniform_seq_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, seed, size, p1, p2);
    }
    else {
        iris_random_uniform_seq_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, seed, size, p1, p2);
    }
}

template <typename T>
void iris_random_normal_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_random_normal_seq_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, seed, size, p1, p2);
    }
    else {
        iris_random_normal_seq_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, seed, size, p1, p2);
    }
}

template <typename T>
void iris_random_log_normal_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
    if (stream != NULL) {
        gpuStream_t gpu_stream = reinterpret_cast<gpuStream_t>(stream);
        iris_random_log_normal_seq_core<T><<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(arr, seed, size, p1, p2);
    }
    else {
        iris_random_log_normal_seq_core<T><<<blocksPerGrid, threadsPerBlock>>>(arr, seed, size, p1, p2);
    }
}
#else // IS_DEFAULT_GPU
// CPU Implementation
template <typename T>
void iris_reset(T *arr, T value, size_t size, void *stream) {
    #pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = value;
    }
}

template <typename T>
void iris_random_uniform_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    std::mt19937 generator(seed+omp_get_thread_num());
    std::uniform_real_distribution<> dist(p1, p2);
    #pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = dist(generator);
    }
}

template <typename T>
void iris_random_normal_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    std::mt19937 generator(seed+omp_get_thread_num());
    std::normal_distribution<> dist(p1, p2);
    #pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = dist(generator);
    }
}

template <typename T>
void iris_random_log_normal_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    std::mt19937 generator(seed+omp_get_thread_num());
    std::lognormal_distribution<> dist(p1, p2);
    #pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = dist(generator);
    }
}

template <typename T>
void iris_random_uniform_sobol_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    fprintf(stderr, "[Error] Undefined openmp function: %s\n", __func__);
}

template <typename T>
void iris_random_normal_sobol_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    fprintf(stderr, "[Error] Undefined openmp function: %s\n", __func__);
}

template <typename T>
void iris_random_log_normal_sobol_seq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    fprintf(stderr, "[Error] Undefined openmp function: %s\n", __func__);
}

template <typename T>
void iris_arithmetic_seq(T *arr, T start, T increment, size_t size, void *stream) {
    #pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = start + i * increment;
    }
}

template <typename T>
void  iris_geometric_seq(T *arr, T start, T ratio, size_t size, void *stream) {
    #pragma omp parallel for
    for(int idx=0; idx<size; idx++) {
        T value = start;
        for(int i=0; i<idx; i++) {
            value *= ratio;
        }
        arr[idx] = value;
    }
}
template <>
void  iris_geometric_seq(float *arr, float start, float ratio, size_t size, void *stream) {
    #pragma omp parallel for
    for(int idx=0; idx<size; idx++) {
        arr[idx] = start * powf(ratio, idx);
    }
}
template <>
void  iris_geometric_seq(double *arr, double start, double ratio, size_t size, void *stream) {
    #pragma omp parallel for
    for(int idx=0; idx<size; idx++) {
        arr[idx] = start * pow(ratio, idx);
    }
}
#endif // IS_DEFAULT_GPU

#define IRIS_RANDOM_SEQ(RTYPE, T, M)   \
extern "C" void iris_random_ ## RTYPE ## _seq_ ## M(T *arr, unsigned long long seed, T p1, T p2, size_t size, void *stream) { \
    iris_random_ ## RTYPE ## _seq<T>(arr, seed, size, p1, p2, stream); \
}

IRIS_RANDOM_SEQ(log_normal_sobol, float,    f32)
IRIS_RANDOM_SEQ(log_normal_sobol, double,   f64)
IRIS_RANDOM_SEQ(log_normal_sobol, int64_t,  i64)
IRIS_RANDOM_SEQ(log_normal_sobol, int32_t,  i32)
IRIS_RANDOM_SEQ(log_normal_sobol, int16_t,  i16)
IRIS_RANDOM_SEQ(log_normal_sobol, int8_t,   i8)
IRIS_RANDOM_SEQ(log_normal_sobol, uint64_t, u64)
IRIS_RANDOM_SEQ(log_normal_sobol, uint32_t, u32)
IRIS_RANDOM_SEQ(log_normal_sobol, uint16_t, u16)
IRIS_RANDOM_SEQ(log_normal_sobol, uint8_t,  u8)

IRIS_RANDOM_SEQ(normal_sobol, float,    f32)
IRIS_RANDOM_SEQ(normal_sobol, double,   f64)
IRIS_RANDOM_SEQ(normal_sobol, int64_t,  i64)
IRIS_RANDOM_SEQ(normal_sobol, int32_t,  i32)
IRIS_RANDOM_SEQ(normal_sobol, int16_t,  i16)
IRIS_RANDOM_SEQ(normal_sobol, int8_t,   i8)
IRIS_RANDOM_SEQ(normal_sobol, uint64_t, u64)
IRIS_RANDOM_SEQ(normal_sobol, uint32_t, u32)
IRIS_RANDOM_SEQ(normal_sobol, uint16_t, u16)
IRIS_RANDOM_SEQ(normal_sobol, uint8_t,  u8)

IRIS_RANDOM_SEQ(uniform_sobol, float,    f32)
IRIS_RANDOM_SEQ(uniform_sobol, double,   f64)
IRIS_RANDOM_SEQ(uniform_sobol, int64_t,  i64)
IRIS_RANDOM_SEQ(uniform_sobol, int32_t,  i32)
IRIS_RANDOM_SEQ(uniform_sobol, int16_t,  i16)
IRIS_RANDOM_SEQ(uniform_sobol, int8_t,   i8)
IRIS_RANDOM_SEQ(uniform_sobol, uint64_t, u64)
IRIS_RANDOM_SEQ(uniform_sobol, uint32_t, u32)
IRIS_RANDOM_SEQ(uniform_sobol, uint16_t, u16)
IRIS_RANDOM_SEQ(uniform_sobol, uint8_t,  u8)

IRIS_RANDOM_SEQ(log_normal, float,    f32)
IRIS_RANDOM_SEQ(log_normal, double,   f64)
IRIS_RANDOM_SEQ(log_normal, int64_t,  i64)
IRIS_RANDOM_SEQ(log_normal, int32_t,  i32)
IRIS_RANDOM_SEQ(log_normal, int16_t,  i16)
IRIS_RANDOM_SEQ(log_normal, int8_t,   i8)
IRIS_RANDOM_SEQ(log_normal, uint64_t, u64)
IRIS_RANDOM_SEQ(log_normal, uint32_t, u32)
IRIS_RANDOM_SEQ(log_normal, uint16_t, u16)
IRIS_RANDOM_SEQ(log_normal, uint8_t,  u8)

IRIS_RANDOM_SEQ(normal, float,    f32)
IRIS_RANDOM_SEQ(normal, double,   f64)
IRIS_RANDOM_SEQ(normal, int64_t,  i64)
IRIS_RANDOM_SEQ(normal, int32_t,  i32)
IRIS_RANDOM_SEQ(normal, int16_t,  i16)
IRIS_RANDOM_SEQ(normal, int8_t,   i8)
IRIS_RANDOM_SEQ(normal, uint64_t, u64)
IRIS_RANDOM_SEQ(normal, uint32_t, u32)
IRIS_RANDOM_SEQ(normal, uint16_t, u16)
IRIS_RANDOM_SEQ(normal, uint8_t,  u8)

IRIS_RANDOM_SEQ(uniform, float,    f32)
IRIS_RANDOM_SEQ(uniform, double,   f64)
IRIS_RANDOM_SEQ(uniform, int64_t,  i64)
IRIS_RANDOM_SEQ(uniform, int32_t,  i32)
IRIS_RANDOM_SEQ(uniform, int16_t,  i16)
IRIS_RANDOM_SEQ(uniform, int8_t,   i8)
IRIS_RANDOM_SEQ(uniform, uint64_t, u64)
IRIS_RANDOM_SEQ(uniform, uint32_t, u32)
IRIS_RANDOM_SEQ(uniform, uint16_t, u16)
IRIS_RANDOM_SEQ(uniform, uint8_t,  u8)


#define IRIS_RESET(T, TAG)   \
extern "C" void iris_reset_ ## TAG(T *arr, T value, size_t size, void *stream) { \
    iris_reset<T>(arr, value, size, stream); \
}
IRIS_RESET(float,    f32)
IRIS_RESET(double,   f64)
IRIS_RESET(int64_t,  i64)
IRIS_RESET(int32_t,  i32)
IRIS_RESET(int16_t,  i16)
IRIS_RESET(int8_t,   i8)
IRIS_RESET(uint64_t, u64)
IRIS_RESET(uint32_t, u32)
IRIS_RESET(uint16_t, u16)
IRIS_RESET(uint8_t,  u8)

#define IRIS_ARITHMATIC_SEQ(T, TAG)   \
extern "C" void iris_arithmetic_seq_ ## TAG(T *arr, T start, T increment, size_t size, void *stream) { \
    iris_arithmetic_seq<T>(arr, start, increment, size, stream); \
}

IRIS_ARITHMATIC_SEQ(float,    f32)
IRIS_ARITHMATIC_SEQ(double,   f64)
IRIS_ARITHMATIC_SEQ(int64_t,  i64)
IRIS_ARITHMATIC_SEQ(int32_t,  i32)
IRIS_ARITHMATIC_SEQ(int16_t,  i16)
IRIS_ARITHMATIC_SEQ(int8_t,   i8)
IRIS_ARITHMATIC_SEQ(uint64_t, u64)
IRIS_ARITHMATIC_SEQ(uint32_t, u32)
IRIS_ARITHMATIC_SEQ(uint16_t, u16)
IRIS_ARITHMATIC_SEQ(uint8_t,  u8)

#define IRIS_GEOMETRIC_SEQ(T, TAG)   \
extern "C" void iris_geometric_seq_ ## TAG(T *arr, T start, T ratio, size_t size, void *stream) { \
    iris_geometric_seq<T>(arr, start, ratio, size, stream); \
}
IRIS_GEOMETRIC_SEQ(float,    f32)
IRIS_GEOMETRIC_SEQ(double,   f64)
IRIS_GEOMETRIC_SEQ(int64_t,  i64)
IRIS_GEOMETRIC_SEQ(int32_t,  i32)
IRIS_GEOMETRIC_SEQ(int16_t,  i16)
IRIS_GEOMETRIC_SEQ(int8_t,   i8)
IRIS_GEOMETRIC_SEQ(uint64_t, u64)
IRIS_GEOMETRIC_SEQ(uint32_t, u32)
IRIS_GEOMETRIC_SEQ(uint16_t, u16)
IRIS_GEOMETRIC_SEQ(uint8_t,  u8)

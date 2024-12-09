#ifdef __HIPCC__
#include <hip/hip_runtime.h>  // HIP header for HIPCC
#define gpuStream_t  hipStream_t
#else
#include <cuda.h>     // CUDA header for NVCC
#include <cuda_runtime.h>     // CUDA header for NVCC
#define gpuStream_t  CUstream //cudaStream_t
#endif
#include <stdint.h>
template <typename T>
__global__ void iris_reset_core(T *arr, T value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = value;
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
void  iris_arithmetic_seq(T *arr, T start, T increment, size_t size, void *stream) {
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

extern "C" void iris_reset_i64(int64_t *arr, int64_t value, size_t size, void *stream) { iris_reset<int64_t>(arr, value, size, stream); }
extern "C" void iris_reset_i32(int32_t *arr, int32_t value, size_t size, void *stream) { iris_reset<int32_t>(arr, value, size, stream); }
extern "C" void iris_reset_i16(int16_t *arr, int16_t value, size_t size, void *stream) { iris_reset<int16_t>(arr, value, size, stream); }
extern "C" void iris_reset_i8 (int8_t  *arr, int8_t value,  size_t size, void *stream) { iris_reset<int8_t> (arr, value, size, stream); }
extern "C" void iris_reset_u64(uint64_t *arr, uint64_t value, size_t size, void *stream) { iris_reset<uint64_t>(arr, value, size, stream); }
extern "C" void iris_reset_u32(uint32_t *arr, uint32_t value, size_t size, void *stream) { iris_reset<uint32_t>(arr, value, size, stream); }
extern "C" void iris_reset_u16(uint16_t *arr, uint16_t value, size_t size, void *stream) { iris_reset<uint16_t>(arr, value, size, stream); }
extern "C" void iris_reset_u8 (uint8_t  *arr, uint8_t value,  size_t size, void *stream) { iris_reset<uint8_t> (arr, value, size, stream); }
extern "C" void iris_reset_float(float *arr, float value, size_t size, void *stream) { iris_reset<float>(arr, value, size, stream); }
extern "C" void iris_reset_double(double *arr, double value, size_t size, void *stream) { iris_reset<double>(arr, value, size, stream); }


extern "C" void iris_arithmetic_seq_i64(int64_t *arr, int64_t start, int64_t increment, size_t size, void *stream) { iris_arithmetic_seq<int64_t>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_i32(int32_t *arr, int32_t start, int32_t increment, size_t size, void *stream) { iris_arithmetic_seq<int32_t>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_i16(int16_t *arr, int16_t start, int16_t increment, size_t size, void *stream) { iris_arithmetic_seq<int16_t>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_i8 (int8_t  *arr, int8_t start, int8_t increment,  size_t size, void *stream) { iris_arithmetic_seq<int8_t> (arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_u64(uint64_t *arr, uint64_t start, uint64_t increment, size_t size, void *stream) { iris_arithmetic_seq<uint64_t>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_u32(uint32_t *arr, uint32_t start, uint32_t increment, size_t size, void *stream) { iris_arithmetic_seq<uint32_t>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_u16(uint16_t *arr, uint16_t start, uint16_t increment, size_t size, void *stream) { iris_arithmetic_seq<uint16_t>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_u8 (uint8_t  *arr, uint8_t start, uint8_t increment,  size_t size, void *stream) { iris_arithmetic_seq<uint8_t> (arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_float(float *arr, float start, float increment, size_t size, void *stream) { iris_arithmetic_seq<float>(arr, start, increment, size, stream); }
extern "C" void iris_arithmetic_seq_double(double *arr, double start, double increment, size_t size, void *stream) { iris_arithmetic_seq<double>(arr, start, increment, size, stream); }

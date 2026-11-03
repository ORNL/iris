#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <iris/iris.h>

#ifdef __HIPCC__
#ifndef ENABLE_HIP_MACROS
#define ENABLE_HIP_MACROS
#endif
#endif // __HIPCC__

#ifdef __CUDACC__
#ifndef ENABLE_CUDA_MACROS
#define ENABLE_CUDA_MACROS
#endif // ENABLE_CUDA_MACROS
#endif // __CUDACC__

#ifdef ENABLE_HIP_MACROS
#define IRIS_HOST2HIP_KERNEL_ARGS     size_t _off, size_t _ndr
#define IRIS_HOST2HIP_KERNEL_BEGIN(i) for (i = _off; i < _off + _ndr; i++) {
#define IRIS_HOST2HIP_KERNEL_END      }
#include <hip/hip_runtime.h>  // HIP header for HIPCC
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
#endif // ENABLE_HIP_MACROS

#ifdef ENABLE_CUDA_MACROS
#include <cuda.h>     // CUDA header for NVCC
#include <cuda_runtime.h>     // CUDA header for NVCC
#include <curand_kernel.h>
#define IRIS_HOST2CUDA_KERNEL_ARGS     size_t _off, size_t _ndr
#define IRIS_HOST2CUDA_KERNEL_BEGIN(i) for (i = _off; i < _off + _ndr; i++) {
#define IRIS_HOST2CUDA_KERNEL_END      }
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
#endif // ENABLE_CUDA_MACROS

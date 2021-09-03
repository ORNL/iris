#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipError_t {
  hipSuccess = 0,
} hipError_t;

typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock,       ///< Maximum number of threads per block.
    hipDeviceAttributeMaxBlockDimX,             ///< Maximum x-dimension of a block.
    hipDeviceAttributeMaxBlockDimY,             ///< Maximum y-dimension of a block.
    hipDeviceAttributeMaxBlockDimZ,             ///< Maximum z-dimension of a block.
    hipDeviceAttributeMaxGridDimX,              ///< Maximum x-dimension of a grid.
    hipDeviceAttributeMaxGridDimY,              ///< Maximum y-dimension of a grid.
    hipDeviceAttributeMaxGridDimZ,              ///< Maximum z-dimension of a grid.
    hipDeviceAttributeMaxSharedMemoryPerBlock,  ///< Maximum shared memory available per block in
                                                ///< bytes.
    hipDeviceAttributeTotalConstantMemory,      ///< Constant memory size in bytes.
    hipDeviceAttributeWarpSize,                 ///< Warp size in threads.
    hipDeviceAttributeMaxRegistersPerBlock,  ///< Maximum number of 32-bit registers available to a
                                             ///< thread block. This number is shared by all thread
                                             ///< blocks simultaneously resident on a
                                             ///< multiprocessor.
    hipDeviceAttributeClockRate,             ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeMemoryClockRate,       ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeMemoryBusWidth,        ///< Global memory bus width in bits.
    hipDeviceAttributeMultiprocessorCount,   ///< Number of multiprocessors on the device.
    hipDeviceAttributeComputeMode,           ///< Compute mode that device is currently in.
    hipDeviceAttributeL2CacheSize,  ///< Size of L2 cache in bytes. 0 if the device doesn't have L2
                                    ///< cache.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,  ///< Maximum resident threads per
                                                    ///< multiprocessor.
    hipDeviceAttributeComputeCapabilityMajor,       ///< Major compute capability version number.
    hipDeviceAttributeComputeCapabilityMinor,       ///< Minor compute capability version number.
    hipDeviceAttributeConcurrentKernels,  ///< Device can possibly execute multiple kernels
                                          ///< concurrently.
    hipDeviceAttributePciBusId,           ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,        ///< PCI Device ID.
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,  ///< Maximum Shared Memory Per
                                                         ///< Multiprocessor.
    hipDeviceAttributeIsMultiGpuBoard,                   ///< Multiple GPU devices.
    hipDeviceAttributeIntegrated,                        ///< iGPU
    hipDeviceAttributeCooperativeLaunch,                 ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,      ///< Support cooperative launch on multiple devices

    hipDeviceAttributeMaxTexture1DWidth,    ///< Maximum number of elements in 1D images
    hipDeviceAttributeMaxTexture2DWidth,    ///< Maximum dimension width of 2D images in image elements
    hipDeviceAttributeMaxTexture2DHeight,   ///< Maximum dimension height of 2D images in image elements
    hipDeviceAttributeMaxTexture3DWidth,    ///< Maximum dimension width of 3D images in image elements
    hipDeviceAttributeMaxTexture3DHeight,   ///< Maximum dimensions height of 3D images in image elements
    hipDeviceAttributeMaxTexture3DDepth,    ///< Maximum dimensions depth of 3D images in image elements

    hipDeviceAttributeHdpMemFlushCntl,      ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,      ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register

    hipDeviceAttributeMaxPitch,             ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeTextureAlignment,     ///<Alignment requirement for textures
    hipDeviceAttributeKernelExecTimeout,    ///<Run time limit for kernels executed on the device
    hipDeviceAttributeCanMapHostMemory,     ///<Device can map host memory into device address space
    hipDeviceAttributeEccEnabled            ///<Device has ECC support enabled

} hipDeviceAttribute_t;

typedef int hipDevice_t;
typedef void* hipDeviceptr_t;

typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;

hipError_t hipInit(unsigned int flags);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipSetDevice(int deviceId);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra);
hipError_t hipDeviceSynchronize(void);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H

#ifdef __cplusplus
extern "C" {
#endif

#define hipDeviceScheduleAuto 0x0
typedef enum hipError_t {
  hipSuccess = 0,
} hipError_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,      ///< Host-to-Host Copy
    hipMemcpyHostToDevice = 1,    ///< Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,    ///< Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3,  ///< Device-to-Device Copy
    hipMemcpyDefault =
        4  ///< Runtime will automatically determine copy-kind based on virtual addresses.
} hipMemcpyKind;

//Flags that can be used with hipHostRegister.
/** Memory is Mapped and Portable.*/
#define hipHostRegisterDefault 0x0

/** Memory is considered registered by all contexts.*/
#define hipHostRegisterPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define hipHostRegisterMapped  0x2

 /** Not supported.*/
#define hipHostRegisterIoMemory 0x4

 /** Coarse Grained host memory lock.*/
#define hipExtHostRegisterCoarseGrained 0x8

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
hipError_t hipCtxGetCurrent(hipCtx_t *pctx);
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra);
hipError_t hipDeviceSynchronize(void);
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch_t;

typedef struct hipDeviceProp_t {
    char name[256];            ///< Device name.
    size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
    int regsPerBlock;          ///< Registers per block.
    int warpSize;              ///< Warp size.
    int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];      ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];        ///< Max grid dimensions (XYZ).
    int clockRate;             ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;       ///< Max global memory clock frequency in khz.
    int memoryBusWidth;        ///< Global memory bus width in bits.
    size_t totalConstMem;      ///< Size of shared memory region (in bytes).
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int multiProcessorCount;          ///< Number of multi-processors (compute units).
    int l2CacheSize;                  ///< L2 cache size.
    int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
    int computeMode;                  ///< Compute mode.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
    int concurrentKernels;     ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;           ///< PCI Domain ID
    int pciBusID;              ///< PCI Bus ID.
    int pciDeviceID;           ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
    int isMultiGpuBoard;                      ///< 1 if device is on a multi-GPU board, 0 if not.
    int canMapHostMemory;                     ///< Check whether HIP can map host memory
    int gcnArch;                              ///< DEPRECATED: use gcnArchName instead
    char gcnArchName[256];                    ///< AMD GCN Arch Name.
    int integrated;            ///< APU vs dGPU
    int cooperativeLaunch;            ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch; ///< HIP device supports cooperative launch on multiple devices
    int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
    int maxTexture1D;          ///< Maximum number of elements in 1D images
    int maxTexture2D[2];       ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture3D[3];       ///< Maximum dimensions (width, height, depth) of 3D images, in image elements
    unsigned int* hdpMemFlushCntl;      ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;      ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    size_t memPitch;                 ///<Maximum pitch in bytes allowed by memory copies
    size_t textureAlignment;         ///<Alignment requirement for textures
    size_t texturePitchAlignment;    ///<Pitch alignment requirement for texture references bound to pitched memory
    int kernelExecTimeoutEnabled;    ///<Run time limit for kernels executed on the device
    int ECCEnabled;                  ///<Device has ECC support enabled
    int tccDriver;                   ///< 1:If device is Tesla device using TCC driver, else 0
    int cooperativeMultiDeviceUnmatchedFunc;        ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;     ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;    ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;   ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched shared memories
    int isLargeBar;                  ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                ///< Revision of the GPU in this device
    int managedMemory;               ///< Device supports allocating managed memory on this system
    int directManagedMemAccessFromHost; ///< Host can directly access managed memory on the device without migration
    int concurrentManagedAccess;     ///< Device can coherently access managed memory concurrently with the CPU
    int pageableMemoryAccess;        ///< Device supports coherently accessing pageable memory
                                     ///< without calling hipHostRegister on it
    int pageableMemoryAccessUsesHostPageTables; ///< Device accesses pageable memory via the host's page tables
} hipDeviceProp_t;

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H */

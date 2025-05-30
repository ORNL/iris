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

//Flags that can be used with hipStreamCreateWithFlags.
/** Default stream creation flags. These are used with hipStreamCreate().*/
#define hipStreamDefault  0x00

/** Stream does not implicitly synchronize with null stream.*/
#define hipStreamNonBlocking 0x01

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

//Flags that can be used with hipEventCreateWithFlags.
/** Default flags.*/
#define hipEventDefault 0x0

/** Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.*/
#define hipEventBlockingSync 0x1

/** Disable event's capability to record timing information. May improve performance.*/
#define hipEventDisableTiming  0x2

/** Event can support IPC. Warnig: It is not supported in HIP.*/
#define hipEventInterprocess 0x4

/** Use a device-scope release when recording this event. This flag is useful to obtain more
 * precise timings of commands between events.  The flag is a no-op on CUDA platforms.*/
#define hipEventReleaseToDevice  0x40000000

/** Use a system-scope release when recording this event. This flag is useful to make
 * non-coherent host memory visible to the host. The flag is a no-op on CUDA platforms.*/
#define hipEventReleaseToSystem  0x80000000

typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                 ///< Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,///< Device can access host registered memory
                                                        ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,       ///< Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                    ///< Device can concurrently copy memory and execute a kernel.
                                                        ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                        ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,           ///< Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,        ///< Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                             ///< 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,               ///< Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,              ///< Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                        ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,               ///< Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,               ///< Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,              ///< Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,               ///< Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,               ///< Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                  ///< Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                ///< Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,         ///< Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,             ///< Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    hipDeviceAttributeUnused1,                          ///< Previously hipDeviceAttributeName
    hipDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                        ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                         ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,         ///< Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                        ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,        ///< Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,           ///< Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,       ///< Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,        ///< Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                 ///< Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUnused2,                          ///< Previously hipDeviceAttributeUuid
    hipDeviceAttributeWarpSize,                         ///< Warp size in threads.
    hipDeviceAttributeMemoryPoolsSupported,             ///< Device supports HIP Stream Ordered Memory Allocator
    hipDeviceAttributeVirtualMemoryManagementSupported, ///< Device supports HIP virtual memory management
    hipDeviceAttributeHostRegisterSupported,            ///< Can device support host memory registration via hipHostRegister
    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeUnused3,                                  ///< Previously hipDeviceAttributeArch
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeUnused4,                                  ///< Previously hipDeviceAttributeGcnArch
    hipDeviceAttributeUnused5,                                  ///< Previously hipDeviceAttributeGcnArchName
    hipDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                                ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeImageSupport,                             ///< '1' if Device supports image, '0' otherwise.
    hipDeviceAttributePhysicalMultiProcessorCount,              ///< All available physical compute
                                                                ///< units for the device
    hipDeviceAttributeFineGrainSupport,                         ///< '1' if Device supports fine grain, '0' otherwise
    hipDeviceAttributeWallClockRate,                            ///< Constant frequency of wall clock in kilohertz.

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
} hipDeviceAttribute_t;

typedef int hipDevice_t;
typedef void* hipDeviceptr_t;

typedef struct ihipEvent_t* hipEvent_t;
typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;

typedef enum hipPointer_attribute {
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1,   ///< The context on which a pointer was allocated
                                         ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,   ///< memory type describing location of a pointer
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,///< address at which the pointer is allocated on device
    HIP_POINTER_ATTRIBUTE_HOST_POINTER,  ///< address at which the pointer is allocated on host
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS,    ///< A pair of tokens for use with linux kernel interface
                                         ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,   ///< Synchronize every synchronous memory operation
                                         ///< initiated on this region
    HIP_POINTER_ATTRIBUTE_BUFFER_ID,     ///< Unique ID for an allocated memory region
    HIP_POINTER_ATTRIBUTE_IS_MANAGED,    ///< Indicates if the pointer points to managed memory
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,///< device ordinal of a device on which a pointer
                                         ///< was allocated or registered
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE, ///< if this pointer maps to an allocation
                                                     ///< that is suitable for hipIpcGetMemHandle
                                                     ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,///< Starting address for this requested pointer
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE,      ///< Size of the address range for this requested pointer
    HIP_POINTER_ATTRIBUTE_MAPPED,          ///< tells if this pointer is in a valid address range
                                           ///< that is mapped to a backing allocation
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,///< Bitmask of allowed hipmemAllocationHandleType
                                           ///< for this allocation @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE, ///< returns if the memory referenced by
                                           ///< this pointer can be used with the GPUDirect RDMA API
                                           ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,    ///< Returns the access flags the device associated with
                                           ///< for the corresponding memory referenced by the ptr
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE   ///< Returns the mempool handle for the allocation if
                                           ///< it was allocated from a mempool
                                           ///< @warning - not supported in HIP
} hipPointer_attribute;

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr);
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
#define hipDeviceMapHost 0x8
hipError_t hipSetDeviceFlags(unsigned flags);
hipError_t hipInit(unsigned int flags);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipSetDevice(int deviceId);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
hipError_t hipDeviceCanAccessPeer(int *canaccess, int device, int peerDevice);
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
hipError_t hipCtxDestroy(hipCtx_t ctx);
hipError_t hipDeviceReset(void);
hipError_t hipCtxGetCurrent(hipCtx_t *pctx);
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipMemset(hipDeviceptr_t ptr, int init, size_t size);
hipError_t hipMemsetAsync(hipDeviceptr_t ptr, int init, size_t size, hipStream_t stream);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFree(void* ptr);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra);
hipError_t hipDeviceSynchronize(void);
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
hipError_t hipEventQuery(hipEvent_t event);
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                        unsigned int flags);

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

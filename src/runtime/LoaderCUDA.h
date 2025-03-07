#ifndef IRIS_SRC_RT_LOADER_CUDA_H
#define IRIS_SRC_RT_LOADER_CUDA_H

#include "Loader.h"
#include <iris/cuda/cuda.h>

namespace iris {
namespace rt {

class LoaderCUDA : public Loader {
public:
  LoaderCUDA();
  ~LoaderCUDA();

  const char* library_precheck() { return "cuInit"; }
  const char* library();// { return "libcuda.so"; }
  int LoadFunctions();

  CUresult (*cuInit)(unsigned int Flags);
  CUresult (*cuDriverGetVersion)(int* driverVersion);
  CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
  CUresult (*cuDeviceGetAttribute)(int* pi, CUdevice_attribute attrib, CUdevice dev);
  CUresult (*cuDeviceGetCount)(int* count);
  CUresult (*cuDeviceGetName)(char* name, int len, CUdevice dev);
  CUresult (*cuGetErrorString)( CUresult error, const char** pStr );
  CUresult (*cuCtxGetCurrent)(CUcontext* pctx);
  CUresult (*cuCtxSetCurrent)(CUcontext ctx);
  CUresult (*cuCtxEnablePeerAccess)(CUcontext ctx, unsigned int flags);
  CUresult (*cudaDeviceReset)();
  CUresult (*cudaSetDevice)(int device);
  CUresult (*cudaDeviceCanAccessPeer)(int *canaccess, int device, int peerDevice);
  CUresult (*cuDeviceEnablePeerAccess)(int peerDevice, unsigned int flags);
  CUresult (*cudaDeviceEnablePeerAccess)(int peerDevice, unsigned int flags);
  CUresult (*cuCtxCreate)(CUcontext* pctx, unsigned int flags,CUdevice dev);
  CUresult (*cuCtxDestroy)(CUcontext pctx);
  CUresult (*cuCtxSynchronize)(void);
  CUresult (*cuStreamAddCallback)(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
  CUresult (*cuStreamCreate)(CUstream* phStream, unsigned int Flags);
  CUresult (*cuStreamQuery)(CUstream hStream);
  CUresult (*cuStreamDestroy)(CUstream hStream);
  CUresult (*cuStreamSynchronize)(CUstream hStream);
  CUresult (*cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);
  CUresult (*cuModuleLoad)(CUmodule* module, const char* fname);
  CUresult (*cuModuleUnload)(CUmodule module);
  CUresult (*cuModuleGetTexRef)(CUtexref* pTexRef, CUmodule hmod, const char* name);
  CUresult (*cuTexRefSetAddress)(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
  CUresult (*cuTexRefSetAddressMode)(CUtexref hTexRef, int dim, CUaddress_mode am);
  CUresult (*cuTexRefSetFilterMode)(CUtexref hTexRef, CUfilter_mode fm);
  CUresult (*cuTexRefSetFlags)(CUtexref hTexRef, unsigned int Flags);
  CUresult (*cuTexRefSetFormat)(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
  CUresult (*cudaGetSymbolAddress)( void** devPtr, const void* symbol );
  CUresult (*cudaMalloc)(CUdeviceptr* dptr, size_t bytesize);
  CUresult (*cuMemAllocAsync) ( CUdeviceptr* dptr, size_t bytesize, CUstream hStream );
  CUresult (*cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize);
  //CUresult (*cuMemset)(CUdeviceptr dptr, int init, size_t bytesize);
  CUresult (*cuMemFree)(CUdeviceptr dptr);
  CUresult (*cuMemFreeAsync)(CUdeviceptr dptr, CUstream hStream);
  CUresult (*cuMemcpyPeerAsync)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
  CUresult (*cuMemcpyPeer)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
  CUresult (*cuMemcpy2D)(const CUDA_MEMCPY2D *pCopy);
  CUresult (*cuMemHostRegister_v2)(void *p, size_t bytesize, unsigned int Flags);
  CUresult (*cuMemHostUnregister)(void *p);
  CUresult (*cudaHostRegister)(void *ptr, size_t size, unsigned int flags);
  CUresult (*cudaHostUnregister)(void *ptr);
  CUresult (*cudaMemset)(void *devPtr, int value, size_t count);
  CUresult (*cudaMemsetAsync)(void *devPtr, int value, size_t count, CUstream hStream);
  CUresult (*cudaMemcpy)( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
  CUresult (*cudaMemcpyAsync)( void* dst, const void* src, size_t count, cudaMemcpyKind kind, CUstream stream);
  CUresult (*cuMemcpy)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
  CUresult (*cuDeviceCanAccessPeer)(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
  CUresult (*cuPointerGetAttributes)(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);
  CUresult (*cuPointerGetAttribute)(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
  CUresult (*cuMemcpyAsync)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
  CUresult (*cudaMemcpy2D)( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind );
  CUresult (*cudaHostGetDevicePointer)(void **pDevice, void *pHost, unsigned int flags);  
  CUresult (*cudaSetDeviceFlags)(unsigned int flags);
  CUresult (*cudaMemcpy2DAsync)( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, CUstream stream);
  CUresult (*cuMemcpy2DUnaligned)(const CUDA_MEMCPY2D *pCopy);
  CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
  CUresult (*cuMemcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t size);
  CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t size, CUstream hStream);
  CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
  CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
  CUresult (*cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  CUresult (*cuMemcpyDtoHAsync)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
//  CUresult (*cuLaunchHostFunc)(CUstream hStream, CUhostFn fn, void *userData);
  CUresult (*cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
  CUresult (*cuEventCreate)(CUevent *phEvent, unsigned int Flags);
  CUresult (*cuEventRecord)(CUevent hEvent, CUstream hStream);
  CUresult (*cuEventDestroy)(CUevent hEvent);
  CUresult (*cuEventSynchronize)(CUevent hEvent);
  CUresult (*cuEventElapsedTime)(float *pMilliseconds, CUevent hStart, CUevent hEnd);
  CUresult (*cuEventQuery)(CUevent hEvent);
  CUresult (*cuStreamWaitEvent)(CUstream hStream, CUevent hEvent, unsigned int Flags);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_CUDA_H */


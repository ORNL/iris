#ifndef IRIS_SRC_RT_DEVICE_CUDA_H
#define IRIS_SRC_RT_DEVICE_CUDA_H

#include "Device.h"
#include "LoaderCUDA.h"
#include "LoaderHost2CUDA.h"
#include <map>

#define IRIS_MAX_GABAGES    256

namespace iris {
namespace rt {

class DeviceCUDA : public Device {
public:
  DeviceCUDA(LoaderCUDA* ld, LoaderHost2CUDA *host2cuda_ld, CUdevice cudev, int devno, int platform);
  ~DeviceCUDA();

  int Compile(char* src);
  int Init();
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  void MemCpy3D(CUdeviceptr dev, uint8_t *host, size_t *off, 
          size_t *dev_sizes, size_t *host_sizes, 
          size_t elem_size, bool host_2_dev);
  int MemH2D(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host);
  int MemD2H(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelLaunchInit(Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  int Custom(int tag, char* params);

  const char* kernel_src() { return "KERNEL_SRC_CUDA"; }
  const char* kernel_bin() { return "KERNEL_BIN_CUDA"; }

  virtual void TaskPre(Task* task);

  LoaderCUDA* ld() { return ld_; }
  LoaderHost2CUDA* host2cuda_ld() { return host2cuda_ld_; }
  CUmodule* module() { return &module_; }

private:
  static void Callback(CUstream stream, CUresult status, void* data);
  void ClearGarbage();

private:
  LoaderCUDA* ld_;
  LoaderHost2CUDA* host2cuda_ld_;
  CUdevice dev_;
  CUcontext ctx_;
  CUstream streams_[IRIS_MAX_DEVICE_NQUEUES];
  CUmodule module_;
  CUresult err_;
  unsigned int shared_mem_bytes_;
  unsigned int shared_mem_offs_[IRIS_MAX_KERNEL_NARGS];
  void* params_[IRIS_MAX_KERNEL_NARGS];
  int max_arg_idx_;
  CUdeviceptr garbage_[IRIS_MAX_GABAGES];
  int ngarbage_;
  std::map<CUfunction, CUfunction> kernels_offs_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_CUDA_H */


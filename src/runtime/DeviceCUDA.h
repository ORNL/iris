#ifndef BRISBANE_SRC_RT_DEVICE_CUDA_H
#define BRISBANE_SRC_RT_DEVICE_CUDA_H

#include "Device.h"
#include "LoaderCUDA.h"
#include <map>

#define BRISBANE_MAX_GABAGES    256

namespace brisbane {
namespace rt {

class DeviceCUDA : public Device {
public:
  DeviceCUDA(LoaderCUDA* ld, CUdevice cudev, int devno, int platform);
  ~DeviceCUDA();

  int Init();
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t off, size_t size, void* host);
  int MemD2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  int Custom(int tag, char* params);

  virtual void TaskPre(Task* task);

  LoaderCUDA* ld() { return ld_; }
  CUmodule* module() { return &module_; }

private:
  static void Callback(CUstream stream, CUresult status, void* data);
  void ClearGarbage();

private:
  LoaderCUDA* ld_;
  CUdevice dev_;
  CUcontext ctx_;
  CUstream streams_[BRISBANE_MAX_DEVICE_NQUEUES];
  CUmodule module_;
  CUresult err_;
  unsigned int shared_mem_bytes_;
  unsigned int shared_mem_offs_[BRISBANE_MAX_KERNEL_NARGS];
  void* params_[BRISBANE_MAX_KERNEL_NARGS];
  int max_arg_idx_;
  CUdeviceptr garbage_[BRISBANE_MAX_GABAGES];
  int ngarbage_;
  std::map<CUfunction, CUfunction> kernels_offs_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_CUDA_H */


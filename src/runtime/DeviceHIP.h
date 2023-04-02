#ifndef IRIS_SRC_RT_DEVICE_HIP_H
#define IRIS_SRC_RT_DEVICE_HIP_H

#include "Device.h"
#include "LoaderHIP.h"
#include "LoaderHost2HIP.h"
#include <map>

namespace iris {
namespace rt {
class Mem;
class BaseMem;
class DeviceHIP : public Device {
public:
  DeviceHIP(LoaderHIP* ld, LoaderHost2HIP *host2hip_ld, hipDevice_t cudev, int ordinal, int devno, int platform);
  ~DeviceHIP();

  int Compile(char* src);
  int Init();
  int ResetMemory(BaseMem *mem, uint8_t reset_value);
  void RegisterPin(void *host, size_t size);
  int MemAlloc(void** mem, size_t size, bool reset);
  int MemFree(void* mem);
  int MemD2D(Task *task, BaseMem *mem, void *dst, void *src, size_t size);
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  void CheckVendorSpecificKernel(Kernel* kernel);
  int KernelLaunchInit(Kernel* kernel);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  void EnablePeerAccess();
  void SetPeerDevices(int *peers, int count);
  int hipdev() { return dev_; }
  const char* kernel_src() { return "KERNEL_SRC_HIP"; }
  const char* kernel_bin() { return "KERNEL_BIN_HIP"; }
  void ResetContext();
  bool IsContextChangeRequired();
  void SetContextToCurrentThread();

private:
  LoaderHIP* ld_;
  LoaderHost2HIP* host2hip_ld_;
  hipCtx_t ctx_;
  hipDevice_t dev_;
  hipDevice_t peers_[IRIS_MAX_NDEVS];
  int peers_count_;
  hipModule_t module_;
  hipError_t err_;
  int ordinal_;
  int devid_;
  unsigned int shared_mem_bytes_;
  unsigned int shared_mem_offs_[IRIS_MAX_KERNEL_NARGS];
  void* params_[IRIS_MAX_KERNEL_NARGS];
  int max_arg_idx_;
  std::map<hipFunction_t, hipFunction_t> kernels_offs_;
  bool atleast_one_command_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_HIP_H */


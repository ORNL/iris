#ifndef IRIS_SRC_RT_DEVICE_HIP_H
#define IRIS_SRC_RT_DEVICE_HIP_H

#include "Device.h"
#include "LoaderHIP.h"
#include "LoaderHost2HIP.h"
#include <map>

namespace iris {
namespace rt {

class DeviceHIP : public Device {
public:
  DeviceHIP(LoaderHIP* ld, LoaderHost2HIP *host2hip_ld, hipDevice_t cudev, int ordinal, int devno, int platform);
  ~DeviceHIP();

  int Compile(char* src);
  int Init();
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t off, size_t size, void* host);
  int MemD2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunchInit(Kernel* kernel);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);

  const char* kernel_src() { return "KERNEL_SRC_HIP"; }
  const char* kernel_bin() { return "KERNEL_BIN_HIP"; }

private:
  LoaderHIP* ld_;
  LoaderHost2HIP* host2hip_ld_;
  hipDevice_t dev_;
  hipModule_t module_;
  hipError_t err_;
  int ordinal_;
  int devid_;
  unsigned int shared_mem_bytes_;
  unsigned int shared_mem_offs_[IRIS_MAX_KERNEL_NARGS];
  void* params_[IRIS_MAX_KERNEL_NARGS];
  int max_arg_idx_;
  std::map<hipFunction_t, hipFunction_t> kernels_offs_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_HIP_H */


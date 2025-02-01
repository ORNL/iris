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
  DeviceHIP(LoaderHIP* ld, LoaderHost2HIP *host2hip_ld, hipDevice_t cudev, int ordinal, int devno, int platform, int local_devno);
  ~DeviceHIP();

  int Compile(char* src, const char *out=NULL, const char *flags=NULL);
  int Init();
  int ResetMemory(Task *task, Command *cmd, BaseMem *mem);
  void RegisterPin(void *host, size_t size);
  void UnRegisterPin(void *host);
  void set_can_share_host_memory_flag(bool flag);
  void *GetSharedMemPtr(void* mem, size_t size);
  bool IsD2DPossible(Device *target);
  int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset);
  int MemFree(BaseMem *mem, void* mem_addr);
  int MemD2D(Task *task, Device *src_dev, BaseMem *mem, void *dst, void *src, size_t size);
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  void CheckVendorSpecificKernel(Kernel* kernel);
  int KernelLaunchInit(Command *cmd, Kernel* kernel);
  void VendorKernelLaunch(void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  void EnablePeerAccess();
  int RegisterCallback(int stream, CallBackType callback_fn, void* data, int flags=0);
  void SetPeerDevices(int *peers, int count);
  int hipdev() { return dev_; }
  const char* kernel_src() { return "KERNEL_SRC_HIP"; }
  const char* kernel_bin() { return "KERNEL_BIN_HIP"; }
  void ResetContext();
  bool IsContextChangeRequired();
  void SetContextToCurrentThread();
  bool IsAddrValidForD2D(BaseMem *mem, void *ptr);
  float GetEventTime(void *event, int stream);
  void CreateEvent(void **event, int flags);
  void RecordEvent(void **event, int stream, int event_creation_flag=iris_event_disable_timing);
  void WaitForEvent(void *event, int stream, int flags=0);
  void DestroyEvent(void *event);
  void EventSynchronize(void *event);
  void *get_ctx() { return (void *)&ctx_; }
  void *GetSymbol(const char *name)  { 
      ASSERT(ld_ != NULL); 
      void *ptr = ld_->GetSymbol(name); 
      if (ptr == NULL) 
          ptr = host2hip_ld_->GetSymbol(name);
      return ptr;
  }

private:
  LoaderHIP* ld_;
  LoaderHost2HIP* host2hip_ld_;
  hipCtx_t ctx_;
  hipDevice_t dev_;
  hipDevice_t peers_[IRIS_MAX_NDEVS];
  hipStream_t *streams_;//[IRIS_MAX_DEVICE_NQUEUES];
  hipEvent_t single_start_time_event_;
  //hipEvent_t start_time_event_[IRIS_MAX_DEVICE_NQUEUES];
  int peers_count_;
  hipModule_t module_;
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


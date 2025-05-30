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
  DeviceCUDA(LoaderCUDA* ld, LoaderHost2CUDA *host2cuda_ld, CUdevice cudev, int ordinal, int devno, int platform, int local_devno);
  ~DeviceCUDA();

  void set_can_share_host_memory_flag(bool flag=true); 
  int Compile(char* src, const char *out=NULL, const char *flags=NULL);
  int Init();
  int ResetMemory(Task *task, Command *cmd, BaseMem *mem);
  void *GetSharedMemPtr(void* mem, size_t size);
  int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset=false);
  int MemFree(BaseMem *mem, void* mem_addr);
  void RegisterPin(void *host, size_t size);
  void UnRegisterPin(void *host);
  int RegisterCallback(int stream, CallBackType callback_fn, void* data, int flags=0);
  void EnablePeerAccess();
  void SetPeerDevices(int *peers, int count);
  bool IsD2DPossible(Device *target);
  int CheckPinnedMemory(void* ptr);
  void MemCpy3D(CUdeviceptr dev, uint8_t *host, size_t *off, 
          size_t *dev_sizes, size_t *host_sizes, 
          size_t elem_size, bool host_2_dev);
  bool IsAddrValidForD2D(BaseMem *mem, void *ptr);
  int MemD2D(Task *task, Device *src_dev, BaseMem *mem, void *dst, void *src, size_t size);
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  void CheckVendorSpecificKernel(Kernel* kernel);
  int KernelLaunchInit(Command *cmd, Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  void VendorKernelLaunch(void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params);
  int Synchronize();
  int Custom(int tag, char* params);
  float GetEventTime(void *event, int stream);
  void CreateEvent(void **event, int flags);
  void RecordEvent(void **event, int stream, int event_creation_flag=iris_event_disable_timing);
  void WaitForEvent(void *event, int stream, int flags=0);
  void DestroyEvent(void *event);
  void EventSynchronize(void *event);
  void *get_ctx() { return (void *)&ctx_; }
  void *get_stream(int index) { return (void *)&streams_[index]; }
  void *GetSymbol(const char *name);

  const char* kernel_src() { return "KERNEL_SRC_CUDA"; }
  const char* kernel_bin() { return "KERNEL_BIN_CUDA"; }

  virtual void TaskPre(Task* task);

  LoaderCUDA* ld() { return ld_; }
  LoaderHost2CUDA* host2cuda_ld() { return host2cuda_ld_; }
  CUmodule* module() { return &module_; }
  int cudev() { return dev_; }
  void ResetContext();
  bool IsContextChangeRequired();
  void SetContextToCurrentThread();

private:
  void ClearGarbage();

private:
  LoaderCUDA* ld_;
  LoaderHost2CUDA* host2cuda_ld_;
  CUdevice dev_;
  CUdevice peers_[IRIS_MAX_NDEVS];
  int peers_count_;
  CUcontext ctx_;
  CUstream *streams_; //[IRIS_MAX_DEVICE_NQUEUES];
  CUevent  single_start_time_event_;
  //CUevent  start_time_event_[IRIS_MAX_DEVICE_NQUEUES];
  CUmodule module_;
  unsigned int shared_mem_bytes_;
  unsigned int shared_mem_offs_[IRIS_MAX_KERNEL_NARGS];
  void* params_[IRIS_MAX_KERNEL_NARGS];
  int max_arg_idx_;
  CUdeviceptr garbage_[IRIS_MAX_GABAGES];
  int ngarbage_;
  std::map<CUfunction, CUfunction> kernels_offs_;
  int ordinal_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_CUDA_H */


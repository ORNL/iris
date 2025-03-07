#ifndef IRIS_SRC_RT_DEVICE_OPENCL_H
#define IRIS_SRC_RT_DEVICE_OPENCL_H

#include "Device.h"
#include "LoaderOpenCL.h"
#include "LoaderHost2OpenCL.h"
#include "Timer.h"
#include <string>

namespace iris {
namespace rt {

class Timer;

class DeviceOpenCL : public Device {
public:
  DeviceOpenCL(LoaderOpenCL* ld, LoaderHost2OpenCL *host2opencl_ld, cl_device_id cldev, cl_context clctx, int devno, int ocldevno, int platform);
  ~DeviceOpenCL();

  int Init();
  bool IsDeviceValid();
  int BuildProgram(char* path);
  int ResetMemory(Task *task, Command *cmd, BaseMem *mem);
  int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset=false);
  int MemFree(BaseMem *mem, void* mem_addr);
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int KernelLaunchInit(Command *cmd, Kernel* kernel);
  void CheckVendorSpecificKernel(Kernel *kernel);
  int Synchronize();
  //int AddCallback(Task* task);
  int RecreateContext();
  //void ExecuteKernel(Command* cmd);
  static std::string GetLoaderHost2OpenCLSuffix(LoaderOpenCL *ld, cl_device_id cldev);
  bool SupportJIT() { return false; }

private:
  int RegisterCallback(int stream, CallBackType callback_fn, void* data, int flags=0);
  int CreateProgram(const char* suffix, char** src, size_t* srclen);
  void CreateEvent(void **event, int flags);
  void RecordEvent(void **event, int stream, int event_creation_flag=iris_event_disable_timing);
  void WaitForEvent(void *event, int stream, int flags);
  void DestroyEvent(void *event);
  float GetEventTime(void *event, int stream);
  void EventSynchronize(void *event);
  void *GetSymbol(const char *name)  { 
      ASSERT(ld_ != NULL); 
      return ld_->GetSymbol(name); 
  }
private:
  LoaderOpenCL* ld_;
  LoaderHost2OpenCL *host2opencl_ld_;
  Timer* timer_;
  cl_device_id cldev_;
  cl_context clctx_;
  cl_event single_start_time_event_;
  cl_command_queue *clcmdq_; //[IRIS_MAX_DEVICE_NQUEUES];
  cl_command_queue default_queue_;
  cl_program clprog_;
  cl_device_type cltype_;
  cl_bool compiler_available_;
  int ocldevno_;
  std::string fpga_bin_suffix_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_OPENCL_H */


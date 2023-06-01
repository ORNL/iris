#ifndef IRIS_SRC_RT_DEVICE_H
#define IRIS_SRC_RT_DEVICE_H

#include "Debug.h"
#include "Config.h"
#include "Timer.h"
#include <map>

#ifndef IRIS_ASYNC_STREAMING
#define IRIS_SYNC_EXECUTION
#endif //IRIS_ASYNC_STREAMING

namespace iris {
namespace rt {

class Command;
class Kernel;
class BaseMem;
class Mem;
class DataMem;
class DataMemRegion;
class Task;
class Timer;
class Worker;

class Device {
public:
  Device(int devno, int platform);
  virtual ~Device();

  virtual void TaskPre(Task* task) { return; }
  virtual void TaskPost(Task* task) { return; }

  void Execute(Task* task);

  void ExecuteInit(Command* cmd);
  virtual void ExecuteKernel(Command* cmd);
  virtual void RegisterPin(void *host, size_t size) { }
  void ExecuteMalloc(Command* cmd);
  void RegisterHost(BaseMem *mem);
  template <typename DMemType>
  void InvokeDMemInDataTransfer(Task *task, Command *cmd, DMemType *mem);
  void ExecuteMemResetInput(Task *task, Command* cmd);
  void ExecuteMemIn(Task *task, Command* cmd);
  //void ExecuteMemInExternal(Command *cmd);
  void ExecuteMemInDMemIn(Task *task, Command* cmd, DataMem *mem);
  void ExecuteMemInDMemRegionIn(Task *task, Command* cmd, DataMemRegion *mem);
  void ExecuteMemOut(Task *task, Command* cmd);
  void ExecuteMemFlushOut(Command* cmd);

  void ExecuteD2D(Command* cmd, Device *dev=NULL);
  void ExecuteH2D(Command* cmd, Device *dev=NULL);
  void ExecuteH2BroadCast(Command* cmd);
  void ExecuteH2DNP(Command* cmd);
  void ExecuteD2H(Command* cmd);
  void ExecuteMap(Command* cmd);
  void ExecuteReleaseMem(Command* cmd);
  void ExecuteHost(Command* cmd);
  void ExecuteCustom(Command* cmd);

  Kernel* ExecuteSelectorKernel(Command* cmd);

  void GetPossibleDevices(int devno, int *nddevs, 
          int &d2d_dev, int &cpu_dev, int &non_cpu_dev);
  int RegisterCommand(int tag, command_handler handler);
  int RegisterHooks();

  virtual int ResetMemory(BaseMem *mem, uint8_t reset_value)=0;
  virtual void ResetContext() { }
  virtual bool IsContextChangeRequired() { return false; }
  virtual int Compile(char* src) { return IRIS_SUCCESS; }
  virtual int Init() = 0;
  virtual int BuildProgram(char* path) { return IRIS_SUCCESS; }
  virtual int MemAlloc(void** mem, size_t size, bool reset=false) = 0;
  virtual int MemFree(void* mem) = 0;
  virtual int MemD2D(Task *task, BaseMem *mem, void *dst, void *src, size_t size) { _error("Device:%d:%s doesn't support MemD2D", devno_, name()); return IRIS_ERROR; }
  virtual int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="") = 0;
  virtual int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="") = 0;
  virtual int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true) = 0;
  virtual int KernelLaunchInit(Kernel* kernel) { return IRIS_SUCCESS; }
  virtual void CheckVendorSpecificKernel(Kernel *kernel) { }
  virtual int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) = 0;
  virtual int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) = 0;
  virtual int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) = 0;
  virtual int Synchronize() = 0;
  virtual int AddCallback(Task* task) = 0;
  virtual int Custom(int tag, char* params) { return IRIS_SUCCESS; }
  virtual int RecreateContext() { return IRIS_ERROR; }
  virtual bool SupportJIT() { return true; }
  virtual void SetPeerDevices(int *peers, int count) { }
  virtual const char* kernel_src() { return " "; }
  virtual const char* kernel_bin() { return " "; }

  void set_shared_memory_buffers(bool flag=true) { shared_memory_buffers_ = flag; }
  bool is_shared_memory_buffers() { return shared_memory_buffers_ && can_share_host_memory_; }
  int platform() { return platform_; }
  int devno() { return devno_; }
  int type() { return type_; }
  int model() { return model_; }
  char* vendor() { return vendor_; }
  char* name() { return name_; }
  bool busy() { return busy_; }
  bool idle() { return !busy_; }
  bool enable() { return enable_; }
  bool native_kernel_not_exists() { return native_kernel_not_exists_; }
  void enableD2D() { is_d2d_possible_ = true; }
  bool isD2DEnabled() { return is_d2d_possible_; }
  int ok() { return errid_; }
  void set_worker(Worker* worker) { worker_ = worker; }
  Worker* worker() { return worker_; }
  double Now() { return timer_->Now(); }
protected:
  int devno_;
  int platform_;
  int type_;
  int model_;
  char vendor_[128];
  char name_[256];
  char version_[64];
  int driver_version_;
  size_t max_compute_units_;
  size_t max_work_group_size_;
  size_t max_work_item_sizes_[3];
  int max_block_dims_[3];
  int nqueues_;
  int q_;
  int errid_;

  char kernel_path_[256];

  bool busy_;
  bool enable_;
  bool shared_memory_buffers_;
  bool can_share_host_memory_;
  bool is_d2d_possible_;
  bool native_kernel_not_exists_;

  Worker* worker_;
  Timer* timer_;
  hook_task hook_task_pre_;
  hook_task hook_task_post_;
  hook_command hook_command_pre_;
  hook_command hook_command_post_;

  std::map<int, command_handler> cmd_handlers_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_H */

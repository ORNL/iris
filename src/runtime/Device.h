#ifndef IRIS_SRC_RT_DEVICE_H
#define IRIS_SRC_RT_DEVICE_H

#include "Config.h"
#include <map>

#define IRIS_SYNC_EXECUTION

namespace iris {
namespace rt {

class Command;
class Kernel;
class Mem;
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
  void ExecuteMalloc(Command* cmd);
  void ExecuteH2D(Command* cmd);
  void ExecuteH2DNP(Command* cmd);
  void ExecuteD2H(Command* cmd);
  void ExecuteMap(Command* cmd);
  void ExecuteReleaseMem(Command* cmd);
  void ExecuteHost(Command* cmd);
  void ExecuteCustom(Command* cmd);

  Kernel* ExecuteSelectorKernel(Command* cmd);

  int RegisterCommand(int tag, command_handler handler);
  int RegisterHooks();

  virtual int Compile(char* src) { return IRIS_SUCCESS; }
  virtual int Init() = 0;
  virtual int BuildProgram(char* path) { return IRIS_SUCCESS; }
  virtual int MemAlloc(void** mem, size_t size) = 0;
  virtual int MemFree(void* mem) = 0;
  virtual int MemH2D(Mem* mem, size_t off, size_t size, void* host) = 0;
  virtual int MemD2H(Mem* mem, size_t off, size_t size, void* host) = 0;
  virtual int KernelGet(void** kernel, const char* name) = 0;
  virtual int KernelLaunchInit(Kernel* kernel) { return IRIS_SUCCESS; }
  virtual int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) = 0;
  virtual int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) = 0;
  virtual int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) = 0;
  virtual int Synchronize() = 0;
  virtual int AddCallback(Task* task) = 0;
  virtual int Custom(int tag, char* params) { return IRIS_SUCCESS; }
  virtual int RecreateContext() { return IRIS_ERROR; }
  virtual bool SupportJIT() { return true; }
  virtual const char* kernel_src() { return " "; }
  virtual const char* kernel_bin() { return " "; }

  void set_shared_memory_buffers(bool flag=true) { shared_memory_buffers_ = flag; }
  bool is_shared_memory_buffers() { return shared_memory_buffers_; }
  void set_vendor_specific_kernel(bool flag=true) { is_vendor_specific_kernel_ = flag; }
  bool is_vendor_specific_kernel() { return is_vendor_specific_kernel_; }
  int platform() { return platform_; }
  int devno() { return devno_; }
  int type() { return type_; }
  int model() { return model_; }
  char* vendor() { return vendor_; }
  char* name() { return name_; }
  bool busy() { return busy_; }
  bool idle() { return !busy_; }
  bool enable() { return enable_; }
  int ok() { return errid_; }
  void set_worker(Worker* worker) { worker_ = worker; }
  Worker* worker() { return worker_; }

protected:
  int devno_;
  int platform_;
  int type_;
  int model_;
  char vendor_[64];
  char name_[64];
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
  bool is_vendor_specific_kernel_;

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

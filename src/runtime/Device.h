#ifndef BRISBANE_SRC_RT_DEVICE_H
#define BRISBANE_SRC_RT_DEVICE_H

#include "Config.h"
#include <map>

#define BRISBANE_SYNC_EXECUTION

namespace brisbane {
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
  void ExecuteKernel(Command* cmd);
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

  virtual int Init() = 0;
  virtual int BuildProgram(char* path) { return BRISBANE_OK; }
  virtual int MemAlloc(void** mem, size_t size) = 0;
  virtual int MemFree(void* mem) = 0;
  virtual int MemH2D(Mem* mem, size_t off, size_t size, void* host) = 0;
  virtual int MemD2H(Mem* mem, size_t off, size_t size, void* host) = 0;
  virtual int KernelGet(void** kernel, const char* name) = 0;
  virtual int KernelLaunchInit(Kernel* kernel) { return BRISBANE_OK; }
  virtual int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) = 0;
  virtual int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) = 0;
  virtual int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) = 0;
  virtual int Synchronize() = 0;
  virtual int AddCallback(Task* task) = 0;
  virtual int Custom(int tag, char* params) { return BRISBANE_OK; }
  virtual int RecreateContext() { return BRISBANE_ERR; }

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

  bool busy_;
  bool enable_;

  Worker* worker_;
  Timer* timer_;
  hook_task hook_task_pre_;
  hook_task hook_task_post_;
  hook_command hook_command_pre_;
  hook_command hook_command_post_;

  std::map<int, command_handler> cmd_handlers_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_H */

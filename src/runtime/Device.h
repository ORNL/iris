#ifndef IRIS_SRC_RT_DEVICE_H
#define IRIS_SRC_RT_DEVICE_H

#include "Debug.h"
#include "Config.h"
#include "Timer.h"
//#include "CPUEvent.h"
#include <map>
#include <vector>
#include <mutex>
#include <atomic>
using namespace std;

//TODO:
#define ENABLE_SAME_TYPE_GPU_OPTIMIZATION
#define DIRECT_H2D_SYNC
//#define DISABLE_D2D

#define DEFAULT_STREAM_INDEX -2

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
class Platform;
class JuliaHostInterfaceLoader;
class LoaderDefaultKernel;

enum AsyncResolveType
{
    //ASYNC_GENERIC_RESOLVE,
    ASYNC_D2D_RESOLVE,
    ASYNC_DEV_INPUT_RESOLVE,
    ASYNC_D2H_SYNC,
    ASYNC_D2O_SYNC,
    ASYNC_KNOWN_H2D_RESOLVE,
    ASYNC_UNKNOWN_H2D_RESOLVE,
    ASYNC_SAME_DEVICE_DEPENDENCY,
    //ASYNC_H2D_RESOLVE_SYNC,
};
typedef void (*CallBackType)(void *stream, int status, void *data);
class Device {
public:
  Device(int devno, int platform);
  virtual ~Device();

  virtual void TaskPre(Task* task) { return; }
  virtual void TaskPost(Task* task) { return; }

  void Execute(Task* task);

  void ExecuteInit(Command* cmd);
  virtual void ExecuteKernel(Command* cmd);
  void TrackDestroyEvent(void *event) { 
      //printf("Adding event:%p size:%ld obj:%p\n", event, destroy_events_.size(), &destroy_events_);
      destroy_events_mutex_.lock();
      destroy_events_[event] = false;
      destroy_events_mutex_.unlock();
      //printf("Added event:%p size:%ld\n", event, destroy_events_.size());
  }
  void EnableDestroyEvent(void *event) { 
      //printf("Adding event:%p size:%ld obj:%p\n", event, destroy_events_.size(), &destroy_events_);
      destroy_events_[event] = true;
      //printf("Added event:%p size:%ld\n", event, destroy_events_.size());
  }
  void FreeDestroyEvents() 
  {
      //printf("FreeDestroyEvents size:%ld obj:%p\n", destroy_events_.size(), &destroy_events_);
      destroy_events_mutex_.lock();
      for(auto it = destroy_events_.begin(); it != destroy_events_.end(); ) {
          if (it->second) {
              //printf("Destroying event:%p\n", it->first);
              DestroyEvent(it->first);
              it = destroy_events_.erase(it);
          }
          else
              ++it;
      }
      destroy_events_mutex_.unlock();
  }
  virtual void RegisterPin(void *host, size_t size) { }
  virtual void UnRegisterPin(void *host) { }
  void ExecuteMalloc(Command* cmd);
  void RegisterHost(BaseMem *mem);
  virtual float GetEventTime(void *event, int stream) { return 0.0f; }
  virtual void CreateEvent(void **event, int flags);
  virtual void RecordEvent(void **event, int stream, int event_creation_flag=iris_event_disable_timing);
  virtual void WaitForEvent(void *event, int stream, int flags=0);
  virtual void DestroyEvent(void *event);
  virtual void DestroyEvent(BaseMem *mem, void *event);
  virtual void EventSynchronize(void *event);
  virtual void EnablePeerAccess() { }
  virtual bool IsD2DPossible(Device *target) { return true; }
  void ProactiveTransfers(Task *task, Command *cmd);
  void WaitForTaskInputAvailability(int devno, Task *task, Command *cmd);
  template <typename DMemType>
  void WaitForDataAvailability(int devno, Task *task, DMemType *mem, int read_stream=-1);
  template <typename DMemType>
  void InvokeDMemInDataTransfer(Task *task, Command *cmd, DMemType *mem, BaseMem *parent=NULL, DMemType *src_mem=NULL);
  void ExecuteMemResetInput(Task *task, Command* cmd);
  void ExecuteMemIn(Task *task, Command* cmd);
  //void ExecuteMemInExternal(Command *cmd);
  void ExecuteMemInDMemIn(Task *task, Command* cmd, DataMem *mem);
  void ExecuteMemInDMemRegionIn(Task *task, Command* cmd, DataMemRegion *mem);
  void ExecuteMemOut(Task *task, Command* cmd);
  void ExecuteMemFlushOut(Command* cmd);
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
  void ExecuteMemFlushOutToShadow(Command* cmd);
#endif
#endif
  void HandleHiddenDMemIns(Task *task);
  void HandleHiddenDMemOuts(Task *task);
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

  void ResolveH2DStartEvents(Task *task, BaseMem *mem, bool async, BaseMem *src_mem=NULL);
  void ResolveH2DEndEvents(Task *task, BaseMem *mem, bool async);
  void ResolveDeviceWrite(Task *task, BaseMem *mem, Device *input_dev, bool instant_wait, BaseMem *src_mem=NULL);
  template <AsyncResolveType resolve_type>
  inline void ResolveInputWriteDependency(Task *task, BaseMem *mem, bool async, Device *select_src_dev=NULL, BaseMem *src_mem=NULL);
  template <AsyncResolveType resolve_type>
  inline void ResolveOutputWriteDependency(Task *task, BaseMem *mem, bool async, Device *select_src_dev);
  inline void DeviceEventExchange(Task *task, BaseMem *mem, void *input_event, int input_stream, Device *input_dev);
  void SynchronizeInputToMemory(Task *task, BaseMem *mem);
  void GetPossibleDevices(BaseMem *mem, int devno, int *nddevs, 
          int &d2d_dev, int &cpu_dev, int &non_cpu_dev, bool async);
  virtual int RegisterCallback(int stream, CallBackType callback_fn, void *data, int flags=0);
  int RegisterCommand(int tag, command_handler handler);
  int RegisterHooks();
  void ExecuteDMEM2DMEM(Task *task, Command *cmd);

  virtual int ResetMemory(Task *task, Command *cmd, BaseMem *mem)=0;
  virtual void ResetContext() { }
  virtual void SetContextToCurrentThread() { }
  virtual bool IsContextChangeRequired() { return false; }
  virtual bool IsDeviceValid() { return true; }
  virtual int Compile(char* src, const char *out=NULL, const char *flags=NULL) { return IRIS_SUCCESS; }
  virtual int Init() = 0;
  virtual int BuildProgram(char* path) { return IRIS_SUCCESS; }
  virtual void *GetSharedMemPtr(void* mem, size_t size) { return mem; }
  virtual int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset=false) = 0;
  virtual int MemFree(BaseMem *mem, void* mem_addr) = 0;
  virtual int MemD2D(Task *task, Device *src_dev, BaseMem *mem, void *dst, void *src, size_t size) { _error("Device:%d:%s doesn't support MemD2D", devno_, name()); return IRIS_ERROR; }
  virtual int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="") = 0;
  virtual int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="") = 0;
  virtual int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true) = 0;
  virtual int KernelLaunchInit(Command *cmd, Kernel* kernel) { return IRIS_SUCCESS; }
  virtual void CheckVendorSpecificKernel(Kernel *kernel) { }
  virtual int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) = 0;
  virtual int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) = 0;
  virtual int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) = 0;
  virtual void VendorKernelLaunch(void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params) { }
  virtual int Synchronize() = 0;
  virtual int AddCallback(Task* task);
  static void Callback(void *stream, int status, void* data);
  virtual int Custom(int tag, char* params) { return IRIS_SUCCESS; }
  virtual int RecreateContext() { return IRIS_ERROR; }
  virtual bool SupportJIT() { return true; }
  virtual void SetPeerDevices(int *peers, int count) { }
  virtual bool IsAddrValidForD2D(BaseMem *mem, void *ptr) { return true; }
  virtual const char* kernel_src() { return " "; }
  virtual const char* kernel_bin() { return " "; }
  virtual void *GetSymbol(const char *name) { return NULL; }
  void set_shared_memory_buffers(bool flag=true) { shared_memory_buffers_ = flag; }
  virtual void set_can_share_host_memory_flag(bool flag=true) { 
      // We leave this decision to device specific
      // By default it still go with default can_share_host_memory_ 
  }
  bool is_shared_memory_buffers() { return shared_memory_buffers_ && can_share_host_memory_; }
  void set_async(bool flag=true) { async_ = flag; }
  template <class Task> bool is_async(Task *task, bool stream_policy_check=true) { 
      return is_async(false) && task->is_async() && 
          (!stream_policy_check || 
           (stream_policy(task) != STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL)); 
  }
  bool is_async(bool stream_policy_check=true) { 
      return async_ && (!stream_policy_check || 
              stream_policy() != STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL); 
  }
  void set_root_device(Device *root) { root_dev_ = root; }
  double first_event_cpu_end_time() { return first_event_cpu_end_time_; }
  double first_event_cpu_begin_time() { return first_event_cpu_begin_time_; }
  double first_event_cpu_mid_point_time() { return first_event_cpu_mid_point_time_; }
  void set_first_event_cpu_end_time(double time) { 
      first_event_cpu_end_time_ = time; 
      first_event_cpu_mid_point_time_ = first_event_cpu_begin_time_ + (first_event_cpu_end_time_-first_event_cpu_begin_time_)/2.0f;
  }
  void set_first_event_cpu_begin_time(double time) { first_event_cpu_begin_time_ = time; }
  void EnableJuliaInterface(); 
  bool IsFree();
  int active_tasks() { return active_tasks_; }
  void FreeActiveTask() { active_tasks_--; }
  void ReserveActiveTask() { active_tasks_++; }
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
  virtual void *get_ctx() { return NULL; }
  virtual void *get_stream(int index) { return NULL; }
  void enableD2D() { is_d2d_possible_ = true; }
  bool isD2DEnabled() { return is_d2d_possible_; }
  int ok() { return errid_; }
  void set_worker(Worker* worker) { worker_ = worker; }
  Worker* worker() { return worker_; }
  int GetStream(Task *task);
  int GetStream(Task *task, BaseMem *mem, bool new_stream=false);
  StreamPolicy stream_policy(Task *task);
  StreamPolicy stream_policy() { return stream_policy_; }
  double Now() { return timer_->Now(); }
  const char *kernel_path() { return kernel_path_.c_str(); }
protected:
  LoaderDefaultKernel* ld_default() { return ld_default_; }
  void CallMemReset(BaseMem *mem, size_t size, ResetData & data, void *stream);
  void LoadDefaultKernelLibrary(const char *key, const char *flags);
private:
  LoaderDefaultKernel *ld_default_;
  int get_new_stream_queue(int offset=0) {
    int nqs = ((nqueues_-1)-offset);
    if (nqs <= 0) return current_queue_ + offset+1;
    unsigned long new_current_queue;
    do {
        new_current_queue = current_queue_ + 1;
    } while (!__sync_bool_compare_and_swap(&current_queue_, current_queue_, new_current_queue));
    int stream = new_current_queue%nqs+offset+1;
    //printf("New queue:%d\n", stream);
    return stream;
  }
  int get_new_copy_stream_queue() {
    unsigned long new_current_queue;
    do {
        new_current_queue = current_copy_queue_ + 1;
    } while (!__sync_bool_compare_and_swap(&current_copy_queue_, current_copy_queue_, new_current_queue));
    int stream = new_current_queue%n_copy_engines_ + 1;
    //printf("New copy queue:%d\n", stream);
    return stream;
  }
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
  Task **dev_2_child_task_;
  int nqueues_;
  int errid_;
  int current_queue_;
  int current_copy_queue_;
  int n_copy_engines_;

  std::string kernel_path_;

  bool busy_;
  bool enable_;
  bool shared_memory_buffers_;
  bool can_share_host_memory_;
  bool is_d2d_possible_;
  bool native_kernel_not_exists_;
  bool async_;

  Worker* worker_;
  Timer* timer_;
  hook_task hook_task_pre_;
  hook_task hook_task_post_;
  hook_command hook_command_pre_;
  hook_command hook_command_post_;

  std::map<int, command_handler> cmd_handlers_;
  StreamPolicy stream_policy_;
  Platform *platform_obj_;
private:
  mutex destroy_events_mutex_;
  map<void *, bool> destroy_events_;
  Device *root_dev_;
  double first_event_cpu_begin_time_;
  double first_event_cpu_end_time_;
  double first_event_cpu_mid_point_time_;
  std::atomic<int> active_tasks_;
protected:
  int *peer_access_;
  int local_devno_;
  Device *root_device() { return root_dev_; }
  JuliaHostInterfaceLoader *julia_if_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_H */

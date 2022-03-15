#ifndef IRIS_SRC_RT_PLATFORM_H
#define IRIS_SRC_RT_PLATFORM_H

#include <iris/iris.h>
#include <pthread.h>
#include <stddef.h>
#include <vector>
#include <set>
#include <map>
#include <mutex>
#include <string>
#include "Config.h"

namespace iris {
namespace rt {

class Device;
class Filter;
class Graph;
class JSON;
class Kernel;
class LoaderCUDA;
class LoaderHost2OpenCL;
class LoaderHost2CUDA;
class LoaderHost2HIP;
class LoaderHIP;
class LoaderLevelZero;
class LoaderOpenCL;
class LoaderOpenMP;
class LoaderHexagon;
class Mem;
class Polyhedral;
class Pool;
class PresentTable;
class Profiler;
class Queue;
class Scheduler;
class SigHandler;
class Task;
class Timer;
class Worker;

class Platform {
private:
  Platform();
  ~Platform();

public:
  int Init(int* argc, char*** argv, int sync);
  int Finalize();
  int Synchronize();

  int EnvironmentInit();
  int EnvironmentSet(const char* key, const char* value, bool overwrite);
  int EnvironmentGet(const char* key, char** value, size_t* vallen);

  int PlatformCount(int* nplatforms);
  int PlatformInfo(int platform, int param, void* value, size_t* size);
  int PlatformBuildProgram(int model, char* path);

  int DeviceCount(int* ndevs);
  int DeviceInfo(int device, int param, void* value, size_t* size);
  int DeviceSetDefault(int device);
  int DeviceGetDefault(int* device);
  int DeviceSynchronize(int ndevs, int* devices);

  int PolicyRegister(const char* lib, const char* name, void* params);
  int RegisterCommand(int tag, int device, command_handler handler);
  int RegisterHooksTask(hook_task pre, hook_task post);
  int RegisterHooksCommand(hook_command pre, hook_command post);

  int KernelCreate(const char* name, iris_kernel* brs_kernel);
  int KernelGet(const char* name, iris_kernel* brs_kernel);
  int KernelSetArg(iris_kernel brs_kernel, int idx, size_t size, void* value);
  int KernelSetMem(iris_kernel brs_kernel, int idx, iris_mem mem, size_t off, size_t mode);
  int KernelSetMap(iris_kernel brs_kernel, int idx, void* host, size_t mode);
  int KernelRelease(iris_kernel brs_kernel);

  int TaskCreate(const char* name, bool perm, iris_task* brs_task);
  int TaskDepend(iris_task brs_task, int ntasks, iris_task* brs_tasks);
  int TaskKernel(iris_task brs_task, iris_kernel brs_kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int TaskKernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges);
  int TaskKernelSelector(iris_task task, iris_selector_kernel func, void* params, size_t params_size);
  int TaskHost(iris_task task, iris_host_task func, void* params);
  int TaskCustom(iris_task task, int tag, void* params, size_t params_size);
  int TaskMalloc(iris_task brs_task, iris_mem brs_mem);
  int TaskH2D(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host);
  int TaskD2H(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host);
  int TaskH2DFull(iris_task brs_task, iris_mem brs_mem, void* host);
  int TaskD2HFull(iris_task brs_task, iris_mem brs_mem, void* host);
  int TaskMap(iris_task brs_task, void* host, size_t size);
  int TaskMapTo(iris_task brs_task, void* host, size_t size);
  int TaskMapToFull(iris_task brs_task, void* host);
  int TaskMapFrom(iris_task brs_task, void* host, size_t size);
  int TaskMapFromFull(iris_task brs_task, void* host);
  int TaskSubmit(iris_task brs_task, int brs_policy, const char* opt, int wait);
  int TaskWait(iris_task brs_task);
  int TaskWaitAll(int ntasks, iris_task* brs_tasks);
  int TaskAddSubtask(iris_task brs_task, iris_task brs_subtask);
  int TaskKernelCmdOnly(iris_task brs_task);
  int TaskRelease(iris_task brs_task);
  int TaskReleaseMem(iris_task brs_task, iris_mem brs_mem);
  int SetParamsMap(iris_task brs_task, int *params_map);
  int SetSharedMemoryModel(int flag);
  int TaskInfo(iris_task brs_task, int param, void* value, size_t* size);

  int MemCreate(size_t size, iris_mem* brs_mem);
  int MemSetIntermediate(iris_mem brs_mem, bool flag=true);
  int MemArch(iris_mem brs_mem, int device, void** arch);
  int MemMap(void* host, size_t size);
  int MemUnmap(void* host);
  int MemReduce(iris_mem brs_mem, int mode, int type);
  int MemRelease(iris_mem brs_mem);

  int GraphCreate(iris_graph* brs_graph);
  int GraphCreateJSON(const char* json, void** params,  iris_graph* brs_graph);
  int GraphTask(iris_graph brs_graph, iris_task brs_task, int brs_policy, const char* opt);
  int GraphSubmit(iris_graph brs_graph, int brs_policy, int sync);
  int GraphWait(iris_graph brs_graph);
  int GraphWaitAll(int ngraphs, iris_graph* brs_graphs);

  int RecordStart();
  int RecordStop();

  int TimerNow(double* time);

  int ndevs() { return ndevs_; }
  int device_default() { return dev_default_; }
  Device** devices() { return devs_; }
  Device* device(int devno) { return devs_[devno]; }
  Polyhedral* polyhedral() { return polyhedral_; }
  Worker** workers() { return workers_; }
  Worker* worker(int i) { return workers_[i]; }
  Queue* queue() { return queue_; }
  Pool* pool() { return pool_; }
  Scheduler* scheduler() { return scheduler_; }
  Timer* timer() { return timer_; }
  Kernel* null_kernel() { return null_kernel_; }
  char* app() { return app_; }
  char* host() { return host_; }
  Profiler** profilers() { return profilers_; }
  int nprofilers() { return nprofilers_; }
  double time_app() { return time_app_; }
  double time_init() { return time_init_; }
  bool enable_profiler() { return enable_profiler_; }
  hook_task hook_task_pre() { return hook_task_pre_; }
  hook_task hook_task_post() { return hook_task_post_; }
  hook_command hook_command_pre() { return hook_command_pre_; }
  hook_command hook_command_post() { return hook_command_post_; }

  Kernel* GetKernel(const char* name);
  Mem* GetMem(iris_mem brs_mem);
  Mem* GetMem(void* host, size_t* off);

private:
  int SetDevsAvailable();
  int InitCUDA();
  int InitHIP();
  int InitLevelZero();
  int InitOpenCL();
  int InitOpenMP();
  int InitHexagon();
  int InitDevices(bool sync);
  int InitScheduler();
  int InitWorkers();
  int FilterSubmitExecute(Task* task);
  int ShowKernelHistory();

public:
  static Platform* GetPlatform();

private:
  bool init_;
  bool finalize_;

  char platform_names_[IRIS_MAX_NPLATFORMS][64];
  int nplatforms_;
  Device* devs_[IRIS_MAX_NDEVS];
  int ndevs_;
  int dev_default_;
  int devs_enabled_[IRIS_MAX_NDEVS];
  int ndevs_enabled_;

  std::vector<LoaderHost2OpenCL*> loaderHost2OpenCL_;
  LoaderHost2HIP * loaderHost2HIP_;
  LoaderHost2CUDA * loaderHost2CUDA_;
  LoaderCUDA* loaderCUDA_;
  LoaderHIP* loaderHIP_;
  LoaderLevelZero* loaderLevelZero_;
  LoaderOpenCL* loaderOpenCL_;
  LoaderOpenMP* loaderOpenMP_;
  LoaderHexagon* loaderHexagon_;
  size_t arch_available_;

  Queue* queue_;

  std::set<Kernel*> kernels_;
  std::set<Mem*> mems_;
  std::map<std::string, std::string> env_;

  PresentTable* present_table_;
  Pool* pool_;

  Worker* workers_[IRIS_MAX_NDEVS];

  Scheduler* scheduler_;
  Timer* timer_;
  Polyhedral* polyhedral_;
  bool polyhedral_available_;
  Filter* filter_task_split_;
  SigHandler* sig_handler_;
  JSON* json_;

  bool recording_;

  bool enable_profiler_;
  Profiler* profilers_[8];
  int nprofilers_;

  Kernel* null_kernel_;

  pthread_mutex_t mutex_;
  hook_task hook_task_pre_;
  hook_task hook_task_post_;
  hook_command hook_command_pre_;
  hook_command hook_command_post_;

  char app_[256];
  char host_[256];
  double time_app_;
  double time_init_;

private:
  static Platform* singleton_;
  static std::once_flag flag_singleton_;
  static std::once_flag flag_finalize_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_PLATFORM_H */

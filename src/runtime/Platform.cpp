#include "Platform.h"
#include "Debug.h"
#include "Utils.h"
#include "Command.h"
#include "DeviceCUDA.h"
#include "DeviceHexagon.h"
#include "DeviceHIP.h"
#include "DeviceLevelZero.h"
#include "DeviceOpenCL.h"
#include "DeviceOpenMP.h"
#include "FilterTaskSplit.h"
#include "Graph.h"
#include "History.h"
#include "JSON.h"
#include "Kernel.h"
#include "LoaderCUDA.h"
#include "LoaderHexagon.h"
#include "LoaderHIP.h"
#include "LoaderLevelZero.h"
#include "LoaderOpenCL.h"
#include "LoaderOpenMP.h"
#include "Mem.h"
#include "Policies.h"
#include "Polyhedral.h"
#include "Pool.h"
#include "PresentTable.h"
#include "Profiler.h"
#include "ProfilerDOT.h"
#include "ProfilerGoogleCharts.h"
#include "QueueTask.h"
#include "Scheduler.h"
#include "SigHandler.h"
#include "Task.h"
#include "Timer.h"
#include "Worker.h"
#include <unistd.h>
#include <algorithm>

namespace brisbane {
namespace rt {

char brisbane_log_prefix_[256];

Platform::Platform() {
  init_ = false;
  finalize_ = false;
  nplatforms_ = 0;
  ndevs_ = 0;
  ndevs_enabled_ = 0;
  dev_default_ = 0;

  queue_ = NULL;
  pool_ = NULL;
  scheduler_ = NULL;
  polyhedral_ = NULL;
  sig_handler_ = NULL;
  filter_task_split_ = NULL;
  timer_ = NULL;
  null_kernel_ = NULL;
  loaderCUDA_ = NULL;
  loaderHIP_ = NULL;
  loaderLevelZero_ = NULL;
  loaderOpenCL_ = NULL;
  loaderOpenMP_ = NULL;
  arch_available_ = 0UL;
  present_table_ = NULL;
  recording_ = false;
  enable_profiler_ = getenv("IRIS_PROFILE");
  nprofilers_ = 0;
  time_app_ = 0.0;
  time_init_ = 0.0;
  hook_task_pre_ = NULL;
  hook_task_post_ = NULL;
  hook_command_pre_ = NULL;
  hook_command_post_ = NULL;

  pthread_mutex_init(&mutex_, NULL);
}

Platform::~Platform() {
  if (!init_) return;
  if (scheduler_) delete scheduler_;
  for (int i = 0; i < ndevs_; i++) delete workers_[i];
  if (queue_) delete queue_;
  if (loaderCUDA_) delete loaderCUDA_;
  if (loaderHIP_) delete loaderHIP_;
  if (loaderLevelZero_) delete loaderLevelZero_;
  if (loaderOpenCL_) delete loaderOpenCL_;
  if (loaderOpenMP_) delete loaderOpenMP_;
  if (present_table_) delete present_table_;
  if (polyhedral_) delete polyhedral_;
  if (filter_task_split_) delete filter_task_split_;
  if (timer_) delete timer_;
  if (null_kernel_) delete null_kernel_;
  if (enable_profiler_)
    for (int i = 0; i < nprofilers_; i++) delete profilers_[i];
  if (sig_handler_) delete sig_handler_;
  if (json_) delete json_;
  if (pool_) delete pool_;

  pthread_mutex_destroy(&mutex_);
}

int Platform::Init(int* argc, char*** argv, int sync) {
  pthread_mutex_lock(&mutex_);
  if (init_) {
    pthread_mutex_unlock(&mutex_);
    return BRISBANE_ERR;
  }

  gethostname(brisbane_log_prefix_, 256);
  gethostname(host_, 256);
  if (argv && *argv) sprintf(app_, "%s", (*argv)[0]);
  else sprintf(app_, "%s", "app");

  timer_ = new Timer();
  timer_->Start(BRISBANE_TIMER_APP);

  timer_->Start(BRISBANE_TIMER_PLATFORM);
  sig_handler_ = new SigHandler();

  json_ = new JSON(this);

  EnvironmentInit();

  char* logo = NULL;
  EnvironmentGet("LOGO", &logo, NULL);
  if (strcmp("on", logo) == 0) Utils::Logo(true);

  SetDevsAvailable();

  char* archs = NULL;
  EnvironmentGet("ARCHS", &archs, NULL);
  _info("IRIS architectures[%s]", archs);
  const char* delim = " :;.,";
  char arch_str[128];
  memset(arch_str, 0, 128);
  strncpy(arch_str, archs, strlen(archs));
  char* rest = arch_str;
  char* a = NULL;
  while ((a = strtok_r(rest, delim, &rest))) {
    if (strcasecmp(a, "cuda") == 0) {
      if (!loaderCUDA_) InitCUDA();
    } else if (strcasecmp(a, "hip") == 0) {
      if (!loaderHIP_) InitHIP();
    } else if (strcasecmp(a, "levelzero") == 0) {
      if (!loaderLevelZero_) InitLevelZero();
    } else if (strcasecmp(a, "opencl") == 0) {
      if (!loaderOpenCL_) InitOpenCL();
    } else if (strcasecmp(a, "openmp") == 0) {
      if (!loaderOpenMP_) InitOpenMP();
    } else if (strcasecmp(a, "hexagon") == 0) {
      if (!loaderHexagon_) InitHexagon();
    } else _error("not support arch[%s]", a);
  }
  if (ndevs_enabled_ > ndevs_) ndevs_enabled_ = ndevs_;
  polyhedral_ = new Polyhedral();
  polyhedral_available_ = polyhedral_->Load() == BRISBANE_OK;
  if (polyhedral_available_)
    filter_task_split_ = new FilterTaskSplit(polyhedral_, this);

  brisbane_kernel null_brs_kernel;
  KernelCreate("brisbane_null", &null_brs_kernel);
  null_kernel_ = null_brs_kernel->class_obj;

  if (enable_profiler_) {
    profilers_[nprofilers_++] = new ProfilerDOT(this);
    profilers_[nprofilers_++] = new ProfilerGoogleCharts(this);
  }

  present_table_ = new PresentTable();
  queue_ = new QueueTask(this);
  pool_ = new Pool(this);

  InitScheduler();
  InitWorkers();
  InitDevices(sync);

  _info("nplatforms[%d] ndevs[%d] ndevs_enabled[%d] scheduler[%d] hub[%d] polyhedral[%d] profile[%d]",
      nplatforms_, ndevs_, ndevs_enabled_, scheduler_ != NULL, scheduler_ ? scheduler_->hub_available() : 0,
      polyhedral_available_, enable_profiler_);

  timer_->Stop(BRISBANE_TIMER_PLATFORM);

  init_ = true;

  pthread_mutex_unlock(&mutex_);

  return BRISBANE_OK;
}

int Platform::Synchronize() {
  int* devices = new int[ndevs_];
  for (int i = 0; i < ndevs_; i++) devices[i] = i;
  int ret = DeviceSynchronize(ndevs_, devices);
  delete devices;
  return ret;
}

int Platform::EnvironmentInit() {
  EnvironmentSet("ARCHS", "openmp:cuda:hip:levelzero:hexagon:opencl", false);

  EnvironmentSet("KERNEL_CUDA",     "kernel.ptx",         false);
  EnvironmentSet("KERNEL_HEXAGON",  "kernel.hexagon.so",  false);
  EnvironmentSet("KERNEL_HIP",      "kernel.hip",         false);
  EnvironmentSet("KERNEL_OPENMP",   "kernel.openmp.so",   false);
  EnvironmentSet("KERNEL_SPV",      "kernel.spv",         false);

  EnvironmentSet("LOGO",            "off",                false);
  return BRISBANE_OK;
}

int Platform::EnvironmentSet(const char* key, const char* value, bool overwrite) {
  std::string keystr = std::string(key);
  std::string valstr = std::string(value);
  auto it = env_.find(keystr);
  if (it != env_.end()) {
    if (!overwrite) return BRISBANE_ERR;
    env_.erase(it);
  }
  env_.insert(std::pair<std::string, std::string>(keystr, valstr));
  return BRISBANE_OK;
}

int Platform::EnvironmentGet(const char* key, char** value, size_t* vallen) {
  char env_key[128];
  sprintf(env_key, "IRIS_%s", key);
  const char* val = getenv(env_key);
  if (!val) {
    std::string keystr = std::string(key);
    auto it = env_.find(keystr);
    if (it == env_.end()) {
      if (vallen) *vallen = 0;
      return BRISBANE_ERR;
    }
    val = it->second.c_str();
  }

  if (*value == NULL) *value = (char*) malloc(strlen(val) + 1);
  strcpy(*value, val);
  if (vallen) *vallen = strlen(val) + 1;
  return BRISBANE_OK;
}

int Platform::SetDevsAvailable() {
  const char* enabled = getenv("IRIS_DEV_ENABLED");
  if (!enabled) {
    for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) devs_enabled_[i] = i;
    ndevs_enabled_ = BRISBANE_MAX_NDEVS;
    return BRISBANE_OK;
  }
  _info("IRIS ENABLED DEVICES[%s]", enabled);
  const char* delim = " :;.,";
  char str[128];
  memset(str, 0, 128);
  strncpy(str, enabled, strlen(enabled));
  char* rest = str;
  char* a = NULL;
  while ((a = strtok_r(rest, delim, &rest))) {
    devs_enabled_[ndevs_enabled_++] = atoi(a);
  }
  for (int i = 0; i < ndevs_enabled_; i++) {
    _debug("devs_available[%d]", devs_enabled_[i]);
  }
  return BRISBANE_OK;
}

int Platform::InitCUDA() {
  if (arch_available_ & brisbane_nvidia) {
    _trace("%s", "skipping CUDA architecture");
    return BRISBANE_ERR;
  }
  loaderCUDA_ = new LoaderCUDA();
  if (loaderCUDA_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping CUDA architecture");
    return BRISBANE_ERR;
  }
  CUresult err = CUDA_SUCCESS;
  err = loaderCUDA_->cuInit(0);
  if (err != CUDA_SUCCESS) {
    _trace("skipping CUDA architecture CUDA_ERROR[%d]", err);
    return BRISBANE_ERR;
  }
  int ndevs = 0;
  err = loaderCUDA_->cuDeviceGetCount(&ndevs);
  _cuerror(err);
  if (getenv("IRIS_SINGLE")) ndevs = 1;
  _trace("CUDA platform[%d] ndevs[%d]", nplatforms_, ndevs);
  for (int i = 0; i < ndevs; i++) {
    CUdevice dev;
    err = loaderCUDA_->cuDeviceGet(&dev, i);
    _cuerror(err);
    devs_[ndevs_] = new DeviceCUDA(loaderCUDA_, dev, ndevs_, nplatforms_);
    arch_available_ |= devs_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "CUDA");
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitHIP() {
  if (arch_available_ & brisbane_amd) {
    _trace("%s", "skipping HIP architecture");
    return BRISBANE_ERR;
  }
  loaderHIP_ = new LoaderHIP();
  if (loaderHIP_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping HIP architecture");
    return BRISBANE_ERR;
  }
  hipError_t err = hipSuccess;
  err = loaderHIP_->hipInit(0);
  _hiperror(err);
  int ndevs = 0;
  err = loaderHIP_->hipGetDeviceCount(&ndevs);
  _hiperror(err);
  if (getenv("IRIS_SINGLE")) ndevs = 1;
  _trace("HIP platform[%d] ndevs[%d]", nplatforms_, ndevs);
  for (int i = 0; i < ndevs; i++) {
    hipDevice_t dev;
    err = loaderHIP_->hipDeviceGet(&dev, i);
    _hiperror(err);
    devs_[ndevs_] = new DeviceHIP(loaderHIP_, dev, i, ndevs_, nplatforms_);
    arch_available_ |= devs_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "HIP");
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitLevelZero() {
  if (arch_available_ & brisbane_gpu_intel) {
    _trace("%s", "skipping LevelZero architecture");
    return BRISBANE_ERR;
  }
  loaderLevelZero_ = new LoaderLevelZero();
  if (loaderLevelZero_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping LevelZero architecture");
    return BRISBANE_ERR;
  }

  ze_result_t err = ZE_RESULT_SUCCESS;
  err = loaderLevelZero_->zeInit(0);
  _zeerror(err);

  ze_driver_handle_t driver;
  uint32_t ndrivers = 0;
  err = loaderLevelZero_->zeDriverGet(&ndrivers, nullptr);
  _zeerror(err);

  _info("LevelZero driver count[%u]",  ndrivers);
  if (ndrivers != 1) return BRISBANE_ERR;

  err = loaderLevelZero_->zeDriverGet(&ndrivers, &driver);
  _zeerror(err);

  uint32_t ndevs = 0;
  err = loaderLevelZero_->zeDeviceGet(driver, &ndevs, nullptr);
  _zeerror(err);
  _info("LevelZero ndevs[%u]", ndevs);

  ze_device_handle_t* devs = new ze_device_handle_t[ndevs]; 
  err = loaderLevelZero_->zeDeviceGet(driver, &ndevs, devs);
  _zeerror(err);

  ze_context_handle_t zectx;
  ze_context_desc_t zectx_desc = {};
  zectx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  err = loaderLevelZero_->zeContextCreate(driver, &zectx_desc, &zectx);
  _zeerror(err);

  for (uint32_t i = 0; i < ndevs; i++) {
    devs_[ndevs_] = new DeviceLevelZero(loaderLevelZero_, devs[i], zectx, driver, ndevs_, nplatforms_);
    arch_available_ |= devs_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "LevelZero");
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitOpenMP() {
  if (arch_available_ & brisbane_cpu) {
    _trace("%s", "skipping OpenMP architecture");
    return BRISBANE_ERR;
  }
  loaderOpenMP_ = new LoaderOpenMP();
  if (loaderOpenMP_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping OpenMP architecture");
    return BRISBANE_ERR;
  }
  _trace("OpenMP platform[%d] ndevs[%d]", nplatforms_, 1);
  devs_[ndevs_] = new DeviceOpenMP(loaderOpenMP_, ndevs_, nplatforms_);
  arch_available_ |= devs_[ndevs_]->type();
  ndevs_++;
  strcpy(platform_names_[nplatforms_], "OpenMP");
  nplatforms_++;
  return BRISBANE_OK;
}

int Platform::InitHexagon() {
  if (arch_available_ & brisbane_hexagon) {
    _trace("%s", "skipping Hexagon architecture");
    return BRISBANE_ERR;
  }
  loaderHexagon_ = new LoaderHexagon();
  if (loaderHexagon_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping Hexagon architecture");
    return BRISBANE_ERR;
  }
  _trace("Hexagon platform[%d] ndevs[%d]", nplatforms_, 1);
  devs_[ndevs_] = new DeviceHexagon(loaderHexagon_, ndevs_, nplatforms_);
  arch_available_ |= devs_[ndevs_]->type();
  ndevs_++;
  strcpy(platform_names_[nplatforms_], "Hexagon");
  nplatforms_++;
  return BRISBANE_OK;
}

int Platform::InitOpenCL() {
  loaderOpenCL_ = new LoaderOpenCL();
  if (loaderOpenCL_->Load() != BRISBANE_OK) {
    _trace("%s", "skipping OpenCL architecture");
    return BRISBANE_ERR;
  }
  cl_platform_id cl_platforms[BRISBANE_MAX_NDEVS];
  cl_context cl_contexts[BRISBANE_MAX_NDEVS];
  cl_device_id cl_devices[BRISBANE_MAX_NDEVS];
  cl_int err;

  cl_uint nplatforms = BRISBANE_MAX_NDEVS;

  err = loaderOpenCL_->clGetPlatformIDs(nplatforms, cl_platforms, &nplatforms);
  _trace("OpenCL nplatforms[%u]", nplatforms);
  if (!nplatforms) return BRISBANE_OK;
  cl_uint ndevs = 0;
  char vendor[64];
  char platform_name[64];
  for (cl_uint i = 0; i < nplatforms; i++) {
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    _clerror(err);
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    _clerror(err);
    _trace("OpenCL platform[%s] from [%s]", platform_name, vendor);

    if ((arch_available_ & brisbane_nvidia) && strstr(vendor, "NVIDIA") != NULL) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    if ((arch_available_ & brisbane_amd) && strstr(vendor, "Advanced Micro Devices") != NULL) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    err = loaderOpenCL_->clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &ndevs);
    if (!ndevs) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    err = loaderOpenCL_->clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, ndevs, cl_devices, NULL);
    _clerror(err);
    cl_contexts[i] = loaderOpenCL_->clCreateContext(NULL, ndevs, cl_devices, NULL, NULL, &err);
    _clerror(err);
    if (err != CL_SUCCESS) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    for (cl_uint j = 0; j < ndevs; j++) {
      cl_device_type dev_type;
      err = loaderOpenCL_->clGetDeviceInfo(cl_devices[j], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
      _clerror(err);
      if ((arch_available_ & brisbane_cpu) && (dev_type == CL_DEVICE_TYPE_CPU)) continue;
      devs_[ndevs_] = new DeviceOpenCL(loaderOpenCL_, cl_devices[j], cl_contexts[i], ndevs_, nplatforms_);
      arch_available_ |= devs_[ndevs_]->type();
      ndevs_++;
    }
    _trace("adding platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
    sprintf(platform_names_[nplatforms_], "OpenCL %s", vendor);
    nplatforms_++;
  }
  return BRISBANE_OK;
}

int Platform::InitDevices(bool sync) {
  if (!ndevs_) {
    dev_default_ = -1;
    ___error("%s", "NO AVAILABLE DEVICES!");
    return BRISBANE_ERR;
  }
  char* c = getenv("IRIS_DEVICE_DEFAULT");
  if (c) dev_default_ = atoi(c);

  Task** tasks = new Task*[ndevs_];
  for (int i = 0; i < ndevs_; i++) {
    tasks[i] = new Task(this);
    tasks[i]->set_system();
    Command* cmd = Command::CreateInit(tasks[i]);
    tasks[i]->AddCommand(cmd);
    workers_[i]->Enqueue(tasks[i]);
  }
  if (sync) for (int i = 0; i < ndevs_; i++) tasks[i]->Wait();
  delete[] tasks;
  return BRISBANE_OK;
}

int Platform::PlatformCount(int* nplatforms) {
  if (nplatforms) *nplatforms = nplatforms_;
  return BRISBANE_OK;
}

int Platform::PlatformInfo(int platform, int param, void* value, size_t* size) {
  if (platform >= nplatforms_) return BRISBANE_ERR;
  switch (param) {
    case brisbane_name:
      if (size) *size = strlen(platform_names_[platform]);
      strcpy((char*) value, platform_names_[platform]);
      break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int Platform::PlatformBuildProgram(int model, char* path) {
  for (int i = 0; i < ndevs_; i++)
    if (devs_[i]->model() == model) devs_[i]->BuildProgram(path);
  return BRISBANE_OK;
}

int Platform::DeviceCount(int* ndevs) {
  if (ndevs) *ndevs = ndevs_;
  return BRISBANE_OK;
}

int Platform::DeviceInfo(int device, int param, void* value, size_t* size) {
  if (device >= ndevs_) return BRISBANE_ERR;
  Device* dev = devs_[device];
  switch (param) {
    case brisbane_platform  : if (size) *size = sizeof(int);            *((int*) value) = dev->platform();      break;
    case brisbane_vendor    : if (size) *size = strlen(dev->vendor());  strcpy((char*) value, dev->vendor());   break;
    case brisbane_name      : if (size) *size = strlen(dev->name());    strcpy((char*) value, dev->name());     break;
    case brisbane_type      : if (size) *size = sizeof(int);            *((int*) value) = dev->type();          break;
    default: return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int Platform::DeviceSetDefault(int device) {
  dev_default_ = device;
  return BRISBANE_OK;
}

int Platform::DeviceGetDefault(int* device) {
  *device = dev_default_;
  return BRISBANE_OK;
}

int Platform::DeviceSynchronize(int ndevs, int* devices) {
  Task* task = new Task(this, BRISBANE_MARKER);
  if (scheduler_) {
    for (int i = 0; i < ndevs; i++) {
      if (devices[i] >= ndevs_) {
        _error("devices[%d]", devices[i]);
        continue;
      }
      Task* subtask = new Task(this, BRISBANE_MARKER);
      subtask->set_devno(devices[i]);
      task->AddSubtask(subtask);
    }
    scheduler_->Enqueue(task);
  } else workers_[0]->Enqueue(task);
  task->Wait();
  return task->Ok();
}

int Platform::PolicyRegister(const char* lib, const char* name, void* params) {
  return scheduler_->policies()->Register(lib, name, params);
}

int Platform::RegisterCommand(int tag, int device, command_handler handler) {
  for (int i = 0; i < ndevs_; i++)
    if (devs_[i]->type() == device) devs_[i]->RegisterCommand(tag, handler);
  return BRISBANE_OK;
}

int Platform::RegisterHooksTask(hook_task pre, hook_task post) {
  hook_task_pre_ = pre;
  hook_task_post_ = post;
  for (int i = 0; i < ndevs_; i++) devs_[i]->RegisterHooks();
  return BRISBANE_OK;
}

int Platform::RegisterHooksCommand(hook_command pre, hook_command post) {
  hook_command_pre_ = pre;
  hook_command_post_ = post;
  for (int i = 0; i < ndevs_; i++) devs_[i]->RegisterHooks();
  return BRISBANE_OK;
}

int Platform::KernelCreate(const char* name, brisbane_kernel* brs_kernel) {
  Kernel* kernel = new Kernel(name, this);
  if (brs_kernel) *brs_kernel = kernel->struct_obj();
  kernels_.insert(kernel);
  return BRISBANE_OK;
}

int Platform::KernelGet(const char* name, brisbane_kernel* brs_kernel) {
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    Kernel* kernel = *I;
    if (strcmp(kernel->name(), name) == 0) {
      if (brs_kernel) *brs_kernel = kernel->struct_obj();
      return BRISBANE_OK;
    }
  }
  return KernelCreate(name, brs_kernel);
}

int Platform::KernelSetArg(brisbane_kernel brs_kernel, int idx, size_t size, void* value) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->SetArg(idx, size, value);
  return BRISBANE_OK;
}

int Platform::KernelSetMem(brisbane_kernel brs_kernel, int idx, brisbane_mem brs_mem, size_t off, size_t mode) {
  Kernel* kernel = brs_kernel->class_obj;
  Mem* mem = brs_mem->class_obj;
  kernel->SetMem(idx, mem, off, mode);
  return BRISBANE_OK;
}

int Platform::KernelSetMap(brisbane_kernel brs_kernel, int idx, void* host, size_t mode) {
  Kernel* kernel = brs_kernel->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  if (mem) kernel->SetMem(idx, mem, off, mode);
  else {
    _todo("clearing [%p]", host);
    MemMap(host, 8192);
    Mem* mem = present_table_->Get(host, &off);
    kernel->SetMem(idx, mem, off, mode);
  }
  return BRISBANE_OK;
}

int Platform::KernelRelease(brisbane_kernel brs_kernel) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->Release();
  return BRISBANE_OK;
}

int Platform::TaskCreate(const char* name, bool perm, brisbane_task* brs_task) {
  Task* task = Task::Create(this, perm ? BRISBANE_TASK_PERM : BRISBANE_TASK, name);
  *brs_task = task->struct_obj();
  return BRISBANE_OK;
}

int Platform::TaskDepend(brisbane_task brs_task, int ntasks, brisbane_task* brs_tasks) {
  Task* task = brs_task->class_obj;
  for (int i = 0; i < ntasks; i++) task->AddDepend(brs_tasks[i]->class_obj);
  return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, brisbane_kernel brs_kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  Task* task = brs_task->class_obj;
  Kernel* kernel = brs_kernel->class_obj;
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, gws, lws);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskCustom(brisbane_task brs_task, int tag, void* params, size_t params_size) {
  Task* task = brs_task->class_obj;
  Command* cmd = Command::CreateCustom(task, tag, params, params_size);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskKernel(brisbane_task brs_task, const char* name, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges) {
  Task* task = brs_task->class_obj;
  Kernel* kernel = GetKernel(name);
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, memranges);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskKernelSelector(brisbane_task brs_task, brisbane_selector_kernel func, void* params, size_t params_size) {
  Task* task = brs_task->class_obj;
  Command* cmd = task->cmd_kernel();
  if (!cmd) return BRISBANE_ERR;
  cmd->set_selector_kernel(func, params, params_size);
  return BRISBANE_OK;
}

int Platform::TaskHost(brisbane_task brs_task, brisbane_host_task func, void* params) {
  Task* task = brs_task->class_obj;
  Command* cmd = Command::CreateHost(task, func, params);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskMalloc(brisbane_task brs_task, brisbane_mem brs_mem) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateMalloc(task, mem);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskH2D(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskD2H(brisbane_task brs_task, brisbane_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskH2DFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host) {
  return TaskH2D(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskD2HFull(brisbane_task brs_task, brisbane_mem brs_mem, void* host) {
  return TaskD2H(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskMap(brisbane_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  Command* cmd = Command::CreateMap(task, host, size);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskMapTo(brisbane_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskMapToFull(brisbane_task brs_task, void* host) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  size_t size = mem->size();
  Command* cmd = Command::CreateH2D(task, mem, off, size - off, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskMapFrom(brisbane_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskMapFromFull(brisbane_task brs_task, void* host) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  size_t size = mem->size();
  Command* cmd = Command::CreateD2H(task, mem, off, size - off, host);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::TaskSubmit(brisbane_task brs_task, int brs_policy, const char* opt, int sync) {
  Task* task = brs_task->class_obj;
  task->Submit(brs_policy, opt, sync);
  if (recording_) json_->RecordTask(task);
  if (scheduler_) {
    FilterSubmitExecute(task);
    scheduler_->Enqueue(task);
  } else workers_[0]->Enqueue(task);
  if (sync) task->Wait();
  return BRISBANE_OK;
}

int Platform::TaskWait(brisbane_task brs_task) {
  Task* task = brs_task->class_obj;
  task->Wait();
  return BRISBANE_OK;
}

int Platform::TaskWaitAll(int ntasks, brisbane_task* brs_tasks) {
  int iret = BRISBANE_OK;
  for (int i = 0; i < ntasks; i++) iret &= TaskWait(brs_tasks[i]);
  return iret;
}

int Platform::TaskAddSubtask(brisbane_task brs_task, brisbane_task brs_subtask) {
  Task* task = brs_task->class_obj;
  Task* subtask = brs_subtask->class_obj;
  task->AddSubtask(subtask);
  return BRISBANE_OK;
}

int Platform::TaskKernelCmdOnly(brisbane_task brs_task) {
  Task* task = brs_task->class_obj;
  return (task->ncmds() == 1 && task->cmd_kernel()) ? BRISBANE_OK : BRISBANE_ERR;
}

int Platform::TaskRelease(brisbane_task brs_task) {
  Task* task = brs_task->class_obj;
  task->Release();
  return BRISBANE_OK;
}

int Platform::TaskReleaseMem(brisbane_task brs_task, brisbane_mem brs_mem) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateReleaseMem(task, mem);
  task->AddCommand(cmd);
  return BRISBANE_OK;
}

int Platform::MemCreate(size_t size, brisbane_mem* brs_mem) {
  Mem* mem = new Mem(size, this);
  if (brs_mem) *brs_mem = mem->struct_obj();
  mems_.insert(mem);
  return BRISBANE_OK;
}

int Platform::MemArch(brisbane_mem brs_mem, int device, void** arch) {
  if (!arch) return BRISBANE_ERR;
  Mem* mem = brs_mem->class_obj;
  Device* dev = devs_[device];
  void* ret = mem->arch(dev);
  if (!ret) return BRISBANE_ERR;
  *arch = ret;
  return BRISBANE_OK;
}

int Platform::MemMap(void* host, size_t size) {
  Mem* mem = new Mem(size, this);
  mem->SetMap(host, size);
  mems_.insert(mem);
  present_table_->Add(host, size, mem);
  return BRISBANE_OK;
}

int Platform::MemUnmap(void* host) {
  Mem* mem = present_table_->Remove(host);
  mem->Release();
  return BRISBANE_OK;
}

int Platform::MemReduce(brisbane_mem brs_mem, int mode, int type) {
  Mem* mem = brs_mem->class_obj;
  mem->Reduce(mode, type);
  return BRISBANE_OK;
}

int Platform::MemRelease(brisbane_mem brs_mem) {
  Mem* mem = brs_mem->class_obj;
  mem->Release();
  return BRISBANE_OK;
}

int Platform::GraphCreate(brisbane_graph* brs_graph) {
  Graph* graph = Graph::Create(this);
  *brs_graph = graph->struct_obj();
  return BRISBANE_OK;
}

int Platform::GraphCreateJSON(const char* path, void** params, brisbane_graph* brs_graph) {
  Graph* graph = Graph::Create(this);
  *brs_graph = graph->struct_obj();
  json_->Load(graph, path, params);
  return BRISBANE_OK;
}

int Platform::GraphTask(brisbane_graph brs_graph, brisbane_task brs_task, int brs_policy, const char* opt) {
  Graph* graph = brs_graph->class_obj;
  Task* task = brs_task->class_obj;
  task->set_target_perm(brs_policy, opt);
  graph->AddTask(task);
  return BRISBANE_OK;
}

int Platform::GraphSubmit(brisbane_graph brs_graph, int brs_policy, int sync) {
  Graph* graph = brs_graph->class_obj;
  std::vector<Task*>* tasks = graph->tasks();
  for (std::vector<Task*>::iterator I = tasks->begin(), E = tasks->end(); I != E; ++I) {
    Task* task = *I;
    int policy = task->brs_policy_perm() == brisbane_default ? brs_policy : task->brs_policy_perm();
    task->Submit(policy, task->opt(), sync);
    if (recording_) json_->RecordTask(task);
    if (scheduler_) scheduler_->Enqueue(task);
    else workers_[0]->Enqueue(task);
  }
  if (sync) graph->Wait();
  return BRISBANE_OK;
}

int Platform::GraphWait(brisbane_graph brs_graph) {
  Graph* graph = brs_graph->class_obj;
  graph->Wait();
  return BRISBANE_OK;
}

int Platform::GraphWaitAll(int ngraphs, brisbane_graph* brs_graphs) {
  int iret = BRISBANE_OK;
  for (int i = 0; i < ngraphs; i++) iret &= GraphWait(brs_graphs[i]);
  return iret;
}

int Platform::RecordStart() {
  recording_ = true;
  return BRISBANE_OK;
}

int Platform::RecordStop() {
  json_->RecordFlush();
  recording_ = false;
  return BRISBANE_OK;
}

int Platform::FilterSubmitExecute(Task* task) {
  if (!polyhedral_available_) return BRISBANE_OK;
  if (!task->cmd_kernel()) return BRISBANE_OK;
  if (task->brs_policy() & brisbane_all) {
    if (filter_task_split_->Execute(task) != BRISBANE_OK) {
      _trace("poly is not available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
      return BRISBANE_ERR;
    }
    _trace("poly is available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
  }
  return BRISBANE_OK;
}

Kernel* Platform::GetKernel(const char* name) {
  //todo: mutex lock
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    if (strcmp((*I)->name(), name) == 0) return *I;
  }
  Kernel* kernel = new Kernel(name, this);
  kernels_.insert(kernel);
  return kernel;
}

Mem* Platform::GetMem(brisbane_mem brs_mem) {
  //todo: mutex lock
  for (std::set<Mem*>::iterator I = mems_.begin(), E = mems_.end(); I != E; ++I) {
    Mem* mem = *I;
    if (mem->struct_obj() == brs_mem) return mem;
  }
  return NULL;
}

Mem* Platform::GetMem(void* host, size_t* off) {
  return present_table_->Get(host, off);
}

int Platform::TimerNow(double* time) {
  *time = timer_->Now();
  return BRISBANE_OK;
}

int Platform::InitScheduler() {
  if (ndevs_ == 1) {
    _info("No scheduler ndevs[%d]", ndevs_);
    return BRISBANE_OK;
  }
  _info("Scheduler ndevs[%d] ndevs_enabled[%d]", ndevs_, ndevs_enabled_);
  scheduler_ = new Scheduler(this);
  scheduler_->Start();
  return BRISBANE_OK;
}

int Platform::InitWorkers() {
  if (ndevs_ == 1) {
    workers_[0] = new Worker(devs_[0], this, true);
    workers_[0]->Start();
    return BRISBANE_OK;
  }
  for (int i = 0; i < ndevs_; i++) {
    workers_[i] = new Worker(devs_[i], this);
    workers_[i]->Start();
  }
  return BRISBANE_OK;
}

int Platform::ShowKernelHistory() {
  double t_ker = 0.0;
  double t_h2d = 0.0;
  double t_d2h = 0.0;
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    Kernel* kernel = *I;
    History* history = kernel->history();
    _info("kernel[%s] k[%lf][%zu] h2d[%lf][%zu] d2h[%lf][%zu]", kernel->name(), history->t_kernel(), history->c_kernel(), history->t_h2d(), history->c_h2d(), history->t_d2h(), history->c_d2h());
    t_ker += history->t_kernel();
    t_h2d += history->t_h2d();
    t_d2h += history->t_d2h();
  }
  _info("total kernel[%lf] h2d[%lf] d2h[%lf]", t_ker, t_h2d, t_d2h);
  return BRISBANE_OK;
}

Platform* Platform::singleton_ = NULL;
std::once_flag Platform::flag_singleton_;
std::once_flag Platform::flag_finalize_;

Platform* Platform::GetPlatform() {
//  if (singleton_ == NULL) singleton_ = new Platform();
  std::call_once(flag_singleton_, []() { singleton_ = new Platform(); });
  return singleton_;
}

int Platform::Finalize() {
  pthread_mutex_lock(&mutex_);
  if (finalize_) {
    pthread_mutex_unlock(&mutex_);
    return BRISBANE_ERR;
  }
  int ret_id = Synchronize();
  ShowKernelHistory();
  time_app_ = timer()->Stop(BRISBANE_TIMER_APP);
  time_init_ = timer()->Total(BRISBANE_TIMER_PLATFORM);
  _info("total execution time:[%lf] sec. initialize:[%lf] sec. t-i:[%lf] sec", time_app_, time_init_, time_app_ - time_init_);
  _info("t10[%lf] t11[%lf] t12[%lf] t13[%lf]", timer()->Total(10), timer()->Total(11), timer()->Total(12), timer()->Total(13));
  _info("t14[%lf] t15[%lf] t16[%lf] t17[%lf]", timer()->Total(14), timer()->Total(15), timer()->Total(16), timer()->Total(17));
  _info("t18[%lf] t19[%lf] t20[%lf] t21[%lf]", timer()->Total(18), timer()->Total(19), timer()->Total(20), timer()->Total(21));
  finalize_ = true;
  pthread_mutex_unlock(&mutex_);
  return ret_id;
}

} /* namespace rt */
} /* namespace brisbane */


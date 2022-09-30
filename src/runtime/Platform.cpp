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
#include "LoaderHost2HIP.h"
#include "LoaderHost2CUDA.h"
#include "LoaderHost2OpenCL.h"
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
#include "SchedulingHistory.h"
#include "QueueTask.h"
#include "Scheduler.h"
#include "SigHandler.h"
#include "Task.h"
#include "Timer.h"
#include "Worker.h"
#include <unistd.h>
#include <algorithm>

namespace iris {
namespace rt {

char iris_log_prefix_[256];

Platform::Platform() {
  init_ = false;
  finalize_ = false;
  nplatforms_ = 0;
  ndevs_ = 0;
  ndevs_enabled_ = 0;
  dev_default_ = 0;
  nfailures_ = 0;

  queue_ = NULL;
  pool_ = NULL;
  scheduler_ = NULL;
  polyhedral_ = NULL;
  sig_handler_ = NULL;
  filter_task_split_ = NULL;
  timer_ = NULL;
  null_kernel_ = NULL;
  loaderHost2HIP_ = NULL;
  loaderHost2CUDA_ = NULL;
  loaderCUDA_ = NULL;
  loaderHIP_ = NULL;
  loaderLevelZero_ = NULL;
  loaderOpenCL_ = NULL;
  loaderOpenMP_ = NULL;
  arch_available_ = 0UL;
  present_table_ = NULL;
  recording_ = false;
  enable_profiler_ = getenv("IRIS_PROFILE");
  enable_scheduling_history_ = getenv("IRIS_HISTORY");
  nprofilers_ = 0;
  time_app_ = 0.0;
  time_init_ = 0.0;
  hook_task_pre_ = NULL;
  hook_task_post_ = NULL;
  hook_command_pre_ = NULL;
  hook_command_post_ = NULL;
  scheduling_history_ = NULL;
  pthread_mutex_init(&mutex_, NULL);
}

Platform::~Platform() {
  if (!init_) return;
  if (scheduler_) delete scheduler_;
  for (int i = 0; i < ndevs_; i++) delete workers_[i];
  if (queue_) delete queue_;
  for(LoaderHost2OpenCL *ld : loaderHost2OpenCL_) {
      delete ld;
  }
  loaderHost2OpenCL_.clear();
  if (loaderHost2HIP_) delete loaderHost2HIP_;
  if (loaderHost2CUDA_) delete loaderHost2CUDA_;
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
  if (scheduling_history_) delete scheduling_history_;
  if (sig_handler_) delete sig_handler_;
  if (json_) delete json_;
  if (pool_) delete pool_;

  pthread_mutex_destroy(&mutex_);
}

int Platform::Init(int* argc, char*** argv, int sync) {
  pthread_mutex_lock(&mutex_);
  if (init_) {
    pthread_mutex_unlock(&mutex_);
    return IRIS_ERROR;
  }

  gethostname(iris_log_prefix_, 256);
  gethostname(host_, 256);
  if (argv && *argv) sprintf(app_, "%s", (*argv)[0]);
  else sprintf(app_, "%s", "app");

  timer_ = new Timer();
  timer_->Start(IRIS_TIMER_APP);

  timer_->Start(IRIS_TIMER_PLATFORM);
  sig_handler_ = new SigHandler();

  json_ = new JSON(this);

  EnvironmentInit();

  char* logo = NULL;
  EnvironmentGet("LOGO", &logo, NULL);
  if (strcmp("on", logo) == 0) Utils::Logo(true);

  char* tmpdir = NULL;
  EnvironmentGet("TMPDIR", &tmpdir, NULL);
  if (Utils::Mkdir(tmpdir) != IRIS_SUCCESS) {
    _error("tmpdir[%s]", tmpdir);
  }

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
  polyhedral_available_ = polyhedral_->Load() == IRIS_SUCCESS;
  if (polyhedral_available_)
    filter_task_split_ = new FilterTaskSplit(polyhedral_, this);

  iris_kernel null_brs_kernel;
  KernelCreate("iris_null", &null_brs_kernel);
  null_kernel_ = null_brs_kernel->class_obj;

  if (enable_profiler_) {
    profilers_[nprofilers_++] = new ProfilerDOT(this);
    profilers_[nprofilers_++] = new ProfilerGoogleCharts(this);
  }
  if (enable_scheduling_history_) scheduling_history_ = new SchedulingHistory(this);


  present_table_ = new PresentTable();
  queue_ = new QueueTask(this);
  pool_ = new Pool(this);

  InitScheduler();
  InitWorkers();
  InitDevices(sync);

  _info("nplatforms[%d] ndevs[%d] ndevs_enabled[%d] scheduler[%d] hub[%d] polyhedral[%d] profile[%d]",
      nplatforms_, ndevs_, ndevs_enabled_, scheduler_ != NULL, scheduler_ ? scheduler_->hub_available() : 0,
      polyhedral_available_, enable_profiler_);

  timer_->Stop(IRIS_TIMER_PLATFORM);

  init_ = true;

  pthread_mutex_unlock(&mutex_);

  return IRIS_SUCCESS;
}

int Platform::Synchronize() {
  int* devices = new int[ndevs_];
  for (int i = 0; i < ndevs_; i++) devices[i] = i;
  int ret = DeviceSynchronize(ndevs_, devices);
  delete devices;
  return nfailures_;
}

int Platform::EnvironmentInit() {
  EnvironmentSet("ARCHS",  "openmp:cuda:hip:levelzero:hexagon:opencl",  false);
  EnvironmentSet("TMPDIR", "/tmp/iris",                                 false);

  EnvironmentSet("KERNEL_SRC_CUDA",     "kernel.cu",          false);
  EnvironmentSet("KERNEL_BIN_CUDA",     "kernel.ptx",         false);
  EnvironmentSet("KERNEL_SRC_HEXAGON",  "kernel.hexagon.cpp", false);
  EnvironmentSet("KERNEL_BIN_HEXAGON",  "kernel.hexagon.so",  false);
  EnvironmentSet("KERNEL_SRC_HIP",      "kernel.hip.cpp",     false);
  EnvironmentSet("KERNEL_BIN_HIP",      "kernel.hip",         false);
  EnvironmentSet("KERNEL_SRC_OPENMP",   "kernel.openmp.h",    false);
  EnvironmentSet("KERNEL_BIN_OPENMP",   "kernel.openmp.so",   false);
  EnvironmentSet("KERNEL_SRC_SPV",      "kernel.cl",          false);
  EnvironmentSet("KERNEL_BIN_SPV",      "kernel.spv",         false);
  EnvironmentSet("KERNEL_HOST2CUDA","kernel.host2cuda.so",false);
  EnvironmentSet("KERNEL_HOST2HIP", "kernel.host2hip.so", false);
  EnvironmentSet("KERNEL_HOST2OPENCL","kernel.host2opencl.so",false);
  EnvironmentSet("KERNEL_HOST2OPENCL_FPGA","kernel.host2opencl.fpga.so",false);
  EnvironmentSet("KERNEL_HOST2OPENCL_XILINX","kernel.host2opencl.xilinx.so",false);
  EnvironmentSet("KERNEL_HOST2OPENCL_INTEL","kernel.host2opencl.intel.so",false);

  EnvironmentSet("LOGO",            "off",                false);
  return IRIS_SUCCESS;
}

int Platform::EnvironmentSet(const char* key, const char* value, bool overwrite) {
  std::string keystr = std::string(key);
  std::string valstr = std::string(value);
  auto it = env_.find(keystr);
  if (it != env_.end()) {
    if (!overwrite) return IRIS_ERROR;
    env_.erase(it);
  }
  env_.insert(std::pair<std::string, std::string>(keystr, valstr));
  return IRIS_SUCCESS;
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
      return IRIS_ERROR;
    }
    val = it->second.c_str();
  }

  if (*value == NULL) *value = (char*) malloc(strlen(val) + 1);
  strcpy(*value, val);
  if (vallen) *vallen = strlen(val) + 1;
  return IRIS_SUCCESS;
}

int Platform::SetDevsAvailable() {
  const char* enabled = getenv("IRIS_DEV_ENABLED");
  if (!enabled) {
    for (int i = 0; i < IRIS_MAX_NDEVS; i++) devs_enabled_[i] = i;
    ndevs_enabled_ = IRIS_MAX_NDEVS;
    return IRIS_SUCCESS;
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
  return IRIS_SUCCESS;
}

int Platform::InitCUDA() {
  if (arch_available_ & iris_nvidia) {
    _trace("%s", "skipping CUDA architecture");
    return IRIS_ERROR;
  }
  loaderCUDA_ = new LoaderCUDA();
  if (loaderCUDA_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping CUDA architecture");
    return IRIS_ERROR;
  }
  loaderHost2CUDA_ = new LoaderHost2CUDA();
  if (loaderHost2CUDA_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping Host2CUDA wrapper calls");
  }
  CUresult err = CUDA_SUCCESS;
  err = loaderCUDA_->cuInit(0);
  if (err != CUDA_SUCCESS) {
    _trace("skipping CUDA architecture CUDA_ERROR[%d]", err);
    return IRIS_ERROR;
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
    devs_[ndevs_] = new DeviceCUDA(loaderCUDA_, loaderHost2CUDA_, dev, ndevs_, nplatforms_);
    arch_available_ |= devs_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "CUDA");
    nplatforms_++;
  }
  return IRIS_SUCCESS;
}

int Platform::InitHIP() {
  if (arch_available_ & iris_amd) {
    _trace("%s", "skipping HIP architecture");
    return IRIS_ERROR;
  }
  loaderHIP_ = new LoaderHIP();
  if (loaderHIP_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping HIP architecture");
    return IRIS_ERROR;
  }
  loaderHost2HIP_ = new LoaderHost2HIP();
  if (loaderHost2HIP_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping Host2HIP wrapper calls");
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
    devs_[ndevs_] = new DeviceHIP(loaderHIP_, loaderHost2HIP_, dev, i, ndevs_, nplatforms_);
    arch_available_ |= devs_[ndevs_]->type();
    ndevs_++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "HIP");
    nplatforms_++;
  }
  return IRIS_SUCCESS;
}

int Platform::InitLevelZero() {
  if (arch_available_ & iris_gpu_intel) {
    _trace("%s", "skipping LevelZero architecture");
    return IRIS_ERROR;
  }
  loaderLevelZero_ = new LoaderLevelZero();
  if (loaderLevelZero_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping LevelZero architecture");
    return IRIS_ERROR;
  }

  ze_result_t err = ZE_RESULT_SUCCESS;
  err = loaderLevelZero_->zeInit(0);
  _zeerror(err);

  ze_driver_handle_t driver;
  uint32_t ndrivers = 0;
  err = loaderLevelZero_->zeDriverGet(&ndrivers, nullptr);
  _zeerror(err);

  _info("LevelZero driver count[%u]",  ndrivers);
  if (ndrivers != 1) return IRIS_ERROR;

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
  return IRIS_SUCCESS;
}

int Platform::InitOpenMP() {
  if (arch_available_ & iris_cpu) {
    _trace("%s", "skipping OpenMP architecture");
    return IRIS_ERROR;
  }
  loaderOpenMP_ = new LoaderOpenMP();
  if (loaderOpenMP_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping OpenMP architecture");
    return IRIS_ERROR;
  }
  _trace("OpenMP platform[%d] ndevs[%d]", nplatforms_, 1);
  devs_[ndevs_] = new DeviceOpenMP(loaderOpenMP_, ndevs_, nplatforms_);
  arch_available_ |= devs_[ndevs_]->type();
  ndevs_++;
  strcpy(platform_names_[nplatforms_], "OpenMP");
  nplatforms_++;
  return IRIS_SUCCESS;
}

int Platform::InitHexagon() {
  if (arch_available_ & iris_hexagon) {
    _trace("%s", "skipping Hexagon architecture");
    return IRIS_ERROR;
  }
  loaderHexagon_ = new LoaderHexagon();
  if (loaderHexagon_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping Hexagon architecture");
    return IRIS_ERROR;
  }
  _trace("Hexagon platform[%d] ndevs[%d]", nplatforms_, 1);
  devs_[ndevs_] = new DeviceHexagon(loaderHexagon_, ndevs_, nplatforms_);
  arch_available_ |= devs_[ndevs_]->type();
  ndevs_++;
  strcpy(platform_names_[nplatforms_], "Hexagon");
  nplatforms_++;
  return IRIS_SUCCESS;
}

int Platform::InitOpenCL() {
  loaderOpenCL_ = new LoaderOpenCL();
  if (loaderOpenCL_->Load() != IRIS_SUCCESS) {
    _trace("%s", "skipping OpenCL architecture");
    return IRIS_ERROR;
  }
  cl_platform_id cl_platforms[IRIS_MAX_NDEVS];
  cl_context cl_contexts[IRIS_MAX_NDEVS];
  cl_device_id cl_devices[IRIS_MAX_NDEVS];
  cl_int err;

  cl_uint nplatforms = IRIS_MAX_NDEVS;

  err = loaderOpenCL_->clGetPlatformIDs(nplatforms, cl_platforms, &nplatforms);
  _trace("OpenCL nplatforms[%u]", nplatforms);
  if (!nplatforms) return IRIS_SUCCESS;
  cl_uint ndevs = 0;
  char vendor[64];
  char platform_name[64];
  for (cl_uint i = 0; i < nplatforms; i++) {
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    _clerror(err);
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    _clerror(err);
    _trace("OpenCL platform[%s] from [%s]", platform_name, vendor);

    if ((arch_available_ & iris_nvidia) && strstr(vendor, "NVIDIA") != NULL) {
      _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
      continue;
    }
    if ((arch_available_ & iris_amd) && strstr(vendor, "Advanced Micro Devices") != NULL) {
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
      if ((arch_available_ & iris_cpu) && (dev_type == CL_DEVICE_TYPE_CPU)) continue;
      std::string suffix = DeviceOpenCL::GetLoaderHost2OpenCLSuffix(loaderOpenCL_, cl_devices[j]);
      LoaderHost2OpenCL *loaderHost2OpenCL = new LoaderHost2OpenCL(suffix.c_str());
      if (loaderHost2OpenCL->Load() != IRIS_SUCCESS) {
        _trace("%s", "skipping Host2OpenCL wrapper calls");
      }
      loaderHost2OpenCL_.push_back(loaderHost2OpenCL);
      devs_[ndevs_] = new DeviceOpenCL(loaderOpenCL_, loaderHost2OpenCL, cl_devices[j], cl_contexts[i], ndevs_, nplatforms_);
      arch_available_ |= devs_[ndevs_]->type();
      ndevs_++;
    }
    _trace("adding platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
    sprintf(platform_names_[nplatforms_], "OpenCL %s", vendor);
    nplatforms_++;
  }
  return IRIS_SUCCESS;
}

int Platform::InitDevices(bool sync) {
  if (!ndevs_) {
    dev_default_ = -1;
    ___error("%s", "NO AVAILABLE DEVICES!");
    return IRIS_ERROR;
  }
  char* c = getenv("IRIS_DEVICE_DEFAULT");
  if (c) dev_default_ = atoi(c);

  Task** tasks = new Task*[ndevs_];
  for (int i = 0; i < ndevs_; i++) {
    tasks[i] = new Task(this);
    tasks[i]->set_name("Initialize");
    tasks[i]->set_system();
    Command* cmd = Command::CreateInit(tasks[i]);
    tasks[i]->AddCommand(cmd);
    workers_[i]->Enqueue(tasks[i]);
  }
  if (sync) for (int i = 0; i < ndevs_; i++) tasks[i]->Wait();
  delete[] tasks;
  return IRIS_SUCCESS;
}

int Platform::PlatformCount(int* nplatforms) {
  if (nplatforms) *nplatforms = nplatforms_;
  return IRIS_SUCCESS;
}

int Platform::PlatformInfo(int platform, int param, void* value, size_t* size) {
  if (platform >= nplatforms_) return IRIS_ERROR;
  switch (param) {
    case iris_name:
      if (size) *size = strlen(platform_names_[platform]);
      strcpy((char*) value, platform_names_[platform]);
      break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int Platform::PlatformBuildProgram(int model, char* path) {
  for (int i = 0; i < ndevs_; i++)
    if (devs_[i]->model() == model) devs_[i]->BuildProgram(path);
  return IRIS_SUCCESS;
}

int Platform::DeviceCount(int* ndevs) {
  if (ndevs) *ndevs = ndevs_;
  return IRIS_SUCCESS;
}

int Platform::DeviceInfo(int device, int param, void* value, size_t* size) {
  if (device >= ndevs_) return IRIS_ERROR;
  Device* dev = devs_[device];
  switch (param) {
    case iris_platform  : if (size) *size = sizeof(int);            *((int*) value) = dev->platform();      break;
    case iris_vendor    : if (size) *size = strlen(dev->vendor());  strcpy((char*) value, dev->vendor());   break;
    case iris_name      : if (size) *size = strlen(dev->name());    strcpy((char*) value, dev->name());     break;
    case iris_type      : if (size) *size = sizeof(int);            *((int*) value) = dev->type();          break;
    default: return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int Platform::DeviceSetDefault(int device) {
  dev_default_ = device;
  return IRIS_SUCCESS;
}

int Platform::DeviceGetDefault(int* device) {
  *device = dev_default_;
  return IRIS_SUCCESS;
}

int Platform::DeviceSynchronize(int ndevs, int* devices) {
  Task* task = new Task(this, IRIS_MARKER, "Marker");
  if (scheduler_) {
    for (int i = 0; i < ndevs; i++) {
      if (devices[i] >= ndevs_) {
        _error("devices[%d]", devices[i]);
        continue;
      }
      Task* subtask = new Task(this, IRIS_MARKER, "Marker");
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
  return IRIS_SUCCESS;
}

int Platform::RegisterHooksTask(hook_task pre, hook_task post) {
  hook_task_pre_ = pre;
  hook_task_post_ = post;
  for (int i = 0; i < ndevs_; i++) devs_[i]->RegisterHooks();
  return IRIS_SUCCESS;
}

int Platform::RegisterHooksCommand(hook_command pre, hook_command post) {
  hook_command_pre_ = pre;
  hook_command_post_ = post;
  for (int i = 0; i < ndevs_; i++) devs_[i]->RegisterHooks();
  return IRIS_SUCCESS;
}

int Platform::KernelCreate(const char* name, iris_kernel* brs_kernel) {
  Kernel* kernel = new Kernel(name, this);
  if (brs_kernel) *brs_kernel = kernel->struct_obj();
  kernels_.insert(kernel);
  return IRIS_SUCCESS;
}

int Platform::KernelGet(const char* name, iris_kernel* brs_kernel) {
  for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
    Kernel* kernel = *I;
    if (strcmp(kernel->name(), name) == 0) {
      if (brs_kernel) *brs_kernel = kernel->struct_obj();
      return IRIS_SUCCESS;
    }
  }
  return KernelCreate(name, brs_kernel);
}

int Platform::KernelSetArg(iris_kernel brs_kernel, int idx, size_t size, void* value) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->SetArg(idx, size, value);
  return IRIS_SUCCESS;
}

int Platform::KernelSetMem(iris_kernel brs_kernel, int idx, iris_mem brs_mem, size_t off, size_t mode) {
  Kernel* kernel = brs_kernel->class_obj;
  Mem* mem = brs_mem->class_obj;
  kernel->SetMem(idx, mem, off, mode);
  return IRIS_SUCCESS;
}

int Platform::KernelSetMap(iris_kernel brs_kernel, int idx, void* host, size_t mode) {
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
  return IRIS_SUCCESS;
}

int Platform::KernelRelease(iris_kernel brs_kernel) {
  Kernel* kernel = brs_kernel->class_obj;
  kernel->Release();
  return IRIS_SUCCESS;
}

int Platform::TaskCreate(const char* name, bool perm, iris_task* brs_task) {
  Task* task = Task::Create(this, perm ? IRIS_TASK_PERM : IRIS_TASK, name);
  *brs_task = task->struct_obj();
  return IRIS_SUCCESS;
}

int Platform::TaskDepend(iris_task brs_task, int ntasks, iris_task* brs_tasks) {
  Task* task = brs_task->class_obj;
  for (int i = 0; i < ntasks; i++) task->AddDepend(brs_tasks[i]->class_obj);
  return IRIS_SUCCESS;
}

int Platform::TaskKernel(iris_task brs_task, iris_kernel brs_kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  Task* task = brs_task->class_obj;
  Kernel* kernel = brs_kernel->class_obj;
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, gws, lws);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskCustom(iris_task brs_task, int tag, void* params, size_t params_size) {
  Task* task = brs_task->class_obj;
  Command* cmd = Command::CreateCustom(task, tag, params, params_size);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskKernel(iris_task brs_task, const char* name, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges) {
  Task* task = brs_task->class_obj;
  Kernel* kernel = GetKernel(name);
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, memranges);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::SetParamsMap(iris_task brs_task, int *params_map)
{
  Task *task = brs_task->class_obj;
  Command *cmd_kernel = task->cmd_kernel();
  cmd_kernel->set_params_map(params_map);
  return IRIS_SUCCESS;
}

int Platform::SetSharedMemoryModel(int flag)
{
    for (int i = 0; i < ndevs_; i++) {
        if (devs_[i]) devs_[i]->set_shared_memory_buffers((bool)flag);
    }
    return IRIS_SUCCESS;
}

int Platform::TaskKernelSelector(iris_task brs_task, iris_selector_kernel func, void* params, size_t params_size) {
  Task* task = brs_task->class_obj;
  Command* cmd = task->cmd_kernel();
  if (!cmd) return IRIS_ERROR;
  cmd->set_selector_kernel(func, params, params_size);
  return IRIS_SUCCESS;
}

int Platform::TaskHost(iris_task brs_task, iris_host_task func, void* params) {
  Task* task = brs_task->class_obj;
  Command* cmd = Command::CreateHost(task, func, params);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMalloc(iris_task brs_task, iris_mem brs_mem) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateMalloc(task, mem);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2D(iris_task brs_task, iris_mem brs_mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateH2D(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2D(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskD2H(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskD2H(iris_task brs_task, iris_mem brs_mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateD2H(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2DFull(iris_task brs_task, iris_mem brs_mem, void* host) {
  return TaskH2D(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskD2HFull(iris_task brs_task, iris_mem brs_mem, void* host) {
  return TaskD2H(brs_task, brs_mem, 0ULL, brs_mem->class_obj->size(), host);
}

int Platform::TaskMap(iris_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  Command* cmd = Command::CreateMap(task, host, size);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapTo(iris_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapToFull(iris_task brs_task, void* host) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  size_t size = mem->size();
  Command* cmd = Command::CreateH2D(task, mem, off, size - off, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapFrom(iris_task brs_task, void* host, size_t size) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapFromFull(iris_task brs_task, void* host) {
  Task* task = brs_task->class_obj;
  size_t off = 0ULL;
  Mem* mem = present_table_->Get(host, &off);
  size_t size = mem->size();
  Command* cmd = Command::CreateD2H(task, mem, off, size - off, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

void Platform::IncrementErrorCount(){
  nfailures_++;
}

int Platform::NumErrors(){
  return nfailures_;
}

int Platform::TaskSubmit(iris_task brs_task, int brs_policy, const char* opt, int sync) {
  Task* task = brs_task->class_obj;
  task->Submit(brs_policy, opt, sync);
  if (recording_) json_->RecordTask(task);
  if (enable_profiler_){
    for (int i = 0; i < nprofilers_; i++) profilers_[i]->CompleteTask(task);
  }
  if (scheduler_) {
    FilterSubmitExecute(task);
    scheduler_->Enqueue(task);
  } else workers_[0]->Enqueue(task);
  if (sync) task->Wait();
  return nfailures_;
}

int Platform::TaskWait(iris_task brs_task) {
  Task* task = brs_task->class_obj;
  task->Wait();
  return IRIS_SUCCESS;
}

int Platform::TaskWaitAll(int ntasks, iris_task* brs_tasks) {
  int iret = IRIS_SUCCESS;
  for (int i = 0; i < ntasks; i++) iret &= TaskWait(brs_tasks[i]);
  return iret;
}

int Platform::TaskAddSubtask(iris_task brs_task, iris_task brs_subtask) {
  Task* task = brs_task->class_obj;
  Task* subtask = brs_subtask->class_obj;
  task->AddSubtask(subtask);
  return IRIS_SUCCESS;
}

int Platform::TaskKernelCmdOnly(iris_task brs_task) {
  Task* task = brs_task->class_obj;
  return (task->ncmds() == 1 && task->cmd_kernel()) ? IRIS_SUCCESS : IRIS_ERROR;
}

int Platform::TaskRelease(iris_task brs_task) {
  Task* task = brs_task->class_obj;
  task->Release();
  return IRIS_SUCCESS;
}

int Platform::TaskReleaseMem(iris_task brs_task, iris_mem brs_mem) {
  Task* task = brs_task->class_obj;
  Mem* mem = brs_mem->class_obj;
  Command* cmd = Command::CreateReleaseMem(task, mem);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskInfo(iris_task brs_task, int param, void* value, size_t* size) {
  Task* task = brs_task->class_obj;
  if (param == iris_ncmds) {
    if (size) *size = sizeof(int);
    *((int*) value) = task->ncmds();
  } else if (param == iris_ncmds_kernel) {
    if (size) *size = sizeof(int);
    *((int*) value) = task->ncmds_kernel();
  } else if (param == iris_ncmds_memcpy) {
    if (size) *size = sizeof(int);
    *((int*) value) = task->ncmds_memcpy();
  } else if (param == iris_cmds) {
    if (size) *size = sizeof(int) * task->ncmds();
    int* cmd_types = (int*) value;
    for (int i = 0; i < task->ncmds(); i++) {
      cmd_types[i] = task->cmd(i)->type();
    }    
  }
  return IRIS_SUCCESS;
}

int Platform::MemCreate(size_t size, iris_mem* brs_mem) {
  Mem* mem = new Mem(size, this);
  if (brs_mem) *brs_mem = mem->struct_obj();
  if (mem->size()==0) return IRIS_ERROR;

  mems_.insert(mem);
  return IRIS_SUCCESS;
}

int Platform::MemSetIntermediate(iris_mem brs_mem, bool flag) {
  Mem* mem = brs_mem->class_obj;
  mem->set_intermediate(true);
  return IRIS_SUCCESS;
}

int Platform::MemArch(iris_mem brs_mem, int device, void** arch) {
  if (!arch) return IRIS_ERROR;
  Mem* mem = brs_mem->class_obj;
  Device* dev = devs_[device];
  void* ret = mem->arch(dev);
  if (!ret) return IRIS_ERROR;
  *arch = ret;
  return IRIS_SUCCESS;
}

int Platform::MemMap(void* host, size_t size) {
  Mem* mem = new Mem(size, this);
  mem->SetMap(host, size);
  mems_.insert(mem);
  present_table_->Add(host, size, mem);
  return IRIS_SUCCESS;
}

int Platform::MemUnmap(void* host) {
  Mem* mem = present_table_->Remove(host);
  mem->Release();
  return IRIS_SUCCESS;
}

int Platform::MemReduce(iris_mem brs_mem, int mode, int type) {
  Mem* mem = brs_mem->class_obj;
  mem->Reduce(mode, type);
  return IRIS_SUCCESS;
}

int Platform::MemRelease(iris_mem brs_mem) {
  Mem* mem = brs_mem->class_obj;
  mem->Release();
  return IRIS_SUCCESS;
}

int Platform::GraphCreate(iris_graph* brs_graph) {
  Graph* graph = Graph::Create(this);
  *brs_graph = graph->struct_obj();
  return IRIS_SUCCESS;
}

int Platform::GraphCreateJSON(const char* path, void** params, iris_graph* brs_graph) {
  Graph* graph = Graph::Create(this);
  *brs_graph = graph->struct_obj();
  int retcode = json_->Load(graph, path, params);
  return retcode;
}

int Platform::GraphTask(iris_graph brs_graph, iris_task brs_task, int brs_policy, const char* opt) {
  Graph* graph = brs_graph->class_obj;
  Task* task = brs_task->class_obj;
  task->set_brs_policy(brs_policy);
  task->set_opt(opt);
  graph->AddTask(task);
  return IRIS_SUCCESS;
}

int Platform::GraphSubmit(iris_graph brs_graph, int brs_policy, int sync) {
  Graph* graph = brs_graph->class_obj;
  std::vector<Task*>* tasks = graph->tasks();
  for (std::vector<Task*>::iterator I = tasks->begin(), E = tasks->end(); I != E; ++I) {
    Task* task = *I;
    //preference is to honour the policy embedded in the task-graph.
    if (task->brs_policy() == iris_default) {
      task->set_brs_policy(brs_policy);
    }
    task->Submit(task->brs_policy(), task->opt(), sync);
    if (recording_) json_->RecordTask(task);
    if (scheduler_) scheduler_->Enqueue(task);
    else workers_[0]->Enqueue(task);
  }
  if (sync) graph->Wait();
  return IRIS_SUCCESS;
}

int Platform::GraphWait(iris_graph brs_graph) {
  Graph* graph = brs_graph->class_obj;
  graph->Wait();
  return IRIS_SUCCESS;
}

int Platform::GraphWaitAll(int ngraphs, iris_graph* brs_graphs) {
  int iret = IRIS_SUCCESS;
  for (int i = 0; i < ngraphs; i++) iret &= GraphWait(brs_graphs[i]);
  return iret;
}

int Platform::RecordStart() {
  recording_ = true;
  return IRIS_SUCCESS;
}

int Platform::RecordStop() {
  json_->RecordFlush();
  recording_ = false;
  return IRIS_SUCCESS;
}

int Platform::FilterSubmitExecute(Task* task) {
  if (!polyhedral_available_) return IRIS_SUCCESS;
  if (!task->cmd_kernel()) return IRIS_SUCCESS;
  if (task->brs_policy() & iris_all) {
    if (filter_task_split_->Execute(task) != IRIS_SUCCESS) {
      _trace("poly is not available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
      return IRIS_ERROR;
    }
    _trace("poly is available kernel[%s] task[%lu]", task->cmd_kernel()->kernel()->name(), task->uid());
  }
  return IRIS_SUCCESS;
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

Mem* Platform::GetMem(iris_mem brs_mem) {
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
  return IRIS_SUCCESS;
}

int Platform::InitScheduler() {
  /*
  if (ndevs_ == 1) {
    _info("No scheduler ndevs[%d]", ndevs_);
    return IRIS_SUCCESS;
  }
  */
  _info("Scheduler ndevs[%d] ndevs_enabled[%d]", ndevs_, ndevs_enabled_);
  scheduler_ = new Scheduler(this);
  scheduler_->Start();
  return IRIS_SUCCESS;
}

int Platform::InitWorkers() {
  if (!scheduler_) {
    workers_[0] = new Worker(devs_[0], this, true);
    workers_[0]->Start();
    return IRIS_SUCCESS;
  }
  for (int i = 0; i < ndevs_; i++) {
    workers_[i] = new Worker(devs_[i], this);
    workers_[i]->Start();
  }
  return IRIS_SUCCESS;
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
  return IRIS_SUCCESS;
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
    return IRIS_ERROR;
  }
  int ret_id = Synchronize();
  ShowKernelHistory();
  time_app_ = timer()->Stop(IRIS_TIMER_APP);
  time_init_ = timer()->Total(IRIS_TIMER_PLATFORM);
  _info("total execution time:[%lf] sec. initialize:[%lf] sec. t-i:[%lf] sec", time_app_, time_init_, time_app_ - time_init_);
  _info("t10[%lf] t11[%lf] t12[%lf] t13[%lf]", timer()->Total(10), timer()->Total(11), timer()->Total(12), timer()->Total(13));
  _info("t14[%lf] t15[%lf] t16[%lf] t17[%lf]", timer()->Total(14), timer()->Total(15), timer()->Total(16), timer()->Total(17));
  _info("t18[%lf] t19[%lf] t20[%lf] t21[%lf]", timer()->Total(18), timer()->Total(19), timer()->Total(20), timer()->Total(21));
  finalize_ = true;
  pthread_mutex_unlock(&mutex_);
 if (scheduling_history_) delete scheduling_history_;
   return ret_id;
}

} /* namespace rt */
} /* namespace iris */


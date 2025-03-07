#include "Platform.h"
#include "Debug.h"
#include "Utils.h"
#include "Command.h"
#include "DeviceCUDA.h"
#include "DeviceQIREE.h"
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
#include "LoaderQIREE.h"
#include "LoaderHost2HIP.h"
#include "LoaderHost2CUDA.h"
#include "LoaderHost2OpenCL.h"
#include "LoaderHexagon.h"
#include "LoaderHIP.h"
#include "LoaderLevelZero.h"
#include "LoaderOpenCL.h"
#include "LoaderOpenMP.h"
#include "Mem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include "Policies.h"
#include "Polyhedral.h"
#include "Pool.h"
#include "PresentTable.h"
#include "Profiler.h"
#include "ProfilerDOT.h"
#include "ProfilerGoogleCharts.h"
#include "ProfilerEventRecord.h"
#include "SchedulingHistory.h"
#include "QueueTask.h"
#include "Scheduler.h"
#include "SigHandler.h"
#include "Task.h"
#include "Timer.h"
#include "Worker.h"
#ifdef AUTO_PAR
#include "AutoDAG.h"
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace iris {
namespace rt {

char iris_log_prefix_[256];

Platform::Platform() {
  Reset();
  pthread_mutex_init(&mutex_, NULL);
}

Platform::~Platform() {
  pthread_mutex_destroy(&mutex_);
  _debug2("Platform deleted");
}

void Platform::Reset() {
  init_ = false;
  tmp_dir_[0] = '\0';
  disable_d2d_ = false;
  disable_data_transfers_ = false;
  dmem_register_pin_flag_ = true;
  finalize_ = false;
  release_task_flag_ = true;
  async_ = false;
  nplatforms_ = 0;
  ndevs_ = 0;
  ndevs_enabled_ = 0;
  disable_init_devices_ = false;
  disable_init_workers_ = false;
  enable_default_kernels_load_ = false;
  disable_init_scheduler_ = false;
  dev_default_ = 0;
  nfailures_ = 0;

  queue_ = NULL;
  pool_ = NULL;
  scheduler_ = NULL;
  polyhedral_ = NULL;
  sig_handler_ = NULL;
  openmp_device_factor_ = 1;
  qiree_device_factor_ = 1;
  cuda_device_factor_ = 1;
  hip_device_factor_ = 1;
  event_profile_enabled_ = false;
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
  loaderQIREE_ = NULL;
  loaderHexagon_ = NULL;
  arch_available_ = 0UL;
  present_table_ = NULL;
  stream_policy_ = STREAM_POLICY_DEFAULT;
  nstreams_ = IRIS_MAX_DEVICE_NSTREAMS;
  ncopy_streams_ = IRIS_MAX_DEVICE_NCOPY_STREAMS;
  is_malloc_async_ = false;
  recording_ = false;
  enable_profiler_ = false;
  enable_scheduling_history_ = false; 
  nprofilers_ = 0;
  time_app_ = 0.0;
  time_init_ = 0.0;
  hook_task_pre_ = NULL;
  hook_task_post_ = NULL;
  hook_command_pre_ = NULL;
  hook_command_post_ = NULL;
  scheduling_history_ = NULL;
  enable_proactive_ = false;
  disable_kernel_launch_ = false;
#ifdef AUTO_PAR
  auto_dag_ = NULL;
  enable_auto_par_ = true;
#else
  enable_auto_par_ = false;
#endif
}

void Platform::Clean() {
  for (int i = 0; i < ndevs_; i++) delete workers_[i];
  if (queue_) delete queue_;
  if (tmp_dir_[0] != '\0') {
      char cmd[270];
      //printf("Removing tmp_dir:%s\n", tmp_dir_);
      sprintf(cmd, "rm -rf %s", tmp_dir_);
      int result = system(cmd);
      (void)result;
  }
  for (int i = 0; i < ndevs_; i++) delete devs_[i];
  if (scheduler_) delete scheduler_;
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
  printf("Total Shadow created %d\n", auto_dag_->get_number_of_shadow());
  _trace("Total Shadow created[%d]", auto_dag_->get_number_of_shadow());
#endif
  delete auto_dag_;
#endif
  /*
  for (std::set<Mem*>::iterator I = mems_.begin(), E = mems_.end(); I != E; ++I) {
      Mem *mem = *I;
      delete mem;
  }*/
  // All loaders should be cleared now only
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
  if (loaderQIREE_) delete loaderQIREE_;
  if (loaderHexagon_) delete loaderHexagon_;
  if (present_table_) delete present_table_;
  if (polyhedral_) delete polyhedral_;
  if (filter_task_split_) delete filter_task_split_;
  if (timer_) delete timer_;
  if (null_kernel_) delete null_kernel_;
  for (int i = 0; i < nprofilers_; i++) delete profilers_[i];
  if (sig_handler_) delete sig_handler_;
  if (json_) delete json_;
  if (pool_) delete pool_;
  kernel_history_.clear();
}

int Platform::JuliaInit(bool decoupled_init) {
#if 1
  if (decoupled_init) {
    disable_init_devices_ = true;
    disable_init_workers_ = true;
    disable_init_scheduler_ = true;
  }
#endif
  enable_default_kernels_load_ = true;
  return IRIS_SUCCESS;
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

  EnvironmentBoolRead("DISABLE_KERNEL_LAUNCH", disable_kernel_launch_);
  EnvironmentBoolRead("PROFILE", enable_profiler_);
  EnvironmentBoolRead("HISTORY", enable_scheduling_history_);
  EnvironmentBoolRead("EVENT_PROFILE", event_profile_enabled_);
  EnvironmentBoolRead("MALLOC_ASYNC", is_malloc_async_);
  EnvironmentBoolRead("DISABLE_D2D", disable_d2d_);
  EnvironmentBoolRead("DISABLE_DATA_TRANSFERS", disable_data_transfers_);
  EnvironmentIntRead("OPENMP_DEVICE_FACTOR", openmp_device_factor_);
  EnvironmentIntRead("QIREE_DEVICE_FACTOR", qiree_device_factor_);
  EnvironmentIntRead("CUDA_DEVICE_FACTOR", cuda_device_factor_);
  EnvironmentIntRead("HIP_DEVICE_FACTOR", hip_device_factor_);
  EnvironmentIntRead("NSTREAMS", nstreams_);
  EnvironmentIntRead("NCOPY_STREAMS", ncopy_streams_);
  int stream_policy = (int) stream_policy_;
  EnvironmentIntRead("STREAM_POLICY", stream_policy);
  stream_policy_ = (StreamPolicy) stream_policy;
  bool async = async_; EnvironmentBoolRead("ASYNC", async); set_async(async);
#ifdef IRIS_ASYNC_STREAMING
  set_async(true);
#endif
  if (is_async()) 
      _info("Asynchronous is enabled");
  char* archs = NULL;
  EnvironmentGet("ARCHS", &archs, NULL);
  if (archs == NULL) 
      EnvironmentGet("ARCH", &archs, NULL);
  _info("IRIS architectures[%s]", archs);
  const char* delim = " :;.,";
  std::string arch_str = std::string(archs);
#if 0
  char arch_str[128];
  memset(arch_str, 0, 128);
  strncpy(arch_str, archs, strlen(archs)+1);
#endif
  char* rest = (char *)arch_str.c_str();
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
    } else if (strcasecmp(a, "qiree") == 0) {
      if (!loaderQIREE_) InitQIREE();
    } else if (strcasecmp(a, "hexagon") == 0) {
      if (!loaderHexagon_) InitHexagon();
    } else _error("not support arch[%s]", a);
  }
  if (ndevs_enabled_ > ndevs_) ndevs_enabled_ = ndevs_;
  polyhedral_ = new Polyhedral();
  polyhedral_available_ = false;
#ifndef ENABLE_RISCV
  polyhedral_available_ = polyhedral_->Load() == IRIS_SUCCESS;
#endif
  if (polyhedral_available_)
    filter_task_split_ = new FilterTaskSplit(polyhedral_, this);

  if (enable_profiler_) {
    profilers_[nprofilers_++] = new ProfilerDOT(this);
    profilers_[nprofilers_++] = new ProfilerGoogleCharts(this);
    profilers_[nprofilers_++] = new ProfilerGoogleCharts(this, true);
  }
  if (event_profile_enabled_) 
    profilers_[nprofilers_++] = new ProfilerEventRecord(this);

  if (enable_scheduling_history_) scheduling_history_ = new SchedulingHistory(this);


  present_table_ = new PresentTable();
  queue_ = new QueueTask(this);
  pool_ = new Pool(this);

  iris_kernel null_brs_kernel;
  KernelCreate("iris_null", &null_brs_kernel);
  null_kernel_ = get_kernel_object(null_brs_kernel);

  if (!disable_init_scheduler_) 
    InitScheduler();
  if (!disable_init_workers_) 
    InitWorkers();
  if (!disable_init_devices_) 
    InitDevices(sync);
  

#ifdef AUTO_PAR
  auto_dag_ = new AutoDAG(this, false);
#endif


  _info("nplatforms[%d] ndevs[%d] ndevs_enabled[%d] scheduler[%d] hub[%d] polyhedral[%d] profile[%d]",
      nplatforms_, ndevs_, ndevs_enabled_, scheduler_ != NULL, scheduler_ ? scheduler_->hub_available() : 0,
      polyhedral_available_, enable_profiler_);

  timer_->Stop(IRIS_TIMER_PLATFORM);

  init_ = true;
  finalize_ = false;
  pthread_mutex_unlock(&mutex_);
  return IRIS_SUCCESS;
}

int Platform::InitWorker(int dev)
{
  if (!disable_init_workers_) return IRIS_SUCCESS;
  ASSERT(dev < ndevs_);
  workers_[dev] = new Worker(devs_[dev], this);
  return IRIS_SUCCESS;
}
int Platform::StartWorker(int dev, bool use_pthread)
{
  //fflush(stderr);
  ASSERT(dev < ndevs_);
  if (use_pthread)
      workers_[dev]->Start();
  else {
      //fflush(stdout);
      workers_[dev]->StartWithOutThread();
      //fflush(stdout);
  }
  return IRIS_SUCCESS;
}

int Platform::InitDevice(int dev)
{
  int i = dev;
  char task_name[128];
  sprintf(task_name, "Initialize-%d", i);
  //tasks[i] = new Task(this, IRIS_TASK, task_name);
  TaskCreate(task_name, false, &init_tasks_[i]);
  Task *task = get_task_object(init_tasks_[i]);
  task->set_system();
  task->Retain();
  Command* cmd = Command::CreateInit(task);
  task->AddCommand(cmd);
  _debug2("Initialize task:%lu:%s ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
  task->Retain();
  task->Retain();
  workers_[i]->Enqueue(task);
  return IRIS_SUCCESS;
}
int Platform::InitDevicesSynchronize(int sync)
{
  if (!ndevs_) {
    dev_default_ = -1;
    ___error("%s", "NO AVAILABLE DEVICES!");
    return IRIS_ERROR;
  }
  char* c = getenv("IRIS_DEVICE_DEFAULT");
  if (c) dev_default_ = atoi(c);
  if (sync) for (int i = 0; i < ndevs_; i++) {
    Task *task = get_task_object(init_tasks_[i]);
    task->Wait();
    task->Release();
  }
  Synchronize();
  for(int i=0; i<ndevs_; i++) {
    devs_[i]->EnablePeerAccess();
  }
  return IRIS_SUCCESS;
}

int Platform::Synchronize() {
  int* devices = new int[ndevs_];
  for (int i = 0; i < ndevs_; i++) devices[i] = i;
  int ret = DeviceSynchronize(ndevs_, devices);
  delete [] devices;
  //track().Clear();
  task_track().Clear();
  return ret;
}
void Platform::EnvironmentIntRead(const char *env_name, int & env_var) {
    char *env_val_char = NULL;
    EnvironmentGet(env_name, &env_val_char, NULL);
    if (env_val_char != NULL && atoi(env_val_char) >=0 )
        env_var = atoi(env_val_char);
}
void Platform::EnvironmentBoolRead(const char *env_name, bool & flag) {
    char *env_val_char = NULL;
    EnvironmentGet(env_name, &env_val_char, NULL);
    if (env_val_char != NULL && atoi(env_val_char) == 1)
        flag = true;
    else if (env_val_char != NULL && atoi(env_val_char) == 0)
        flag = false;
}
int Platform::EnvironmentInit() {
#ifdef ANDROID
  char tmp_dir_str[256];
  char current_dir[256];
  getcwd(current_dir, 256);
  sprintf(tmp_dir_str, "%s/tmp", current_dir);
  mkdir(tmp_dir_str, 0700);
  sprintf(tmp_dir_str, "%s/iris-XXXXXX", tmp_dir_str);
#else
  char tmp_dir_str[] = "/tmp/iris-XXXXXX";
#endif
  char *tmp_dir = mkdtemp(tmp_dir_str);
  strcpy(tmp_dir_, tmp_dir);
  //printf("Temp directory:%s\n", tmp_dir_);
#ifdef ENABLE_RISCV
  EnvironmentSet("ARCHS",  "openmp",  false);
#else
  // Removed qiree 
  EnvironmentSet("ARCHS",  "openmp:cuda:hip:levelzero:hexagon:opencl",  false);
#endif
  EnvironmentSet("DEFAULT_OMP_KERNELS",  "default_cpu_gpu_kernels.cpp", false);
  EnvironmentSet("DEFAULT_CUDA_KERNELS", "default_cpu_gpu_kernels.cpp", false);
  EnvironmentSet("DEFAULT_HIP_KERNELS",  "default_cpu_gpu_kernels.cpp", false);
  EnvironmentSet("TMPDIR",               tmp_dir_,               false);
  EnvironmentSet("OPENCL_VENDORS",       "all",                  false);
  EnvironmentSet("INCLUDE_DIR",          "include/iris",         false);
  EnvironmentSet("KERNEL_DIR",           "",                     false);
  EnvironmentSet("KERNEL_SRC_CUDA",      "kernel.cu",            false);
  EnvironmentSet("KERNEL_BIN_CUDA",      "kernel.ptx",           false);
  EnvironmentSet("KERNEL_XILINX_XCLBIN", "kernel.xilinx.xclbin", false);
  EnvironmentSet("KERNEL_FPGA_XCLBIN",   "kernel-fpga.xclbin",   false);
  EnvironmentSet("KERNEL_INTEL_AOCX",    "kernel.intel.aocx",    false);
  EnvironmentSet("KERNEL_SRC_HEXAGON",   "kernel.hexagon.cpp",   false);
  EnvironmentSet("KERNEL_BIN_HEXAGON",   "kernel.hexagon.so",    false);
  EnvironmentSet("KERNEL_SRC_HIP",       "kernel.hip.cpp",       false);
  EnvironmentSet("KERNEL_BIN_HIP",       "kernel.hip",           false);
  EnvironmentSet("KERNEL_SRC_OPENMP",    "kernel.openmp.h",      false);
  EnvironmentSet("KERNEL_BIN_OPENMP",    "kernel.openmp.so",     false);
  EnvironmentSet("KERNEL_QIR",           "kernel.qir.ll",        false);
  EnvironmentSet("KERNEL_SRC_SPV",       "kernel.cl",            false);
  EnvironmentSet("KERNEL_BIN_SPV",       "kernel.spv",           false);
  EnvironmentSet("KERNEL_JULIA",         "libjulia.so",          false);
  EnvironmentSet("LIB_CUDA",             "libcuda.so",           false);
  EnvironmentSet("LIB_QIREE",            "libqir.xacc.lib.so",           false);
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
int Platform::GetFilePath(const char *key, char **value, size_t* vallen)
{
    bool is_malloc = false;
    if (*value == NULL)  is_malloc = true;
    EnvironmentGet(key, value, vallen);
    //printf("Key:%s value:%s\n", key, *value);
    std::ifstream infile(*value);
    if (! infile.good()) {
        char *libdir=NULL; 
        EnvironmentGet("KERNEL_DIR", &libdir, NULL);
        //printf("KD: Key:%s value:%s\n", key, libdir);
        if (strlen(libdir) != 0) {
            char output[1024]; 
            strcpy(output, (const char*) libdir);
#ifdef _WIN32
            strcat(output, "\\");
#else
            strcat(output, "/");
#endif
            strcat(output, *value);
            if (is_malloc) {
                free(*value);
                *value = (char *)malloc(strlen(output)+1);
            }
            strcpy(*value, output);
            if (vallen) *vallen = strlen(*value) + 1;
        }
        //printf("Output:%s\n", *value);
    }
    return IRIS_SUCCESS;
}
int Platform::EnvironmentGet(const char* key, char** value, size_t* vallen, char sep) {
  char env_key[128];
  if (sep == '\0')
      sprintf(env_key, "IRIS%s", key);
  else
      sprintf(env_key, "IRIS%c%s", sep, key);
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
  std::string str = enabled;
#if 0
  char str[128];
  memset(str, 0, 128);
  strncpy(str, enabled, strlen(enabled)+1);
#endif
  char* rest = (char *)str.c_str();
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
  //if (arch_available_ & iris_nvidia) {
  //  _trace("%s", "skipping CUDA architecture");
  //  return IRIS_ERROR;
  //}
  loaderCUDA_ = new LoaderCUDA();
  if (loaderCUDA_->LoadExtHandle("libcudart.so") != IRIS_SUCCESS) {
    _trace("%s", "skipping CUDA RT architecture");
    return IRIS_ERROR;
  }
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

  int max_ndevs = -1;
  EnvironmentIntRead("MAX_CUDA_DEVICE", max_ndevs);
  // added the following to limit the number of devices
  char* c = getenv("IRIS_MAX_CUDA_DEVICE");
  if (c) ndevs = atoi(c);

  _trace("CUDA platform[%d] ndevs[%d]", nplatforms_, ndevs);
  int mdevs =0;
  int *cudevs = new int[ndevs*cuda_device_factor_];
  for (int i = 0; i < ndevs*cuda_device_factor_; i++) {
    if (ndevs_ > IRIS_MAX_NDEVS) {
        _error("This platform has more than max devices: %d ! Hence ignoring further CUDA devices\n", IRIS_MAX_NDEVS);
        break;
    }
    CUdevice dev;
    err = loaderCUDA_->cuDeviceGet(&dev, i%ndevs);
    _cuerror(err);
    devs_[ndevs_] = new DeviceCUDA(loaderCUDA_, loaderHost2CUDA_, dev,
            i%ndevs, ndevs_, nplatforms_, i);
    devs_[ndevs_]->set_root_device(devs_[ndevs_-i]);
    if (is_julia_enabled()) 
        devs_[ndevs_]->EnableJuliaInterface();
    arch_available_ |= devs_[ndevs_]->type();
    cudevs[mdevs] = dev;
    ndevs_++;
    mdevs++;
    if (max_ndevs != -1 && mdevs >= max_ndevs) break;
#ifdef ENABLE_SINGLE_DEVICE_PER_CU
    break;
#endif
  }
  for(int i=0; i<mdevs; i++) {
      DeviceCUDA *idev = (DeviceCUDA *)devs_[ndevs_-mdevs+i];
      idev->SetPeerDevices(cudevs, mdevs);
  }
#if 0
  for(int i=0; i<mdevs; i++) {
      for(int j=0; j<mdevs; j++) {
          if (i != j) {
              //printf("i:%d j:%d ii:%d jj:%d\n",i,j,ndevs_-mdevs+i,ndevs_-mdevs+j);
              DeviceCUDA *jdev = (DeviceCUDA *)devs_[ndevs_-mdevs+j];
              idev->EnablePeerAccess(jdev->cudev());
          }
      }
  }
#endif
  delete [] cudevs;
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "CUDA");
    first_dev_of_type_[nplatforms_] = devs_[ndevs_-mdevs];
    nplatforms_++;
  }
  return IRIS_SUCCESS;
}

int Platform::InitHIP() {
  //if (arch_available_ & iris_amd) {
  //  _trace("%s", "skipping HIP architecture");
  //  return IRIS_ERROR;
  //}
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

  int max_ndevs = -1;
  EnvironmentIntRead("MAX_HIP_DEVICE", max_ndevs);
  // added the following to limit the number of devices
  char* c = getenv("IRIS_MAX_HIP_DEVICE");
  if (c) ndevs = atoi(c);

  _trace("HIP platform[%d] ndevs[%d]", nplatforms_, ndevs);
  int mdevs =0;
  int *hipdevs= new int[ndevs*hip_device_factor_];
  for (int i = 0; i < ndevs*hip_device_factor_; i++) {
    if (ndevs_ > IRIS_MAX_NDEVS) {
        _error("This platform has more than max devices: %d ! Hence ignoring further HIP devices\n", IRIS_MAX_NDEVS);
        break;
    }
    hipDevice_t dev;
    err = loaderHIP_->hipDeviceGet(&dev, i%ndevs);
    _hiperror(err);
    devs_[ndevs_] = new DeviceHIP(loaderHIP_, loaderHost2HIP_, dev,
            i%ndevs, ndevs_, nplatforms_, i);
    devs_[ndevs_]->set_root_device(devs_[ndevs_-i]);
    if (is_julia_enabled()) 
        devs_[ndevs_]->EnableJuliaInterface();
    arch_available_ |= devs_[ndevs_]->type();
    hipdevs[mdevs] = dev;
    ndevs_++;
    mdevs++;
    if (max_ndevs != -1 && mdevs >= max_ndevs) break;
#ifdef ENABLE_SINGLE_DEVICE_PER_CU
    break;
#endif
  }
  for(int i=0; i<mdevs; i++) {
      DeviceHIP *idev = (DeviceHIP *)devs_[ndevs_-mdevs+i];
      idev->SetPeerDevices(hipdevs, mdevs);
      /*
      for(int j=0; j<mdevs; j++) {
          if (i != j) {
              //printf("i:%d j:%d ii:%d jj:%d\n",i,j,ndevs_-mdevs+i,ndevs_-mdevs+j);
              DeviceHIP *jdev = (DeviceHIP *)devs_[ndevs_-mdevs+j];
              idev->EnablePeerAccess(jdev->hipdev());
          }
      }
      */
  }
  delete [] hipdevs;
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "HIP");
    first_dev_of_type_[nplatforms_] = devs_[ndevs_-mdevs];
    nplatforms_++;
  }
  return IRIS_SUCCESS;
}

int Platform::InitLevelZero() {
  //if (arch_available_ & iris_gpu_intel) {
  //  _trace("%s", "skipping LevelZero architecture");
  //  return IRIS_ERROR;
  //}
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

  int mdevs = 0;
  for (uint32_t i = 0; i < ndevs; i++) {
    if (ndevs_ > IRIS_MAX_NDEVS) {
        _error("This platform has more than max devices: %d ! Hence ignoring further LevelZero devices\n", IRIS_MAX_NDEVS);
        break;
    }
    devs_[ndevs_] = new DeviceLevelZero(loaderLevelZero_, devs[i], zectx, driver, ndevs_, nplatforms_);
    arch_available_ |= devs_[ndevs_]->type();
    ndevs_++; mdevs++;
  }
  if (ndevs) {
    strcpy(platform_names_[nplatforms_], "LevelZero");
    first_dev_of_type_[nplatforms_] = devs_[ndevs_-mdevs];
    nplatforms_++;
  }
  return IRIS_SUCCESS;
}

int Platform::InitOpenMP() {
  //if (arch_available_ & iris_cpu) {
  //  _trace("%s", "skipping OpenMP architecture");
  //  return IRIS_ERROR;
  //}
  loaderOpenMP_ = new LoaderOpenMP();
  if (loaderOpenMP_->Load() != IRIS_SUCCESS) {
      char *filename = (char *)malloc(512);
      EnvironmentGet("KERNEL_BIN_OPENMP", &filename, NULL);
      _warning("couldn't find OpenMP architecture kernel library:%s", filename);
      free(filename);
  }
  int mdevs = 0;
  for(int i=0; i<openmp_device_factor_; i++) {
      if (ndevs_ > IRIS_MAX_NDEVS) {
          _error("This platform has more than max devices: %d ! Hence ignoring further OpenMP devices\n", IRIS_MAX_NDEVS);
          break;
      }
      _trace("OpenMP platform[%d] dev[%d] ndevs[%d]", nplatforms_, ndevs_, ndevs_+1);
      _printf("OpenMP platform[%d] dev[%d] ndevs[%d]", nplatforms_, ndevs_, ndevs_+1);
      devs_[ndevs_] = new DeviceOpenMP(loaderOpenMP_, ndevs_, nplatforms_);
      if (is_julia_enabled()) 
          devs_[ndevs_]->EnableJuliaInterface();
      arch_available_ |= devs_[ndevs_]->type();
      ndevs_++;
      mdevs++;
  }
  strcpy(platform_names_[nplatforms_], "OpenMP");
  first_dev_of_type_[nplatforms_] = devs_[ndevs_-mdevs];
  nplatforms_++;
  return IRIS_SUCCESS;
}

int Platform::InitQIREE() {
  loaderQIREE_ = new LoaderQIREE();
  if (loaderQIREE_->Load() != IRIS_SUCCESS) {
      char *filename = (char *)malloc(512);
      EnvironmentGet("KERNEL_QIR", &filename, NULL);
      _warning("couldn't find QIR architecture kernel library:%s", filename);
      free(filename);
  }
  int mdevs = 0;
  for(int i=0; i<qiree_device_factor_; i++) {
      _trace("QIR platform[%d] dev[%d] ndevs[%d]", nplatforms_, ndevs_, ndevs_+1);
      devs_[ndevs_] = new DeviceQIREE(loaderQIREE_, ndevs_, nplatforms_);
      if (is_julia_enabled()) 
          devs_[ndevs_]->EnableJuliaInterface();
      arch_available_ |= devs_[ndevs_]->type();
      ndevs_++;
      mdevs++;
  }
  strcpy(platform_names_[nplatforms_], "QIR");
  first_dev_of_type_[nplatforms_] = devs_[ndevs_-mdevs];
  nplatforms_++;
  return IRIS_SUCCESS;
}

int Platform::InitHexagon() {
  //if (arch_available_ & iris_hexagon) {
  //  _trace("%s", "skipping Hexagon architecture");
  //  return IRIS_ERROR;
  //}
  loaderHexagon_ = new LoaderHexagon();
  if (loaderHexagon_->Load() != IRIS_SUCCESS) {
    char *filename = (char *)malloc(512);
    EnvironmentGet("KERNEL_BIN_HEXAGON", &filename, NULL);
    _trace("couldn't find Hexagon architecture kernel library:%s, hence skipping", filename);
    free(filename);
    return IRIS_ERROR;
  }
  _trace("Hexagon platform[%d] ndevs[%d]", nplatforms_, 1);
  devs_[ndevs_] = new DeviceHexagon(loaderHexagon_, ndevs_, nplatforms_);
  arch_available_ |= devs_[ndevs_]->type();
  ndevs_++;
  strcpy(platform_names_[nplatforms_], "Hexagon");
  first_dev_of_type_[nplatforms_] = devs_[ndevs_-1];
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
  char *ocl_vendors_str = NULL;
  EnvironmentGet("OPENCL_VENDORS", &ocl_vendors_str, NULL);
  std::map<std::string, int> ocl_vendors_map;
  std::map<int, bool> ocl_vendors;
  ocl_vendors_map["all"] = CL_DEVICE_TYPE_ALL;
  ocl_vendors_map["fpga"] = CL_DEVICE_TYPE_ACCELERATOR;;
  ocl_vendors_map["gpu"] = CL_DEVICE_TYPE_GPU;;
  ocl_vendors_map["cpu"] = CL_DEVICE_TYPE_CPU;;
  ocl_vendors[CL_DEVICE_TYPE_CPU] = false;
  ocl_vendors[CL_DEVICE_TYPE_GPU] = false;
  ocl_vendors[CL_DEVICE_TYPE_ACCELERATOR] = false;
  ocl_vendors[CL_DEVICE_TYPE_ALL] = false;
  Utils::ReadMap(ocl_vendors_map, ocl_vendors, ocl_vendors_str);
  int max_ndevs = -1;
  EnvironmentIntRead("MAX_OPENCL_DEVICE", max_ndevs);
  char vendor[64];
  char platform_name[64];
  int ocldevno = 0;
  bool is_all = ocl_vendors[CL_DEVICE_TYPE_ALL];
  for (cl_uint i = 0; i < nplatforms; i++) {
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    _clerror(err);
    err = loaderOpenCL_->clGetPlatformInfo(cl_platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    _clerror(err);
    _trace("OpenCL platform[%s] from [%s]", platform_name, vendor);

    //if ((arch_available_ & iris_nvidia) && strstr(vendor, "NVIDIA") != NULL) {
    //  _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
    //  continue;
    //}
    //if ((arch_available_ & iris_amd) && strstr(vendor, "Advanced Micro Devices") != NULL) {
    //  _trace("skipping platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
    //  continue;
    //}
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
    int mdevs = 0;
    for (cl_uint j = 0; j < ndevs; j++) {
      if (ndevs_ > IRIS_MAX_NDEVS) {
          _error("This platform has more than max devices: %d ! Hence ignoring further OpenCL devices\n", IRIS_MAX_NDEVS);
          break;
      }
      cl_device_type dev_type;
      err = loaderOpenCL_->clGetDeviceInfo(cl_devices[j], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
      _clerror(err);
      if ((arch_available_ & iris_cpu) && (dev_type == CL_DEVICE_TYPE_CPU)) continue;
      if (!is_all && !ocl_vendors[dev_type]) continue;
      std::string suffix = DeviceOpenCL::GetLoaderHost2OpenCLSuffix(loaderOpenCL_, cl_devices[j]);
      LoaderHost2OpenCL *loaderHost2OpenCL = new LoaderHost2OpenCL(suffix.c_str());
      if (loaderHost2OpenCL->Load() != IRIS_SUCCESS) {
        _trace("%s", "skipping Host2OpenCL wrapper calls");
      }
      loaderHost2OpenCL_.push_back(loaderHost2OpenCL);
      DeviceOpenCL *dev = new DeviceOpenCL(loaderOpenCL_, loaderHost2OpenCL, cl_devices[j], cl_contexts[i], ndevs_, ocldevno, nplatforms_);
      if (!dev->IsDeviceValid()) {
          delete dev;
          continue;
      }
      devs_[ndevs_] = dev;
      devs_[ndevs_]->set_root_device(devs_[ndevs_-mdevs]);
      arch_available_ |= devs_[ndevs_]->type();
      ndevs_++;
      mdevs++;
      ocldevno++;
      if (max_ndevs != -1 && ocldevno >= max_ndevs) break;
#ifdef ENABLE_SINGLE_DEVICE_PER_CU
      break;
#endif
    }
    if (mdevs > 0) {
        _trace("adding platform[%d] [%s %s] ndevs[%u]", nplatforms_, vendor, platform_name, ndevs);
        sprintf(platform_names_[nplatforms_], "OpenCL %s", vendor);
        first_dev_of_type_[nplatforms_] = devs_[ndevs_-mdevs];
        nplatforms_++;
#ifdef ENABLE_SINGLE_DEVICE_PER_CU
        if (ocldevno > 0) break;
#endif
    }
    if (max_ndevs != -1 && ocldevno >= max_ndevs) break;
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

  //Task** tasks = new Task*[ndevs_];
  iris_task *tasks = new iris_task[ndevs_];
  char task_name[128];
  for (int i = 0; i < ndevs_; i++) {
    sprintf(task_name, "Initialize-%d", i);
    //tasks[i] = new Task(this, IRIS_TASK, task_name);
    TaskCreate(task_name, false, &tasks[i]);
    Task *task = get_task_object(tasks[i]);
    task->set_system();
    task->Retain();
    Command* cmd = Command::CreateInit(task);
    task->AddCommand(cmd);
    _debug2("Initialize task:%lu:%s ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
    task->Retain();
    task->Retain();
    workers_[i]->Enqueue(task);
  }
  if (sync) for (int i = 0; i < ndevs_; i++) {
    Task *task = get_task_object(tasks[i]);
    task->Wait();
    task->Release();
    //Task *task = get_task_object(tasks[i]);
    //TODO: add clause to send `call workers_[1]->Enqueue(task)` if task has been missed? 1 in 20k run deadlock
    //if(task != NULL && task->status() != IRIS_COMPLETE) {task->Retain(); task->Wait(); task->Release();}
    //TaskWait(tasks[i]);
  }
  //for (int i = 0; i < ndevs_; i++) {
    //Task *task = get_task_object(tasks[i]);
    //_debug2("Release Initialize task:%lu:%s ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
    //task->Release();
  //}
  delete[] tasks;
  Synchronize();
  for(int i=0; i<ndevs_; i++) {
    devs_[i]->EnablePeerAccess();
  }
  return IRIS_SUCCESS;
}

void Platform::ShowOverview() {
  int nplatforms = 0;
  int ndevices = 0;
  PlatformCount(&nplatforms);
  DeviceCount(&ndevices);
  std::cout << "IRIS is using " << ndevices << " devices with " << nplatforms << " platforms:" << std::endl;
  char name[256];
  char vendor[256];
  const char* backend="Unknown";
  const char* type="Unknown";
  //print table header
  std::cout << setw(8) << setfill('-') << "--" << "-+-" <<
    setw(32) << setfill('-') << "----" << "-+-" <<
    setw(32) << setfill('-') << "-----" << "-+-" <<
    setw(32) << setfill('-') << "------" <<  "-+-" <<
    setw(12) << setfill('-') << "----" << "-" << std::endl;
  std::cout << setw(8) << setfill(' ') << "id" << " | " <<
    setw(32) << setfill(' ') << "name" << " | " <<
    setw(32) << setfill(' ') << "vendor" << " | " <<
    setw(32) << setfill(' ') << "backend" <<  " | " <<
    setw(12) << setfill(' ') << "type" << " " << std::endl;
  std::cout << setw(8) << setfill('-') << "--" << "-+-" <<
    setw(32) << setfill('-') << "----" << "-+-" <<
    setw(32) << setfill('-') << "-----" << "-+-" <<
    setw(32) << setfill('-') << "------" <<  "-+-" <<
    setw(12) << setfill('-') << "----" << "-" << std::endl;
  for (int i = 0; i < ndevices; i++){
    int type_id;
    int backend_id;
    DeviceInfo(i, iris_name,    name,       NULL);
    DeviceInfo(i, iris_vendor,  vendor,     NULL);
    DeviceInfo(i, iris_type,    &type_id,   NULL);
    DeviceInfo(i, iris_backend, &backend_id,NULL);
    switch(backend_id){
      case (iris_cuda):     backend = "cuda";       break;
      case (iris_hexagon):  backend = "hexagon";    break;
      case (iris_hip):      backend = "hip";        break;
      case (iris_levelzero):backend = "levelzero";  break;
      case (iris_opencl):   backend = "opencl";     break;
      case (iris_openmp):   backend = "cpu";        break;
    }
    switch(type_id){
      case (iris_cpu):      type = "CPU";         break;
      case (iris_gpu):      type = "GPU";         break;
      case (iris_amd) :     type = "AMD GPU";     break;
      case (iris_nvidia) :  type = "NVIDIA GPU";  break;
      case (iris_phi):      type = "PHI";         break;
      case (iris_fpga):     type = "FPGA";        break;
      case (iris_dsp):      type = "DSP";         break;
    }
    std::cout << setw(8) << setfill(' ') << i << " | " <<
      setw(32) << setfill(' ') << name << " | "<<
      setw(32) << setfill(' ') << vendor  << " | " <<
      setw(32) << setfill(' ') << backend <<  " | " <<
      setw(12) << setfill(' ') << type << " " << std::endl;
  } 
  std::cout << setw(8) << setfill('-') << "--" << "-+-" <<
    setw(32) << setfill('-') << "----" << "-+-" <<
    setw(32) << setfill('-') << "-----" << "-+-" <<
    setw(32) << setfill('-') << "------" <<  "-+-" <<
    setw(12) << setfill('-') << "----" << "-" << std::endl;
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
    case iris_backend   : if (size) *size = sizeof(int);            *((int*) value) = dev->model();         break;
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
  task->Retain();
  task->Retain();
  //iris_task brs_task = *(task->struct_obj());
  if (scheduler_) {
    char sync_task[128];
    for (int i = 0; i < ndevs; i++) {
      if (devices[i] >= ndevs_) {
        _error("devices[%d]", devices[i]);
        continue;
      }
      sprintf(sync_task, "Marker-%d", i);
      Task* subtask = new Task(this, IRIS_MARKER, sync_task);
      subtask->set_devno(devices[i]);
      subtask->Retain();
      subtask->set_user(true);
      _debug2("Created marker task:%lu:%s ref_cnt:%d", subtask->uid(), subtask->name(), subtask->ref_cnt());
      task->AddSubtask(subtask);
    }
    scheduler_->Enqueue(task);
  } else{
    workers_[0]->Enqueue(task);
  }
  task->Wait();
  //printf("Task uid:%lu ref_cnt:%u\n", task->uid(), task->ref_cnt());
  task->Release();
  //TaskWait(brs_task);
  // Task::Ok returns only Device::Ok. However, the parent task doesn't map to any 
  // device. It is meaningless to call task->Ok(). Hence, returning  IRIS_SUCCESS.
  //printf("1Task uid:%lu ref_cnt:%u\n", task->uid(), task->ref_cnt());
  task->Release();
  return IRIS_SUCCESS;
}

int Platform::PolicyRegister(const char* lib, const char* name, void* params) {
  return scheduler_->policies()->Register(lib, name, params);
}

int Platform::CalibrateCommunicationMatrix(double *comm_time, size_t data_size, int iterations, bool pin_memory_flag)
{
    uint8_t *A = iris::AllocateRandomArray<uint8_t>(data_size);
    uint8_t *nopin_A = iris::AllocateRandomArray<uint8_t>(data_size);
    if (pin_memory_flag) 
        iris_register_pin_memory(A, data_size);
    Mem* mem = new Mem(data_size, this);
    size_t gws=1;
    Task* task = Task::Create(this, IRIS_TASK, NULL);
    Command* d2d_cmd = Command::CreateD2D(task, mem, 0, data_size, A, 0);
    Command* h2d_cmd = Command::CreateH2D(task, mem, 0, data_size, A);
    Command* d2h_cmd = Command::CreateD2H(task, mem, 0, data_size, A);
    Command* nopin_h2d_cmd = Command::CreateH2D(task, mem, 0, data_size, nopin_A);
    Command* nopin_d2h_cmd = Command::CreateD2H(task, mem, 0, data_size, nopin_A);
    iris_kernel null_brs_kernel;
    KernelCreate("iris_null", &null_brs_kernel);
    Kernel *null_kernel = get_kernel_object(null_brs_kernel);;
    Command* cmd_kernel = Command::CreateKernel(task, null_kernel, 1, 0, &gws, &gws);
    task->AddCommand(cmd_kernel);
    int ndevs = ndevs_+1;
    for(int i=0; i<ndevs; i++) {
        for(int j=0; j<ndevs; j++) {
            double cmd_time = 0.0f;
            if (i != j) {
                double total_cmd_time = 0.0f;
                for(int k=0; k<iterations; k++) {
                    double lcmd_time = 0.0f;
                    if (j == 0) {
                        devs_[i-1]->ExecuteD2H(d2h_cmd);
                        lcmd_time = d2h_cmd->time_end() - d2h_cmd->time_start();
                    } else if (i == 0) {
                        devs_[j-1]->ExecuteH2D(h2d_cmd);
                        lcmd_time = h2d_cmd->time_end() - h2d_cmd->time_start();
                    } else if (j>0 && devs_[j-1]->type() == iris_cpu) {
                        devs_[i-1]->ExecuteD2H(nopin_d2h_cmd);
                        lcmd_time = nopin_d2h_cmd->time_end() - nopin_d2h_cmd->time_start();
                    } else if (i>0 && devs_[i-1]->type() == iris_cpu) {
                        devs_[j-1]->ExecuteH2D(nopin_h2d_cmd);
                        lcmd_time = nopin_h2d_cmd->time_end() - nopin_h2d_cmd->time_start();
                    } else {
                        d2d_cmd->set_src_dev(i-1);
                        devs_[j-1]->ExecuteD2D(d2d_cmd);
                        lcmd_time = d2d_cmd->time_end() - d2d_cmd->time_start();
                    }
                    total_cmd_time += lcmd_time;
                }
                cmd_time = total_cmd_time / iterations;
            } 
            comm_time[i*ndevs + j] = cmd_time;
        }
    }
    //delete cmd_kernel;
    delete d2d_cmd;
    delete h2d_cmd;
    delete d2h_cmd;
    delete mem;
    delete task;
    return IRIS_SUCCESS;
}

int Platform::RegisterCommand(int tag, int device, command_handler handler) {
  for (int i = 0; i < ndevs_; i++)
    if (devs_[i]->model() == device) devs_[i]->RegisterCommand(tag, handler);
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
  if (brs_kernel) kernel->SetStructObject(brs_kernel);
  //if (brs_kernel) *brs_kernel = kernel->struct_obj();
  std::string name_string = name;
  if (kernels_.find(name_string) != kernels_.end()) {
      std::vector<Kernel *> vec;
      kernels_.insert(std::pair<std::string, std::vector<Kernel *> >(name_string, vec));
  }
  kernels_[name_string].push_back(kernel);
  return IRIS_SUCCESS;
}

int Platform::KernelGet(const char* name, iris_kernel* brs_kernel) {
  // We create a unique kernel each time
  return KernelCreate(name, brs_kernel);
}

int Platform::KernelSetArg(iris_kernel brs_kernel, int idx, size_t size, void* value) {
  Kernel* kernel = Platform::GetPlatform()->get_kernel_object(brs_kernel);
  kernel->SetArg(idx, size, value);
  return IRIS_SUCCESS;
}

int Platform::KernelSetMem(iris_kernel brs_kernel, int idx, iris_mem brs_mem, size_t off, size_t mode) {
  Kernel* kernel = Platform::GetPlatform()->get_kernel_object(brs_kernel);
  BaseMem* mem = Platform::GetPlatform()->get_mem_object(brs_mem);
  kernel->SetMem(idx, mem, off, mode);
  return IRIS_SUCCESS;
}

int Platform::KernelSetMap(iris_kernel brs_kernel, int idx, void* host, size_t mode) {
  Kernel* kernel = Platform::GetPlatform()->get_kernel_object(brs_kernel);
  size_t off = 0ULL;
  Mem* mem = (Mem *)present_table_->Get(host, &off);
  if (mem) kernel->SetMem(idx, mem, off, mode);
  else {
    _todo("clearing [%p]", host);
    MemMap(host, 8192);
    Mem* mem = (Mem *)present_table_->Get(host, &off);
    kernel->SetMem(idx, mem, off, mode);
  }
  return IRIS_SUCCESS;
}

int Platform::KernelRelease(iris_kernel brs_kernel) {
  Kernel* kernel = Platform::GetPlatform()->get_kernel_object(brs_kernel);
  kernel->Release();
  return IRIS_SUCCESS;
}

int Platform::TaskCreate(const char* name, bool perm, iris_task* brs_task) {
  Task* task = Task::Create(this, perm ? IRIS_TASK_PERM : IRIS_TASK, name);
  if (perm) task->DisableRelease();
  if (brs_task) task->SetStructObject(brs_task);
  return IRIS_SUCCESS;
}

int Platform::TaskDepend(iris_task brs_task, int ntasks, iris_task** brs_tasks) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  for (int i = 0; i < ntasks; i++) 
      if (brs_tasks[i] != NULL) 
          task->AddDepend(get_task_object(brs_tasks[i]->uid), brs_tasks[i]->uid);
  return IRIS_SUCCESS;
}

int Platform::TaskDepend(iris_task brs_task, int ntasks, iris_task* brs_tasks) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  for (int i = 0; i < ntasks; i++) 
      task->AddDepend(get_task_object(brs_tasks[i].uid), brs_tasks[i].uid);
  return IRIS_SUCCESS;
}

int Platform::TaskKernel(iris_task brs_task, iris_kernel brs_kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Kernel* kernel = Platform::GetPlatform()->get_kernel_object(brs_kernel);
  kernel->set_task_name(task->name());
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, gws, lws);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskCustom(iris_task brs_task, int tag, void* params, size_t params_size) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Command* cmd = Command::CreateCustom(task, tag, params, params_size);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskKernel(iris_task brs_task, const char* name, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Kernel* kernel = GetKernel(name);
  //_trace("Adding kernel:%s:%p to task:%s\n", name, kernel, task->name());
  if (!task->given_name()) task->set_name(name);
  kernel->set_task_name(task->name());
  Command* cmd = Command::CreateKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, memranges);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::SetParamsMap(iris_task brs_task, int *params_map)
{
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Command *cmd_kernel = task->cmd_kernel();
  cmd_kernel->set_params_map(params_map);
  return IRIS_SUCCESS;
}

int Platform::SetSharedMemoryModel(iris_mem brs_mem, DeviceModel model, int flag)
{
    BaseMem* mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    for (int i = 0; i < ndevs_; i++) {
        if (devs_[i] && (model == iris_model_all || devs_[i]->model() == model)) {
            devs_[i]->set_can_share_host_memory_flag((bool)flag);
            devs_[i]->set_shared_memory_buffers((bool)flag);
            mem->set_usm_flag(i, flag);
        }
    }
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
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Command* cmd = task->cmd_kernel();
  if (!cmd) return IRIS_ERROR;
  cmd->set_selector_kernel(func, params, params_size);
  return IRIS_SUCCESS;
}

int Platform::TaskHost(iris_task brs_task, iris_host_python_task func, int64_t params_id) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Command* cmd = Command::CreateHost(task, func, params_id);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskHost(iris_task brs_task, iris_host_task func, void* params) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Command* cmd = Command::CreateHost(task, func, params);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMalloc(iris_task brs_task, iris_mem brs_mem) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem*)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateMalloc(task, mem);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMemFlushOut(iris_task brs_task, iris_mem brs_mem) {
  Task *task = get_task_object(brs_task);
  bool submit = false;
  if (task == NULL) {
    // It is possible that the task submitted earlier is completed.
    // Lets handle this scenario
    brs_task = iris_task_create_struct();
    task = get_task_object(brs_task);
    submit = true;
  }
  assert(task != NULL);
  DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateMemFlushOut(task, mem);
  task->AddCommand(cmd);
  // Submit the task and wait for flush completion
  if (submit) TaskSubmit(brs_task, iris_default, NULL, 1);
  return IRIS_SUCCESS;
}

int Platform::TaskMemResetInput(iris_task brs_task, iris_mem brs_mem, uint8_t reset) 
{
    Task *task = get_task_object(brs_task);
    assert(task != NULL);
    BaseMem* mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    Command *cmd = Command::CreateMemResetInput(task, mem, reset);
    task->AddCommand(cmd);
    return IRIS_SUCCESS;
}

int Platform::TaskMemResetInput(iris_task brs_task, iris_mem brs_mem, ResetData & reset) 
{
    Task *task = get_task_object(brs_task);
    assert(task != NULL);
    BaseMem* mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    Command *cmd = Command::CreateMemResetInput(task, mem, reset);
    task->AddCommand(cmd);
    return IRIS_SUCCESS;
}

int Platform::TaskH2Broadcast(iris_task brs_task, iris_mem brs_mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateH2Broadcast(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2Broadcast(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateH2Broadcast(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskDMEM2DMEM(iris_task brs_task, iris_mem src_mem, iris_mem dst_mem) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  BaseMem* src_mem_iris = (BaseMem *)Platform::GetPlatform()->get_mem_object(src_mem);
  BaseMem* dst_mem_iris = (BaseMem *)Platform::GetPlatform()->get_mem_object(dst_mem);
  ASSERT(src_mem_iris->GetMemHandlerType() == IRIS_DMEM);
  ASSERT(dst_mem_iris->GetMemHandlerType() == IRIS_DMEM);
  Command* cmd = Command::CreateDMEM2DMEM(task, (DataMem *)src_mem_iris, (DataMem *)dst_mem_iris);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2D(iris_task brs_task, iris_mem brs_mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateH2D(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskD2D(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host, int src_dev) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateD2D(task, mem, off, size, host, src_dev);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2D(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskD2H(iris_task brs_task, iris_mem brs_mem, size_t off, size_t size, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskD2H(iris_task brs_task, iris_mem brs_mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateD2H(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskH2BroadcastFull(iris_task brs_task, iris_mem brs_mem, void* host) {
  return TaskH2Broadcast(brs_task, brs_mem, 0ULL, Platform::GetPlatform()->get_mem_object(brs_mem)->size(), host);
}

int Platform::TaskH2DFull(iris_task brs_task, iris_mem brs_mem, void* host) {
  return TaskH2D(brs_task, brs_mem, 0ULL, Platform::GetPlatform()->get_mem_object(brs_mem)->size(), host);
}

int Platform::TaskD2HFull(iris_task brs_task, iris_mem brs_mem, void* host) {
  return TaskD2H(brs_task, brs_mem, 0ULL, Platform::GetPlatform()->get_mem_object(brs_mem)->size(), host);
}

int Platform::TaskMap(iris_task brs_task, void* host, size_t size) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Command* cmd = Command::CreateMap(task, host, size);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapTo(iris_task brs_task, void* host, size_t size) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  size_t off = 0ULL;
  Mem* mem = (Mem *)present_table_->Get(host, &off);
  Command* cmd = Command::CreateH2D(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapToFull(iris_task brs_task, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  size_t off = 0ULL;
  Mem* mem = (Mem *)present_table_->Get(host, &off);
  size_t size = mem->size();
  Command* cmd = Command::CreateH2D(task, mem, off, size - off, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapFrom(iris_task brs_task, void* host, size_t size) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  size_t off = 0ULL;
  Mem* mem = (Mem *)present_table_->Get(host, &off);
  Command* cmd = Command::CreateD2H(task, mem, off, size, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskMapFromFull(iris_task brs_task, void* host) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  size_t off = 0ULL;
  Mem* mem = (Mem *)present_table_->Get(host, &off);
  size_t size = mem->size();
  Command* cmd = Command::CreateD2H(task, mem, off, size - off, host);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

shared_ptr<History> Platform::CreateHistory(string kname)
{
    if (kernel_history_.find(kname) == kernel_history_.end()) {
        std::vector<shared_ptr<History> > vec;
        kernel_history_.insert(std::pair<std::string, std::vector<shared_ptr<History>> >(kname, vec));
    }
    shared_ptr<History> history = make_shared<History>(this);
    kernel_history_[kname].push_back(history);
    return history;
}
void Platform::ProfileCompletedTask(Task *task)
{
    if (!task->marker()){
        for (int i = 0; i < nprofilers_; i++) profilers_[i]->CompleteTask(task);
    }
}

void Platform::IncrementErrorCount(){
  nfailures_++;
}

int Platform::NumErrors(){
  return nfailures_;
}

int Platform::TaskSubmit(iris_task brs_task, int brs_policy, const char* opt, int sync) {
  if (ndevs_ == 0) {
    _error("Cannot submit task:%lu due to no devices found on this system!", brs_task.uid);
    IncrementErrorCount();
    return IRIS_ERROR;
  }
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  if (recording_) json_->RecordTask(task);
  task->Retain();
  task->set_time_submit(timer_);
  task->Submit(brs_policy, opt, sync);
  _trace(" successfully submitted task:%lu:%s", task->uid(), task->name());
  if (scheduler_) {
    FilterSubmitExecute(task);
    scheduler_->Enqueue(task);
  } else { task->Retain(); workers_[0]->Enqueue(task); }
  _debug2("Task wait release:%lu:%s ref_cnt:%d before TaskSubmit\n", task->uid(), task->name(), task->ref_cnt());
  if (sync) TaskWait(brs_task);
  return nfailures_;
}

int Platform::TaskSubmit(Task *task, int brs_policy, const char* opt, int sync) {
  if (ndevs_ == 0) {
    _error("Cannot submit task:%lu due to no devices found on this system!", task->uid());
    IncrementErrorCount();
    return IRIS_ERROR;
  }
  iris_task brs_task = *(task->struct_obj());
  if (recording_) json_->RecordTask(task);
  task->Retain();
  task->set_time_submit(timer_);
  task->Submit(brs_policy, opt, sync);
  _trace(" successfully submitted task:%lu:%s", task->uid(), task->name());
  if (scheduler_) {
    FilterSubmitExecute(task);
    scheduler_->Enqueue(task);
  } else { task->Retain(); workers_[0]->Enqueue(task); }
  if (sync) TaskWait(brs_task);
  return nfailures_;
}
void Platform::TaskSafeRetainStatic(void *data) {
  Task *task = (Task *)data;
  task->Retain();
}
void Platform::TaskSafeRetain(unsigned long uid) {
  task_track_.CallBackIfObjectExists(uid, Platform::TaskSafeRetainStatic);
}
void Platform::TaskSafeRetain(iris_task brs_task) {
  task_track_.CallBackIfObjectExists(brs_task.uid, Platform::TaskSafeRetainStatic);
}
int Platform::TaskWait(iris_task brs_task) {
  _debug2("waiting for brs_task:%lu\n", brs_task.uid);
  TaskSafeRetain(brs_task);
  Task *task = get_task_object(brs_task);
  if (task != NULL) {
    unsigned long uid = task->uid(); string lname = task->name(); 
    _debug2("Task wait release:%lu:%s ref_cnt:%d after callback\n", task->uid(), task->name(), task->ref_cnt());
    task->Wait();
    _debug2("Task wait before release:%lu:%s ref_cnt:%d\n", uid, lname.c_str(), task->ref_cnt());
    int ref_cnt = task->Release();
    _debug2("Task wait after release:%lu:%s ref_cnt:%d\n", uid, lname.c_str(), ref_cnt);
  }
  return IRIS_SUCCESS;
}

int Platform::TaskWaitAll(int ntasks, iris_task* brs_tasks) {
  int iret = IRIS_SUCCESS;
  for (int i = 0; i < ntasks; i++) iret &= TaskWait(brs_tasks[i]);
  return iret;
}

int Platform::TaskAddSubtask(iris_task brs_task, iris_task brs_subtask) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Task *subtask = get_task_object(brs_subtask);
  task->AddSubtask(subtask);
  return IRIS_SUCCESS;
}

int Platform::TaskKernelCmdOnly(iris_task brs_task) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  return (task->ncmds() == 1 && task->cmd_kernel()) ? IRIS_SUCCESS : IRIS_ERROR;
}

void Platform::TaskReleaseStatic(void *data)
{
    Task *task = (Task *)data;
    // This is only for tasks which has to be manuallay released. 
    // Otherwise, runtime will take care of when to release it
    if (!task->IsRelease()) {
        _debug2("Force releasing the task id:%lu:%s", task->uid(), task->name());
        assert(task->ref_cnt() == 1);
        task->EnableRelease();
        task->ForceRelease();
    }
}

int Platform::TaskRelease(iris_task brs_task) {
    _debug2("Releasing the task id:%lu", brs_task.uid);
    //task_track_.CallBackIfObjectExists(brs_task.uid, Platform::TaskReleaseStatic);
    //Retainable object should alive 
    Task *task = get_task_object(brs_task);
    if (task != NULL && !task->IsRelease()) {
        task->EnableRelease();
        _debug2("Releasing the task id:%lu ref_cnt:%d\n", brs_task.uid, task->ref_cnt());
        //assert(task->ref_cnt() == 1);
        task->Release();
        //Task should be alive yet.
    }
    return IRIS_SUCCESS;
}

int Platform::TaskReleaseMem(iris_task brs_task, iris_mem brs_mem) {
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Command* cmd = Command::CreateReleaseMem(task, mem);
  task->AddCommand(cmd);
  return IRIS_SUCCESS;
}

int Platform::TaskInfo(iris_task brs_task, int param, void* value, size_t* size) {
  Task *task = get_task_object(brs_task);
  if (task == NULL) {
    _error("Task:%lu is not alive!", brs_task.uid);
    return IRIS_ERROR;
  }
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
  } else if (param == iris_task_time_submit) {
    if (size) *size = sizeof(size_t);
    *((size_t*) value) = task->ns_time_submit();
  } else if (param == iris_task_time_start) {
    if (size) *size = sizeof(double);
    size_t first_kernel_time = SIZE_MAX;
    //get the earliest command kernels time
    for (int i = 0 ; i < task->ncmds(); i++){
      if (task->cmd(i)->type_kernel()){
        size_t this_time = task->cmd(i)->ns_time_start();
        if (this_time < first_kernel_time) first_kernel_time = this_time;
      }
    }
    //For any given event, this timestamp is always greater than or equal to the iris_task_time_submit timestamp
    if (first_kernel_time == SIZE_MAX) first_kernel_time = task->ns_time_submit();
    *((size_t*) value) = first_kernel_time;
  } else if (param == iris_task_time_end) {
    if (size) *size = sizeof(double);
    //get the latest command kernels time
    size_t last_kernel_time = 0;
    //get the earliest command kernels time
    for (int i = 0 ; i < task->ncmds(); i++){
      if (task->cmd(i)->type_kernel()){
        size_t this_time = task->cmd(i)->ns_time_end();
        if (this_time > last_kernel_time) last_kernel_time = this_time;
      }
    }
    //For any given event, this timestamp is always greater than or equal to the iris_task_time_submit timestamp
    if (last_kernel_time == 0) last_kernel_time = task->ns_time_submit();
    *((size_t*) value) = last_kernel_time;
  }
  return IRIS_SUCCESS;
}

int Platform::DataMemInit(iris_mem brs_mem, bool reset) {
    BaseMem *mem = Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem->GetMemHandlerType() != IRIS_DMEM) {
        _error("IRIS Mem is not supported for initialization with reset value. %ld", mem->uid());
        return IRIS_ERROR;
    }
    mem->init_reset(reset);
    return IRIS_SUCCESS;
}

int Platform::DataMemInit(BaseMem *mem, bool reset) {
    if (mem->GetMemHandlerType() != IRIS_DMEM) {
        _error("IRIS Mem is not supported for initialization with reset value. %ld", mem->uid());
        return IRIS_ERROR;
    }
    mem->init_reset(reset);
    return IRIS_SUCCESS;
}

int Platform::DataMemUpdate(iris_mem brs_mem, void *host) {
  DataMem *mem = (DataMem *) Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->UpdateHost(host);
  return IRIS_SUCCESS;
}

int Platform::UnRegisterPin(void *host) {
#if 0
  for (int i=0; i<ndevs_; i++) 
      devs_[i]->UnRegisterPin(host, size);
#else
  for (int i=0; i<nplatforms_; i++) 
      first_dev_of_type_[i]->UnRegisterPin(host);
#endif
  return IRIS_SUCCESS;
}

int Platform::DataMemUnRegisterPin(DataMem *mem) {
    int status = IRIS_SUCCESS;
    if (mem->is_pin_memory()) {
        void *host = mem->host_ptr();
        if (host == NULL) return IRIS_SUCCESS;
        size_t size =mem->size();
        _trace("UnRegistering PIN for %p size:%lu end_addr:%p", host, size, (char*)host+size);
        status = UnRegisterPin(host);
        mem->set_pin_memory(false);
    }
  return status;
}

int Platform::DataMemUnRegisterPin(iris_mem brs_mem) {
  DataMem *mem = (DataMem *) Platform::GetPlatform()->get_mem_object(brs_mem);
  return DataMemUnRegisterPin(mem);
}


int Platform::RegisterPin(void *host, size_t size) {
#if 0
  for (int i=0; i<ndevs_; i++) 
      devs_[i]->RegisterPin(host, size);
#else
  for (int i=0; i<nplatforms_; i++) 
      first_dev_of_type_[i]->RegisterPin(host, size);
#endif
  return IRIS_SUCCESS;
}

int Platform::DataMemRegisterPin(DataMem *mem) {
  void *host = mem->host_ptr();
  if (host == NULL) return IRIS_SUCCESS;
  size_t size =mem->size();
  mem->set_pin_memory();
  _trace("Registering PIN for %p size:%lu end_addr:%p", host, size, (char*)host+size);
  return RegisterPin(host, size);
}

int Platform::DataMemRegisterPin(iris_mem brs_mem) {
  DataMem *mem = (DataMem *) Platform::GetPlatform()->get_mem_object(brs_mem);
  return DataMemRegisterPin(mem);
}

int Platform::DataMemCreate(iris_mem* brs_mem, void *host, size_t size, int element_type) {
  DataMem* mem = new DataMem(this, host, size, element_type);
  if (brs_mem) mem->SetStructObject(brs_mem);
#ifndef DISABLE_PIN_BY_DEFAULT
  if (dmem_register_pin_flag_)
      DataMemRegisterPin(*brs_mem);
#endif
  //if (brs_mem) *brs_mem = mem->struct_obj();
  if (mem->size()==0) return IRIS_ERROR;

  //mems_.insert(mem);
  return IRIS_SUCCESS;
}

int Platform::DataMemCreate(iris_mem* brs_mem, void *host, size_t size, const char *symbol, int element_type) {
  DataMem* mem = new DataMem(this, host, size, symbol, element_type);
  if (brs_mem) mem->SetStructObject(brs_mem);
#ifndef DISABLE_PIN_BY_DEFAULT
  if (dmem_register_pin_flag_)
    DataMemRegisterPin(*brs_mem);
#endif
  //if (brs_mem) *brs_mem = mem->struct_obj();
  if (mem->size()==0) return IRIS_ERROR;

  //mems_.insert(mem);
  return IRIS_SUCCESS;
}

int Platform::DataMemCreate(iris_mem* brs_mem, void *host, size_t *size, int dim, size_t element_size,  int element_type) {
  DataMem* mem = new DataMem(this, host, size, dim, element_size, element_type);
  if (brs_mem) mem->SetStructObject(brs_mem);
#ifndef DISABLE_PIN_BY_DEFAULT
  if (dmem_register_pin_flag_)
    DataMemRegisterPin(*brs_mem);
#endif
  //if (brs_mem) *brs_mem = mem->struct_obj();
  if (mem->size()==0) return IRIS_ERROR;

  //mems_.insert(mem);
  return IRIS_SUCCESS;
}

int Platform::DataMemCreate(iris_mem* brs_mem, void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type) {
  DataMem* mem = new DataMem(this, host, off, host_size, dev_size, elem_size, dim, element_type);
  if (brs_mem) mem->SetStructObject(brs_mem);
  //if (brs_mem) *brs_mem = mem->struct_obj();
  if (mem->size()==0) return IRIS_ERROR;

  //mems_.insert(mem);
  return IRIS_SUCCESS;
}

int Platform::DataMemCreate(iris_mem* brs_mem, iris_mem root_mem, int region) {
  DataMem *root = (DataMem *) get_mem_object(root_mem);
  DataMemRegion *mem= root->get_region(region);
  if (brs_mem) mem->SetStructObject(brs_mem);
  //if (brs_mem) *brs_mem = mem->struct_obj();
  //mems_.insert(mem);
  if (mem->size()==0) {
      return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

iris_mem *Platform::DataMemCreate(void *host, size_t size, int element_type) {
  DataMem* mem = new DataMem(this, host, size, element_type);
  if (mem->size()==0) return NULL;
  iris_mem *brs_mem = mem->struct_obj();
#ifndef DISABLE_PIN_BY_DEFAULT
  if (dmem_register_pin_flag_)
    DataMemRegisterPin(*brs_mem);
#endif
  return brs_mem;
}

iris_mem *Platform::DataMemCreate(void *host, size_t size, const char *symbol, int element_type) {
  DataMem* mem = new DataMem(this, host, size, symbol, element_type);
  if (mem->size()==0) return NULL;
  iris_mem *brs_mem = mem->struct_obj();
#ifndef DISABLE_PIN_BY_DEFAULT
  if (dmem_register_pin_flag_)
    DataMemRegisterPin(*brs_mem);
#endif
  return brs_mem;
}


iris_mem *Platform::DataMemCreate(void *host, size_t *size, int dim, size_t element_size, int element_type) {
  DataMem* mem = new DataMem(this, host, size, dim, element_size, element_type);
  if (mem->size()==0) return NULL;
  iris_mem *brs_mem = mem->struct_obj();
#ifndef DISABLE_PIN_BY_DEFAULT
  if (dmem_register_pin_flag_)
    DataMemRegisterPin(*brs_mem);
#endif
  return brs_mem;
}

iris_mem *Platform::DataMemCreate(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
  DataMem* mem = new DataMem(this, host, off, host_size, dev_size, elem_size, dim);
  if (mem->size()==0) return NULL;
  return mem->struct_obj();
}

iris_mem *Platform::DataMemCreate(iris_mem root_mem, int region) {
  DataMem *root = (DataMem *) get_mem_object(root_mem);
  DataMemRegion *mem= root->get_region(region);
  if (mem->size()==0) return NULL;
  return mem->struct_obj();
}

int Platform::DataMemEnableOuterDimRegions(iris_mem brs_mem) {
  DataMem *mem= (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->EnableOuterDimensionRegions();
  return IRIS_SUCCESS;
}

int Platform::MemCreate(size_t size, iris_mem* brs_mem) {
  Mem* mem = new Mem(size, this);
  //if (brs_mem) *brs_mem = mem->struct_obj();
  if (brs_mem) mem->SetStructObject(brs_mem);
  if (mem->size()==0) return IRIS_ERROR;

  //mems_.insert(mem);
  return IRIS_SUCCESS;
}

void *Platform::GetDeviceContext(int device)
{
    ASSERT(device < ndevs_);
    return devs_[device]->get_ctx();
}
void *Platform::GetDeviceStream(int device, int index)
{
    ASSERT(device < ndevs_);
    return devs_[device]->get_stream(index);
}
int Platform::MemArch(iris_mem brs_mem, int device, void** arch) {
  if (!arch) return IRIS_ERROR;
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  Device* dev = devs_[device];
  void* ret = mem->arch(dev);
  if (!ret) return IRIS_ERROR;
  *arch = ret;
  return IRIS_SUCCESS;
}

int Platform::MemMap(void* host, size_t size) {
  Mem* mem = new Mem(size, this);
  mem->SetMap(host, size);
  //mems_.insert(mem);
  present_table_->Add(host, size, mem);
  return IRIS_SUCCESS;
}

int Platform::MemUnmap(void* host) {
  BaseMem* mem = present_table_->Remove(host);
  mem->Release();
  return IRIS_SUCCESS;
}

int Platform::MemReduce(iris_mem brs_mem, int mode, int type) {
  Mem* mem = (Mem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->Reduce(mode, type);
  return IRIS_SUCCESS;
}

int Platform::MemRelease(iris_mem brs_mem) {
  BaseMem* mem = Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->Release();
  return IRIS_SUCCESS;
}

int Platform::GraphCreate(iris_graph* brs_graph) {
  Graph* graph = Graph::Create(this);
  if (brs_graph) graph->SetStructObject(brs_graph);
  return IRIS_SUCCESS;
}

int Platform::GraphFree(iris_graph brs_graph) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  delete graph;
  return IRIS_SUCCESS;
}

int Platform::GraphCreateJSON(const char* path, void** params, iris_graph* brs_graph) {
  Graph* graph = Graph::Create(this);
  if (brs_graph) graph->SetStructObject(brs_graph);
  //create a new temporary JSON to process this current graph --- it may be brand new and we don't want to mangle the old memory objects tracked for recording
  JSON* loader_json = new JSON(this);
  int retcode = loader_json->Load(graph, path, params);
  delete loader_json;
  return retcode;
}

int Platform::GraphTask(iris_graph brs_graph, iris_task brs_task, int brs_policy, const char* opt) {
  if (ndevs_ == 0) {
    _error("Cannot submit task:%lu due to no devices found on this system!", brs_task.uid);
    IncrementErrorCount();
    return IRIS_ERROR;
  }
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  Task *task = get_task_object(brs_task);
  assert(task != NULL);
  task->set_brs_policy(brs_policy);
  task->set_opt(opt);
  if (graph != NULL) graph->AddTask(task, brs_task.uid);
  else TaskSubmit(task, brs_policy, opt, 0);
  return IRIS_SUCCESS;
}

int Platform::SetTaskPolicy(iris_task brs_task, int brs_policy)
{
    Task *task = get_task_object(brs_task);
    assert(task != NULL);
    task->set_brs_policy(brs_policy);
    return IRIS_SUCCESS;
}

int Platform::GetTaskPolicy(iris_task brs_task)
{
    Task *task = get_task_object(brs_task);
    assert(task != NULL);
    return task->get_brs_policy();
}


void Platform::set_release_task_flag(bool flag, iris_task brs_task)
{
    Task *task = get_task_object(brs_task);
    assert(task != NULL);
    if (!flag && task->IsRelease()) {
        task->Retain();
        task->DisableRelease();
    }
    else if (flag && !task->IsRelease()) {
        task->EnableRelease();
        task->Release();
    }
    _debug2(" reatined task id:%lu ref_cnt:%d\n", brs_task.uid, task->ref_cnt());
}

int Platform::GraphRelease(iris_graph brs_graph) {
  graph_track_.CallBackIfObjectExists(brs_graph.uid, Graph::GraphRelease);
  return IRIS_SUCCESS;
}

int Platform::GraphRetain(iris_graph brs_graph, bool flag) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  if (graph == NULL) return IRIS_SUCCESS;
  if (flag && graph->IsRelease()) {
      graph->enable_retainable();
  }
  else if (!flag && !graph->IsRelease()) {
      graph->disable_retainable();
  }
  std::vector<Task*>* tasks = graph->tasks();
  for (std::vector<Task*>::iterator I = tasks->begin(), E = tasks->end(); I != E; ++I) {
    Task* task = *I;
    if (flag && task->IsRelease()) {
        task->Retain();
        task->DisableRelease();
        _debug2("Graph task:%lu:%s retained ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
    }
    else if (!flag && !task->IsRelease()) {
        task->EnableRelease();
        task->Release();
    }
  }
  return IRIS_SUCCESS;
}

int Platform::GraphSubmit(iris_graph brs_graph, int brs_policy, int sync) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  if (graph == NULL) { if (sync) return Synchronize(); else return IRIS_SUCCESS; }
  graph->Retain();
  std::vector<Task*>* tasks = graph->tasks();
  //graph->RecordStartTime(devs_[0]->Now());
  for (std::vector<Task*>::iterator I = tasks->begin(), E = tasks->end(); I != E; ++I) {
    Task* task = *I;

#ifdef PRINT_TASK_DEP
    printf("Task: %s task id %d\n", task->name(), task->uid());
    for(int i = 0; i < task->ndepends(); i++)
        printf("    Parents %d - %s task id %d\n", i, task->depend(i)->name(), task->depend(i)->uid());
    for(int i = 0; i < task->nchilds(); i++)
        printf("    Childs %d - %s task id %d\n", i, task->Child(i)->name(), task->Child(i)->uid());
#endif
/*    for(int i = 0; i < task->nchilds(); i++)
        printf("    Childs %d - %s\n", i, task->Child(i)->name());
*/ 
    //printf("Task name %s depend count %d\n", task->name(), task->ndepends());
    //for(int i = 0; i < task->ndepends(); i++)
    //    printf("    depend %d - %s\n", i, task->depend(i)->name());
    //preference is to honour the policy embedded in the task-graph.
    if (task->brs_policy() == iris_default) {
      task->set_brs_policy(brs_policy);
    }
    _debug2("Graph submit task:%lu:%s retained ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
    task->Retain();
    task->set_time_submit(timer_);
    task->Submit(task->brs_policy(), task->opt(), sync);
    if (recording_) json_->RecordTask(task);
    if (scheduler_) scheduler_->Enqueue(task);
    else { task->Retain(); workers_[0]->Enqueue(task); }
  }
  if (sync) graph->Wait();
#ifdef AUTO_PAR
#ifdef AUTO_FLUSH
  //printf("-------------------------------------------------------\n");
  auto_dag_->set_current_graph(NULL);
#endif
#endif
  return IRIS_SUCCESS;
}

int Platform::GraphSubmit(iris_graph brs_graph, int *order, int brs_policy, int sync) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  if (graph == NULL) { if (sync) return Synchronize(); else return IRIS_SUCCESS; }
  graph->Retain();
  std::vector<Task*> & tasks = graph->tasks_list();
  //graph->RecordStartTime(devs_[0]->Now());
  for(size_t i=0; i<tasks.size(); i++) {
    Task* task = tasks[order[i]];
    //preference is to honour the policy embedded in the task-graph.
    if (task->brs_policy() == iris_default) {
      task->set_brs_policy(brs_policy);
    }
    _debug2("Graph submit task:%lu:%s retained ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
    task->Retain();
    task->set_time_submit(timer_);
    task->Submit(task->brs_policy(), task->opt(), sync);
    if (recording_) json_->RecordTask(task);
    if (scheduler_) scheduler_->Enqueue(task);
    else { task->Retain(); workers_[0]->Enqueue(task); }
  }
  if (sync) graph->Wait();
  return IRIS_SUCCESS;
}

int Platform::GraphWait(iris_graph brs_graph) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
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
  if (task->brs_policy() & iris_ftf) {
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
  //for (std::set<Kernel*>::iterator I = kernels_.begin(), E = kernels_.end(); I != E; ++I) {
  //  if (strcmp((*I)->name(), name) == 0) return *I;
  //}
  Kernel* kernel = new Kernel(name, this);
  //kernels_.insert(kernel);
  std::string name_string = name;
  if (kernels_.find(name_string) != kernels_.end()) {
      std::vector<Kernel *> vec;
      kernels_.insert(std::pair<std::string, std::vector<Kernel *> >(name_string, vec));
  }
  kernels_[name_string].push_back(kernel);
  return kernel;
}

BaseMem* Platform::GetMem(iris_mem brs_mem) {
  //todo: mutex lock
  return Platform::GetPlatform()->get_mem_object(brs_mem);
#if 0
  for (std::set<BaseMem*>::iterator I = mems_.begin(), E = mems_.end(); I != E; ++I) {
    BaseMem* mem = *I;
    if (mem == Platform::GetPlatform()->get_mem_object(brs_mem)) return mem;
  }
  return NULL;
#endif
}

BaseMem* Platform::GetMem(void* host, size_t* off) {
  return present_table_->Get(host, off);
}

int Platform::TimerNow(double* time) {
  *time = timer_->Now();
  return IRIS_SUCCESS;
}

int Platform::InitScheduler(bool use_pthread) {
  /*
  if (ndevs_ == 1) {
    _info("No scheduler ndevs[%d]", ndevs_);
    return IRIS_SUCCESS;
  }
  */
  _info("Scheduler ndevs[%d] ndevs_enabled[%d]", ndevs_, ndevs_enabled_);
  scheduler_ = new Scheduler(this);
  if (use_pthread)
      scheduler_->Start();
  else
      scheduler_->StartWithOutThread();
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
#if 0
    double t_ker = 0.0f;
    double t_h2d = 0.0f;
    double t_d2d = 0.0f;
    double t_d2h_h2d = 0.0f;
    double t_d2o = 0.0f;
    double t_o2d = 0.0f;
    double t_d2h = 0.0f;
    size_t total_c_ker = 0;
    size_t total_c_h2d = 0;
    size_t total_c_d2d = 0;
    size_t total_c_d2h_h2d = 0;
    size_t total_c_d2o = 0;
    size_t total_c_o2d = 0;
    size_t total_c_d2h = 0;
    size_t total_size_h2d = 0;
    size_t total_size_d2d = 0;
    size_t total_size_d2h_h2d = 0;
    size_t total_size_d2o = 0;
    size_t total_size_o2d = 0;
    size_t total_size_d2h = 0;
    for (std::map<std::string, std::vector<Kernel*> >::iterator I = kernels_.begin(), 
            E = kernels_.end(); I != E; ++I) {
        std::string name = I->first;
        std::vector<Kernel*> & kernel_vector = I->second;
        double k_ker = 0.0f;
        double k_h2d = 0.0f;
        double k_d2d = 0.0f;
        double k_d2h_h2d = 0.0f;
        double k_o2d = 0.0f;
        double k_d2o = 0.0f;
        double k_d2h = 0.0f;
        size_t c_ker = 0;
        size_t c_d2d = 0;
        size_t c_d2h_h2d = 0;
        size_t c_d2o = 0;
        size_t c_o2d = 0;
        size_t c_h2d = 0;
        size_t c_d2h = 0;
        size_t size_d2d = 0;
        size_t size_d2h_h2d = 0;
        size_t size_d2o = 0;
        size_t size_o2d = 0;
        size_t size_h2d = 0;
        size_t size_d2h = 0;
        for(std::vector<Kernel *>::iterator kI = kernel_vector.begin(), kE = kernel_vector.end(); kI != kE; ++kI) {
            Kernel *kernel = *kI;
            shared_ptr<History> history = kernel->history();
            k_ker += history->t_kernel();
            k_h2d += history->t_h2d();
            k_d2d += history->t_d2d();
            k_d2h_h2d += history->t_d2h_h2d();
            k_d2o += history->t_d2o();
            k_o2d += history->t_o2d();
            k_d2h += history->t_d2h();
            c_ker += history->c_kernel();
            c_h2d += history->c_h2d();
            c_d2d += history->c_d2d();
            c_d2h_h2d += history->c_d2h_h2d();
            c_d2o += history->c_d2o();
            c_o2d += history->c_o2d();
            c_d2h += history->c_d2h();
            size_h2d += history->size_h2d();
            size_d2d += history->size_d2d();
            size_d2h_h2d += history->size_d2h_h2d();
            size_d2o += history->size_d2o();
            size_o2d += history->size_o2d();
            size_d2h += history->size_d2h();
            //printf("Name:%s kname:%s time:%f acc:%f\n", name.c_str(), kernel->get_task_name(), history->t_kernel(), k_ker);
        }
        _info("kernel[%s] k[%lf][%zu] h2d[%lf][%zu][%ld] o2d[%lf][%zu][%ld] d2o[%lf][%zu][%ld] d2h_h2d[%lf][%zu][%ld] d2d[%lf][%zu][%ld] d2h[%lf][%zu][%ld]", name.c_str(), k_ker, c_ker, k_h2d, c_h2d, size_h2d, k_o2d, c_o2d, size_o2d, k_d2o, c_d2o, size_d2o, k_d2h_h2d, c_d2h_h2d, size_d2h_h2d, k_d2d, c_d2d, size_d2d, k_d2h, c_d2h, size_d2h);
        t_ker += k_ker;
        total_c_ker += c_ker;
        t_h2d += k_h2d;
        t_d2d += k_d2d;
        t_d2h_h2d += k_d2h_h2d;
        t_o2d += k_o2d;
        t_d2o += k_d2o;
        t_d2h += k_d2h;
        total_c_h2d +=     c_h2d;
        total_c_d2d +=     c_d2d;
        total_c_d2h_h2d += c_d2h_h2d;
        total_c_o2d +=     c_o2d;
        total_c_d2o +=     c_d2o;
        total_c_d2h +=     c_d2h;
        total_size_h2d +=     size_h2d;
        total_size_d2d +=     size_d2d;
        total_size_d2h_h2d += size_d2h_h2d;
        total_size_o2d +=     size_o2d;
        total_size_d2o +=     size_d2o;
        total_size_d2h +=     size_d2h; 
    }
    t_h2d          += null_kernel()->history()->t_h2d();
    t_d2h          += null_kernel()->history()->t_d2h();
    total_c_h2d    += null_kernel()->history()->c_h2d();
    total_c_d2h    += null_kernel()->history()->c_d2h();
    total_size_h2d += null_kernel()->history()->size_h2d();
    total_size_d2h += null_kernel()->history()->size_d2h();
    _info("total kernel k[%lf][%zu] h2d[%lf][%zu][%ld] o2d[%lf][%zu][%ld] d2o[%lf][%zu][%ld] d2h_h2d[%lf][%zu][%ld] d2d[%lf][%zu][%ld] d2h[%lf][%zu][%ld]", t_ker, total_c_ker, t_h2d, total_c_h2d, total_size_h2d, t_o2d, total_c_o2d, total_size_o2d, t_d2o, total_c_d2o, total_size_d2o, t_d2h_h2d, total_c_d2h_h2d, total_size_d2h_h2d, t_d2d, total_c_d2d, total_size_d2d, t_d2h, total_c_d2h, total_size_d2h);
    return IRIS_SUCCESS;
#else
    double t_ker = 0.0f;
    double t_h2d = 0.0f;
    double t_d2d = 0.0f;
    double t_d2h_h2d = 0.0f;
    double t_d2o = 0.0f;
    double t_o2d = 0.0f;
    double t_d2h = 0.0f;
    size_t total_c_ker = 0;
    size_t total_c_h2d = 0;
    size_t total_c_d2d = 0;
    size_t total_c_d2h_h2d = 0;
    size_t total_c_d2o = 0;
    size_t total_c_o2d = 0;
    size_t total_c_d2h = 0;
    size_t total_size_h2d = 0;
    size_t total_size_d2d = 0;
    size_t total_size_d2h_h2d = 0;
    size_t total_size_d2o = 0;
    size_t total_size_o2d = 0;
    size_t total_size_d2h = 0;
    for (std::map<std::string, vector<shared_ptr<History> > >::iterator I = kernel_history_.begin(), 
            E = kernel_history_.end(); I != E; ++I) {
        std::string name = I->first;
        std::vector<shared_ptr<History>> & history_vector = I->second;
        double k_ker = 0.0f;
        double k_h2d = 0.0f;
        double k_d2d = 0.0f;
        double k_d2h_h2d = 0.0f;
        double k_o2d = 0.0f;
        double k_d2o = 0.0f;
        double k_d2h = 0.0f;
        size_t c_ker = 0;
        size_t c_d2d = 0;
        size_t c_d2h_h2d = 0;
        size_t c_d2o = 0;
        size_t c_o2d = 0;
        size_t c_h2d = 0;
        size_t c_d2h = 0;
        size_t size_d2d = 0;
        size_t size_d2h_h2d = 0;
        size_t size_d2o = 0;
        size_t size_o2d = 0;
        size_t size_h2d = 0;
        size_t size_d2h = 0;
        for(std::vector<shared_ptr<History> >::iterator kI = history_vector.begin(), kE = history_vector.end(); kI != kE; ++kI) {
            shared_ptr<History> history = *kI;
            k_ker += history->t_kernel();
            k_h2d += history->t_h2d();
            k_d2d += history->t_d2d();
            k_d2h_h2d += history->t_d2h_h2d();
            k_d2o += history->t_d2o();
            k_o2d += history->t_o2d();
            k_d2h += history->t_d2h();
            c_ker += history->c_kernel();
            c_h2d += history->c_h2d();
            c_d2d += history->c_d2d();
            c_d2h_h2d += history->c_d2h_h2d();
            c_d2o += history->c_d2o();
            c_o2d += history->c_o2d();
            c_d2h += history->c_d2h();
            size_h2d += history->size_h2d();
            size_d2d += history->size_d2d();
            size_d2h_h2d += history->size_d2h_h2d();
            size_d2o += history->size_d2o();
            size_o2d += history->size_o2d();
            size_d2h += history->size_d2h();
            //printf("Name:%s kname:%s time:%f acc:%f\n", name.c_str(), kernel->get_task_name(), history->t_kernel(), k_ker);
        }
        _info("kernel[%s] k[%lf][%zu] h2d[%lf][%zu][%ld] o2d[%lf][%zu][%ld] d2o[%lf][%zu][%ld] d2h_h2d[%lf][%zu][%ld] d2d[%lf][%zu][%ld] d2h[%lf][%zu][%ld]", name.c_str(), k_ker, c_ker, k_h2d, c_h2d, size_h2d, k_o2d, c_o2d, size_o2d, k_d2o, c_d2o, size_d2o, k_d2h_h2d, c_d2h_h2d, size_d2h_h2d, k_d2d, c_d2d, size_d2d, k_d2h, c_d2h, size_d2h);
        t_ker += k_ker;
        total_c_ker += c_ker;
        t_h2d += k_h2d;
        t_d2d += k_d2d;
        t_d2h_h2d += k_d2h_h2d;
        t_o2d += k_o2d;
        t_d2o += k_d2o;
        t_d2h += k_d2h;
        total_c_h2d +=     c_h2d;
        total_c_d2d +=     c_d2d;
        total_c_d2h_h2d += c_d2h_h2d;
        total_c_o2d +=     c_o2d;
        total_c_d2o +=     c_d2o;
        total_c_d2h +=     c_d2h;
        total_size_h2d +=     size_h2d;
        total_size_d2d +=     size_d2d;
        total_size_d2h_h2d += size_d2h_h2d;
        total_size_o2d +=     size_o2d;
        total_size_d2o +=     size_d2o;
        total_size_d2h +=     size_d2h; 
    }
    _info("total kernel k[%lf][%zu] h2d[%lf][%zu][%ld] o2d[%lf][%zu][%ld] d2o[%lf][%zu][%ld] d2h_h2d[%lf][%zu][%ld] d2d[%lf][%zu][%ld] d2h[%lf][%zu][%ld]", t_ker, total_c_ker, t_h2d, total_c_h2d, total_size_h2d, t_o2d, total_c_o2d, total_size_o2d, t_d2o, total_c_d2o, total_size_d2o, t_d2h_h2d, total_c_d2h_h2d, total_size_d2h_h2d, t_d2d, total_c_d2d, total_size_d2d, t_d2h, total_c_d2h, total_size_d2h);
    return IRIS_SUCCESS;

#endif
}

shared_ptr<Platform> Platform::singleton_ = nullptr;
std::once_flag Platform::flag_singleton_;
std::once_flag Platform::flag_finalize_;

Platform* Platform::GetPlatform() {
//  if (singleton_ == NULL) singleton_ = new Platform();
  std::call_once(flag_singleton_, []() { singleton_ = std::shared_ptr<Platform>(new Platform()); });
  return singleton_.get();
}
int Platform::GetGraphTasksCount(iris_graph brs_graph) {
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    if (graph == NULL) return 0;
    return graph->tasks_count();
}
int Platform::GetGraphTasks(iris_graph brs_graph, iris_task *tasks) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  if (graph == NULL) return 0;
  return graph->iris_tasks(tasks);
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
  if (scheduling_history_) delete scheduling_history_;
  Clean();
  Reset();
  finalize_ = true;
  pthread_mutex_unlock(&mutex_);
  fflush(stdout);
  return ret_id;
}
int Platform::VendorKernelLaunch(int dev_index, void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params)
{
    ASSERT(dev_index < ndevs_);
    Device *dev = devs_[dev_index];
    dev->VendorKernelLaunch(kernel, gridx, gridy, gridz, blockx, blocky, blockz, shared_mem_bytes, stream, params);
    return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


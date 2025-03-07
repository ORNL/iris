#include "DeviceHIP.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderHIP.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "Worker.h"
#include "Mem.h"
#include "Reduction.h"
#include "Task.h"
#include "Utils.h"
#include <iostream>
#include <string>

#define HIP_STREAM_DEFAULT 0

namespace iris {
namespace rt {

DeviceHIP::DeviceHIP(LoaderHIP* ld, LoaderHost2HIP *host2hip_ld, hipDevice_t dev, int ordinal, int devno, int platform, int local_devno) : Device(devno, platform) {
  ld_ = ld;
  local_devno_ = local_devno;
  set_async(true && Platform::GetPlatform()->is_async()); 
  atleast_one_command_ = false;
  host2hip_ld_ = host2hip_ld;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  ordinal_ = ordinal;
  peers_count_ = 0;
#ifndef DISABLE_D2D
  enableD2D();
#endif
  dev_ = dev;
  strcpy(vendor_, "Advanced Micro Devices");
  hipError_t err = ld_->hipDeviceGetName(name_, sizeof(name_), dev_);
  _hiperror(err);
  hipDeviceProp_t props;
  ld_->hipGetDeviceProperties(&props, dev_);
  if (strlen(name_) == 0) {
      strcpy(name_, props.gcnArchName);
  }
  int ae=nqueues_;
  //ld_->hipDeviceGetAttribute(&ae, hipDeviceAttributeAsyncEngineCount, 0);
  //n_copy_engines_ = props.copyEngineCount;
  nqueues_ = ae;
  std::string  name_str = name_;
  if (name_str.find("gfx") != std::string::npos)
      name_str = name_str.replace(name_str.find("gfx"), 3, "GFX");
  strcpy(name_, name_str.c_str());
  type_ = iris_amd;
  model_ = iris_hip;
  streams_ = new hipStream_t[nqueues_*2];
  memset(streams_, 0, sizeof(hipStream_t)*nqueues_*2);
  //memset(start_time_event_, 0, sizeof(hipEvent_t)*IRIS_MAX_DEVICE_NQUEUES);
  single_start_time_event_ = NULL;
  err = ld_->hipDriverGetVersion(&driver_version_);
  _hiperror(err);
  sprintf(version_, "AMD HIP %d", driver_version_);
  _trace("device[%d] platform[%d] vendor[%s] device[%s] ordinal[%d] type[%d] version[%s] async_engines[%d] copy_engines[%d]", devno_, platform_, vendor_, name_, ordinal_, type_, version_, ae, n_copy_engines_);
}

DeviceHIP::~DeviceHIP() {
    _trace("HIP device:%d is getting destroyed", devno());
    host2hip_ld_->finalize(devno());
    if (julia_if_ != NULL) julia_if_->finalize(devno());
    for (int i = 0; i < nqueues_; i++) {
      if (streams_[i] != NULL) {
        hipError_t err = ld_->hipStreamDestroy(streams_[i]);
        _hiperror(err);
      }
      //DestroyEvent(start_time_event_[i]);
    }
    delete [] streams_;
    if (is_async(false) && platform_obj_->is_event_profile_enabled()) 
        DestroyEvent(single_start_time_event_);
    _trace("HIP device:%d is destroyed", devno());
    hipError_t err; 
    err = ld_->hipDeviceReset();
    _hiperror(err);
    err = ld_->hipCtxDestroy(ctx_);
    _hiperror(err);
}
bool DeviceHIP::IsAddrValidForD2D(BaseMem *mem, void *ptr)
{
    int data;
    hipError_t err = ld_->hipCtxSetCurrent(ctx_);
    _hiperror(err);
    err = ld_->hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr);
    if (err == hipSuccess) return true;
    return false;
}
bool DeviceHIP::IsD2DPossible(Device *target)
{
  if (peer_access_ == NULL) return true;
  if (peer_access_[((DeviceHIP *)target)->dev_]) return true;
  return false;
}
void DeviceHIP::SetPeerDevices(int *peers, int count)
{
    std::copy(peers, peers+count, peers_);
    peers_count_ = count;
    peer_access_ = new int[peers_count_];
    memset(peer_access_, 0, sizeof(int)*peers_count_);
}
void DeviceHIP::EnablePeerAccess()
{
    hipError_t err;
    int offset_dev = devno_ - dev_;
    for(int i=0; i<peers_count_; i++) {
        hipDevice_t target_dev = peers_[i];
        if (target_dev == dev_) continue;
        err = ld_->hipCtxSetCurrent(ctx_);
        _hiperror(err);
        err = ld_->hipDeviceCanAccessPeer(&peer_access_[i], dev_, target_dev);
        _hiperror(err);
        int can_access=peer_access_[i];
        if (can_access) {
            DeviceHIP *target = (DeviceHIP *)platform_obj_->device(offset_dev + target_dev);
            hipCtx_t target_ctx = target->ctx_;
            //printf("Can access dev:%d -> %d = %d\n", dev_, target_dev, can_access);
            err = ld_->hipCtxSetCurrent(ctx_);
            _hiperror(err);
            //err = ld_->hipDeviceEnablePeerAccess(target_dev, 0);
            err = ld_->hipCtxEnablePeerAccess(target_ctx, 0);
            _hiperror(err);
        }
    }
}
int DeviceHIP::Compile(char* src, const char *out, const char *flags) {
  char default_comp_flags[] = "--genco";
  char cmd[1024];
  memset(cmd, 0, 256);
  if (flags == NULL) 
      flags = default_comp_flags;
  if (out == NULL) 
      out = kernel_path();
  sprintf(cmd, "hipcc %s -o %s %s > /dev/null 2>&1", src, out, flags);
  if (system(cmd) != EXIT_SUCCESS) {
    int result = system("hipcc --version > /dev/null 2>&1");
    if (result == 0) {
        _error("cmd[%s]", cmd);
        worker_->platform()->IncrementErrorCount();
        return IRIS_ERROR;
    }
    else {
        _warning("hipcc is not available for JIT compilation of cmd [%s]", cmd);
        return IRIS_WARNING;
    }
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::Init() {
  int tb=0, mc=0, bx=0, by=0, bz=0, dx=0, dy=0, dz=0, ck=0; //, ae;
  hipError_t err = ld_->hipSetDevice(ordinal_);
  _hiperror(err);
  err = ld_->hipInit(0);
  _hiperror(err);
  err = ld_->hipCtxCreate(&ctx_, hipDeviceScheduleAuto, ordinal_);
  //EnablePeerAccess();
  _hiperror(err);
  if (is_async(false)) {
      for (int i = 0; i < nqueues_; i++) {
          err = ld_->hipStreamCreate(streams_ + i);
          _hiperror(err);
          if (i < n_copy_engines_) continue;
          streams_[i+nqueues_-n_copy_engines_] = streams_[i];
          //RecordEvent((void **)(start_time_event_+i), i, iris_event_default);
      }
      if (platform_obj_->is_event_profile_enabled()) {
          double start_time = timer_->Now();
          RecordEvent((void **)(&single_start_time_event_), -1, iris_event_default);
          double end_time = timer_->Now();
          set_first_event_cpu_begin_time(start_time);
          set_first_event_cpu_end_time(end_time);
          _event_prof_debug("Event start time of device:%f end time of record:%f", first_event_cpu_begin_time(), first_event_cpu_end_time());
      }
  }
  char flags[128];
  sprintf(flags, "-shared -fPIC");
  LoadDefaultKernelLibrary("DEFAULT_HIP_KERNELS", flags);

  err = ld_->hipGetDevice(&devid_);
  _hiperror(err);
  //host2hip_ld_->set_dev(devno(), model());
  host2hip_ld_->init(devno());
  err = ld_->hipDeviceGetAttribute(&tb, hipDeviceAttributeMaxThreadsPerBlock, devid_);
  err = ld_->hipDeviceGetAttribute(&mc, hipDeviceAttributeMultiprocessorCount, devid_);
  err = ld_->hipDeviceGetAttribute(&bx, hipDeviceAttributeMaxBlockDimX, devid_);
  err = ld_->hipDeviceGetAttribute(&by, hipDeviceAttributeMaxBlockDimY, devid_);
  err = ld_->hipDeviceGetAttribute(&bz, hipDeviceAttributeMaxBlockDimZ, devid_);
  err = ld_->hipDeviceGetAttribute(&dx, hipDeviceAttributeMaxGridDimX, devid_);
  err = ld_->hipDeviceGetAttribute(&dy, hipDeviceAttributeMaxGridDimY, devid_);
  err = ld_->hipDeviceGetAttribute(&dz, hipDeviceAttributeMaxGridDimZ, devid_);
  err = ld_->hipDeviceGetAttribute(&ck, hipDeviceAttributeConcurrentKernels, devid_);
  max_work_group_size_ = tb;
  max_compute_units_ = mc;
  max_block_dims_[0] = bx;
  max_block_dims_[1] = by;
  max_block_dims_[2] = bz;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;

  _info("devid[%d] platform[%d] vendor[%s] device[%s] ordinal[%d] type[%d] version[%s] async_engines[%d] copy_engines[%d] max_compute_units[%zu] max_work_group_size_[%zu] max_work_item_sizes[%zu,%zu,%zu] max_block_dims[%d,%d,%d] concurrent_kernels[%d]", devid_, platform_, vendor_, name_, ordinal_, type_, version_, nqueues_, n_copy_engines_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], max_block_dims_[0], max_block_dims_[1], max_block_dims_[2], ck);

  if (julia_if_ != NULL) julia_if_->init(devno());
  char* path = (char *)kernel_path();
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
    _trace("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return IRIS_SUCCESS;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  ld_->Lock();
  err = ld_->hipModuleLoad(&module_, path);
  ld_->Unlock();
  if (err != hipSuccess) {
    _hiperror(err);
    _error("srclen[%zu] src\n%s", srclen, src);
    if (src) free(src);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceHIP::ResetMemory(Task *task, Command *cmd, BaseMem *mem) {
    int stream_index = 0;
    hipError_t err;
    bool async = false;
    if (is_async(task)) {
        stream_index = GetStream(task); //task->uid() % nqueues_; 
        async = true;
        if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
    }
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }

    ResetData & reset_data = cmd->reset_data();
    if (cmd->reset_data().reset_type_ == iris_reset_memset) {
        uint8_t reset_value = reset_data.value_.u8;
        if (async)
            err = ld_->hipMemsetAsync(mem->arch(this), reset_value, mem->size(), streams_[stream_index]);
        else
            err = ld_->hipMemset(mem->arch(this), reset_value, mem->size());
        _hiperror(err);
        if (err != hipSuccess) {
           worker_->platform()->IncrementErrorCount();
           return IRIS_ERROR;
        }
    }
    else if (ld_default() != NULL) {
        pair<bool, int8_t> out = mem->IsResetPossibleWithMemset(reset_data);
        if (out.first) {
            if (async)
                err = ld_->hipMemsetAsync(mem->arch(this), out.second, mem->size(), streams_[stream_index]);
            else
                err = ld_->hipMemset(mem->arch(this), out.second, mem->size());
            _hiperror(err);
            if (err != hipSuccess) {
                worker_->platform()->IncrementErrorCount();
                return IRIS_ERROR;
            }
        }
        else if (mem->GetMemHandlerType() == IRIS_DMEM || 
                mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
            size_t elem_size = ((DataMem*)mem)->elem_size();
            if (async)
                CallMemReset(mem, mem->size()/elem_size, cmd->reset_data(), streams_[stream_index]);
            else
                CallMemReset(mem, mem->size()/elem_size, cmd->reset_data(), NULL);
        }
        else {
            _error("Unknow reset type for memory:%lu\n", mem->uid());
        }
    }
    else {
        _error("Couldn't find shared library of HIP dev:%d default kernels with reset APIs", devno()); 
        return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}

void DeviceHIP::RegisterPin(void *host, size_t size)
{
    hipError_t err = ld_->hipHostRegister(host, size, hipHostRegisterDefault);
    //_printf("Registering mem:%p size:%lu err:%d\n", host, size, err);
    _hipwarning(err);
    //ld_->hipHostRegister(host, size, hipHostRegisterMapped);
}

void DeviceHIP::UnRegisterPin(void *host)
{
    hipError_t err = ld_->hipHostUnregister(host);
    //_printf("Registering mem:%p err:%d\n", host, err);
    _hipwarning(err);
    //ld_->hipHostRegister(host, size, hipHostRegisterMapped);
}

void DeviceHIP::set_can_share_host_memory_flag(bool flag)
{
    hipError_t err;
    can_share_host_memory_ = flag;
    err = ld_->hipSetDeviceFlags(hipDeviceMapHost);
    _hiperror(err);
}
void *DeviceHIP::GetSharedMemPtr(void* mem, size_t size) 
{ 
    hipError_t err;
    void** hipmem = NULL;
    err = ld_->hipHostRegister(mem, size, hipHostRegisterDefault);
    _hiperror(err);
    err = ld_->hipHostGetDevicePointer((void **)&hipmem, mem, 0); 
    _hiperror(err);
    ASSERT(hipmem != NULL);
    return hipmem; 
}
int DeviceHIP::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
  if (IsContextChangeRequired()) {
      hipError_t err = ld_->hipCtxSetCurrent(ctx_);
      _hiperror(err);
  }
  void** hipmem = mem_addr;
  int stream = mem->recommended_stream(devno());
  bool async = (is_async(false) && stream != DEFAULT_STREAM_INDEX && stream >=0);
  bool l_async = platform_obj_->is_malloc_async() && async && stream >= 0;
  hipError_t err;
  if (l_async)
     err = ld_->hipMallocAsync(hipmem, size, streams_[stream]);
  else
     err = ld_->hipMalloc(hipmem, size);
  _hiperror(err);
  if (err != hipSuccess) {
     worker_->platform()->IncrementErrorCount();
     return IRIS_ERROR;
  }
  if (reset)  {
      if (mem->reset_data().reset_type_ == iris_reset_memset) {
          if (l_async)
              err = ld_->hipMemsetAsync(*hipmem, 0, size, streams_[stream]);
          else
              err = ld_->hipMemset(*hipmem, 0, size);
          _hiperror(err);
          if (err != hipSuccess) {
              worker_->platform()->IncrementErrorCount();
              return IRIS_ERROR;
          }
      }
      else if (ld_default() != NULL) {
          pair<bool, int8_t> out = mem->IsResetPossibleWithMemset();
          if (out.first) {
              if (l_async)
                  err = ld_->hipMemsetAsync(*hipmem, out.second, size, streams_[stream]);
              else
                  err = ld_->hipMemset(*hipmem, out.second, size);
              _hiperror(err);
              if (err != hipSuccess) {
                  worker_->platform()->IncrementErrorCount();
                  return IRIS_ERROR;
              }

          }
          else if (mem->GetMemHandlerType() == IRIS_DMEM || 
                  mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
              size_t elem_size = ((DataMem*)mem)->elem_size();
              if (l_async)
                  CallMemReset(mem, size/elem_size, mem->reset_data(), streams_[stream]);
              else
                  CallMemReset(mem, size/elem_size, mem->reset_data(), NULL);
          }
          else {
              _error("Unknow reset type for memory:%lu\n", mem->uid());
          }
      }
      else {
          _error("Couldn't find shared library of HIP dev:%d default kernels with reset APIs", devno()); 
          return IRIS_ERROR;
      }
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::MemFree(BaseMem *mem, void* mem_addr) {
  void* hipmem = mem_addr;
  int stream = mem->recommended_stream(devno());
  bool async = (is_async(false) && stream != DEFAULT_STREAM_INDEX && stream >=0);
  hipError_t err;
  //printf("Addr: %p free async:%d\n", mem_addr, async);
  //if (async) 
      //err = ld_->hipFreeAsync(hipmem, streams_[stream]);
  //else
      err = ld_->hipFree(hipmem);
  _hiperror(err);
  if (err != hipSuccess){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

bool DeviceHIP::IsContextChangeRequired() {
    return (worker()->self() != worker()->thread());
}
void DeviceHIP::SetContextToCurrentThread()
{
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
}
void DeviceHIP::ResetContext()
{
    //hipCtx_t ctx;
    //ld_->hipCtxGetCurrent(&ctx);
    //_trace("HIP resetting context switch dev[%d][%s] self:%p thread:%p", devno_, name_, (void *)worker()->self(), (void *)worker()->thread());
    //_trace("Resetting Context Switch: %p %p", ctx, ctx_);
    ld_->hipCtxSetCurrent(ctx_);
}
int DeviceHIP::MemD2D(Task *task, Device *src_dev, BaseMem *mem, void *dst, void *src, size_t size) {
  if (mem->is_usm(devno()) || (dst == src) ) return IRIS_SUCCESS;
  atleast_one_command_ = true;
  if (IsContextChangeRequired()) {
      _trace("HIP context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  bool error_occured = false;
  int stream_index=0;
  hipError_t err;
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  if (async) {
      err = ld_->hipMemcpyDtoDAsync(dst, src, size, streams_[stream_index]);
      _hiperror(err);
      if (err != hipSuccess) error_occured = true;
  } 
  else {
      err = ld_->hipMemcpyDtoD(dst, src, size);
      _hiperror(err);
      if (err != hipSuccess) error_occured = true;
  }
  _trace("dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, size, stream_index);
  if (error_occured){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}
int DeviceHIP::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  bool error_occured = false;
  hipError_t err;
  atleast_one_command_ = true;
  if (IsContextChangeRequired()) {
      _trace("HIP context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  void* hipmem = mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
  int stream_index=0;
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  if (dim == 2) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host, stream_index);
      size_t host_row_pitch = elem_size * host_sizes[0];
      void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
      if (!async) {
          err = ld_->hipMemcpy2D((char*) hipmem, dev_sizes[0]*elem_size, host_start,
                  host_row_pitch, dev_sizes[0]*elem_size, dev_sizes[1], 
                  hipMemcpyHostToDevice);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      } 
      else {
          err = ld_->hipMemcpy2DAsync((char*) hipmem, dev_sizes[0]*elem_size, host_start,
                  host_row_pitch, dev_sizes[0]*elem_size, dev_sizes[1], 
                  hipMemcpyHostToDevice, streams_[stream_index]);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
#if 0
      printf("H2D: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), hipmem, host, host_start, elem_size);
      float *A = (float *) host;
      for(int i=0; i<dev_sizes[1]; i++) {
          int ai = off[1] + i;
          for(int j=0; j<dev_sizes[0]; j++) {
              int aj = off[0] + j;
              printf("%10.1lf ", A[ai*host_sizes[1]+aj]);
          }
      }
      printf("\n");
#endif
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d] ref_cn5:%u", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], size, host, stream_index, task->ref_cnt());
      if (!async) {
          err = ld_->hipMemcpyHtoD((char*) hipmem, (uint8_t *)host + off[0]*elem_size, size);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
      else {
          err = ld_->hipMemcpyHtoDAsync((char*) hipmem, (uint8_t *)host + off[0]*elem_size, size, streams_[stream_index]);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
#if 0
      printf("H2D: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), hipmem+off[0], host, host, elem_size);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _event_prof_debug("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]\n", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, size, host, stream_index);
  _trace("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]\n", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, size, host, stream_index);
  if (error_occured){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  atleast_one_command_ = true;
  if (IsContextChangeRequired()) {
      _trace("HIP context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  void* hipmem = mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
  int stream_index=0;
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  bool error_occured = false;
  hipError_t err;
  if (dim == 2) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)hipmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host, stream_index);
      size_t host_row_pitch = elem_size * host_sizes[0];
      void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
      if (!async) {
          err = ld_->hipMemcpy2D((char*) host_start, host_sizes[0]*elem_size, hipmem,
                  dev_sizes[0]*elem_size, dev_sizes[0]*elem_size, dev_sizes[1], 
                  hipMemcpyDeviceToHost);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
      else {
          err = ld_->hipMemcpy2DAsync((char*) host_start, host_sizes[0]*elem_size, hipmem,
                  dev_sizes[0]*elem_size, dev_sizes[0]*elem_size, dev_sizes[1], 
                  hipMemcpyDeviceToHost, streams_[stream_index]);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
#if 0
      printf("D2H: %ld:%s mem:%ld:%p dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), hipmem, host, host_start, elem_size);
      float *A = (float *) host;
      for(int i=0; i<dev_sizes[1]; i++) {
          int ai = off[1] + i;
          for(int j=0; j<dev_sizes[0]; j++) {
              int aj = off[0] + j;
              printf("%10.1lf ", A[ai*host_sizes[1]+aj]);
          }
      }
      printf("\n");
      if (task->uid() == 277) {
          printf("Situation\n");
      }
#endif
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], size, host, stream_index);
      if (!async)  {
          err = ld_->hipMemcpyDtoH((uint8_t *)host + off[0]*elem_size, (char*) hipmem, size);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
      else {
          err = ld_->hipMemcpyDtoHAsync((uint8_t *)host + off[0]*elem_size, (char*) hipmem, size, streams_[stream_index]);
          _hiperror(err);
          if (err != hipSuccess) error_occured = true;
      }
#if 0
      printf("D2H: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), hipmem+off[0], host, host, elem_size);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _event_prof_debug("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]\n", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, size, host, stream_index);
  _trace("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, size, host, stream_index);
  //for(int i=0; i<size/4; i++) {
  //   printf("D:%d (%f) ", i, *(((float *)host)+i));
  //}
  //printf("\n");
  if (error_occured){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
#ifdef TRACE_DISABLE
  hipCtx_t ctx;
  ld_->hipCtxGetCurrent(&ctx);
  _trace("Getting Context for Kernel launch Context Switch: dev:%d cctx:%p octx:%p self:%p thread:%p", devno_, ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
  if (ctx != ctx_) {
      _trace("Context wrong for HIP resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      _trace("Context wrong for Kernel launch Context Switch: %p %p", ctx, ctx_);
  }
#endif
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  int kernel_idx=-1;
  if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->host_kernel(&kernel_idx, name) == IRIS_SUCCESS) {
      *kernel_bin = host2hip_ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      return IRIS_SUCCESS;
  }
  if (IsContextChangeRequired()) {
      _trace("Changed Context for HIP resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  if (native_kernel_not_exists()) {
      if (report_error) {
          _error("HIP kernel:%s not found !", name);
          worker_->platform()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  hipFunction_t* hipkernel = (hipFunction_t*) kernel_bin;
  hipError_t err = ld_->hipModuleGetFunction(hipkernel, module_, name);
  if (report_error) _hiperror(err);
  if (err != hipSuccess){
      if (report_error) {
          _error("HIP kernel:%s not found !", name);
          worker_->platform()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  char name_off[256];
  memset(name_off, 0, sizeof(name_off));
  sprintf(name_off, "%s_with_offsets", name);
  hipFunction_t hipkernel_off;
  err = ld_->hipModuleGetFunction(&hipkernel_off, module_, name_off);
  if (err == hipSuccess) {
      kernels_offs_.insert(std::pair<hipFunction_t, hipFunction_t>(*hipkernel, hipkernel_off));
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  if (value) params_[idx] = value;
  else {
    shared_mem_offs_[idx] = shared_mem_bytes_;
    params_[idx] = shared_mem_offs_ + idx;
    shared_mem_bytes_ += size;
  }
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  if (kernel->is_vendor_specific_kernel(devno_)) {
     host2hip_ld_->setarg(
            kernel->GetParamWrapperMemory(), kindex, size, value);
  }
  else if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
     julia_if_->setarg(
            kernel->GetParamWrapperMemory(), kindex, size, value);
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
    void **dev_alloc_ptr = mem->arch_ptr(this);
    void *dev_ptr = NULL;
    size_t size = mem->size() - off;
    if (off) {
        *(mem->archs_off() + devno_) = (void*) ((uint8_t *) *dev_alloc_ptr + off);
        params_[idx] = mem->archs_off() + devno_;
        dev_ptr = *(mem->archs_off() + devno_);
    } else {
        params_[idx] = dev_alloc_ptr;
        dev_ptr = *dev_alloc_ptr; 
    }
    _debug2("task:%lu:%s idx:%d::%d off:%lu dev_ptr:%p (%p) dev_alloc_ptr:%p", 
            kernel->task()->uid(), kernel->task()->name(),
            idx, kindex, off, dev_ptr, *dev_alloc_ptr, dev_alloc_ptr);
    if (max_arg_idx_ < idx) max_arg_idx_ = idx;
    if (kernel->is_vendor_specific_kernel(devno_)) {
        host2hip_ld_->setmem(
                kernel->GetParamWrapperMemory(), kindex, dev_ptr, size);
    }
    else if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
        julia_if_->setmem(
                kernel->GetParamWrapperMemory(), mem, kindex, dev_ptr, size);
    }
    return IRIS_SUCCESS;
}

void DeviceHIP::CheckVendorSpecificKernel(Kernel *kernel) {
  kernel->set_vendor_specific_kernel(devno_, false);
  if (host2hip_ld_->host_kernel(kernel->GetParamWrapperMemory(), kernel->name())==IRIS_SUCCESS) {
          kernel->set_vendor_specific_kernel(devno_, true);
  }
  kernel->set_vendor_specific_kernel_check(devno_, true);
}

int DeviceHIP::KernelLaunchInit(Command *cmd, Kernel* kernel) {
    int stream_index = 0;
    hipStream_t *kstream = NULL;
    int nstreams = 0;
    if (is_async(kernel->task(), false)) {
        stream_index = GetStream(kernel->task()); //task->uid() % nqueues_; 
        if (stream_index == DEFAULT_STREAM_INDEX) { stream_index = 0; }
        kstream = &streams_[stream_index];
        //nstreams = nqueues_ - stream_index;
        nstreams = nqueues_-n_copy_engines_;
    }
    host2hip_ld_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
    //printf(" Is task julia enabled:%d param:%p\n", kernel->task()->enable_julia_if(), kernel->GetParamWrapperMemory());
    if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
        julia_if_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
        julia_if_->set_julia_kernel_type(kernel->GetParamWrapperMemory(), kernel->task()->julia_kernel_type());
    }
    return IRIS_SUCCESS;
}

void DeviceHIP::VendorKernelLaunch(void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params) 
{ 
  printf("IRIS Received kernel:%p stream:%p\n", kernel, stream);
  if (IsContextChangeRequired()) {
      ld_->hipCtxSetCurrent(ctx_);
  }
  hipError_t err = ld_->hipModuleLaunchKernel((hipFunction_t)kernel, gridx, gridy, gridz, blockx, blocky, blockz, shared_mem_bytes, (hipStream_t)stream, params, NULL);
  _hiperror(err);
  //ld_->hipStreamSynchronize((CUstream)stream);
}


int DeviceHIP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
#ifdef TRACE_DISABLE
  hipCtx_t ctx;
  ld_->hipCtxGetCurrent(&ctx);
  _trace("Getting Context for Kernel launch Context Switch: dev:%d cctx:%p octx:%p self:%p thread:%p", devno_, ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
  if (ctx != ctx_) {
      _trace("Context wrong for HIP resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      _trace("Context wrong for Kernel launch Context Switch: %p %p", ctx, ctx_);
  }
#endif
  atleast_one_command_ = true;
  hipError_t err;
  int stream_index=0;
  hipStream_t *kstream = NULL;
  bool async = false;
  int nstreams = 0;
  if (is_async(kernel->task(), false)) {
      stream_index = GetStream(kernel->task()); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
      // Though async is set to false, we still pass all streams to kernel to use it
      kstream = &streams_[stream_index];
      //nstreams = nqueues_ - stream_index;
      nstreams = nqueues_-n_copy_engines_;
  }
  if (IsContextChangeRequired()) {
      ld_->hipCtxSetCurrent(ctx_);
  }
  _event_prof_debug("kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]\n", devno_, name_, kernel->name(), kernel->get_task_name(), dim, stream_index);
  if (kernel->is_vendor_specific_kernel(devno_)) {
     if (host2hip_ld_->host_launch((void **)kstream, stream_index, nstreams, kernel->name(), 
                 kernel->GetParamWrapperMemory(), devno(), 
                 dim, off, gws) == IRIS_SUCCESS) {
         if (!async) {
             err = ld_->hipDeviceSynchronize();
             _hiperror(err);
             if (err != hipSuccess){
               worker_->platform()->IncrementErrorCount();
               return IRIS_ERROR;
             }
         }
         return IRIS_SUCCESS;
     }
     worker_->platform()->IncrementErrorCount();
     return IRIS_ERROR;
  }
  _trace("native kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, stream_index);
  hipFunction_t func = (hipFunction_t) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    if (max_compute_units_ != 0) while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0] && max_block_dims_[0] !=0) block[0] >>= 1;
  }
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };
  //int grid[3] = { (int) ((gws[0]-off[0]) / block[0]), (int) ((gws[1]-off[1]) / block[1]), (int) ((gws[2]-off[2]) / block[2]) };

  size_t blockOff_x = off[0] / block[0];
  size_t blockOff_y = off[1] / block[1];
  size_t blockOff_z = off[2] / block[2];
  if (off[0] != 0 || off[1] != 0 || off[2] != 0) {
    params_[max_arg_idx_ + 1] = &blockOff_x;
    params_[max_arg_idx_ + 2] = &blockOff_y;
    params_[max_arg_idx_ + 3] = &blockOff_z;
    if (kernels_offs_.find(func) == kernels_offs_.end()) {
      _trace("off0[%lu] cannot find %s_with_offsets kernel. ignore offsets", off[0], kernel->name());
      _error("HIP kernel name:%s with offset kernel:%s_with_offsets function is not found", kernel->name(), kernel->name());
      worker_->platform()->IncrementErrorCount();
    } else {
      func = kernels_offs_[func];
      _trace("off0[%lu] running %s_with_offsets kernel. max_arg_idx:%d", off[0], kernel->name(), max_arg_idx_);
    }
  }

  _trace("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] off[%ld,%ld,%ld] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], off[0], off[1], off[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index);
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      size_t grid_s[3] =  { (size_t)grid[0],  (size_t)grid[1],  (size_t)grid[2] };
      size_t block_s[3] = { (size_t)block[0], (size_t)block[1], (size_t)block[2] };
      julia_if_->host_launch(kernel->task()->uid(), (void **)kstream, stream_index, (void *)&ctx_, async,
                  nstreams, kernel->name(), 
                  kernel->GetParamWrapperMemory(), ordinal_,
                  dim, grid_s, block_s);
      return IRIS_SUCCESS;
  }
  if (!async) {
      err = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
      _hiperror(err);
      if (err != hipSuccess){
        worker_->platform()->IncrementErrorCount();
        return IRIS_ERROR;
      }
      err = ld_->hipDeviceSynchronize();
      _hiperror(err);
      if (err != hipSuccess){
          worker_->platform()->IncrementErrorCount();
          return IRIS_ERROR;
      }
  }
  else {
      err = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, streams_[stream_index], params_, NULL);
      _hiperror(err);
      if (err != hipSuccess){
          worker_->platform()->IncrementErrorCount();
          return IRIS_ERROR;
      }
  }
  for (int i = 0; i < IRIS_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  return IRIS_SUCCESS;
}

int DeviceHIP::Synchronize() {
  if (! atleast_one_command_) return IRIS_SUCCESS;
  hipError_t err = ld_->hipDeviceSynchronize();
  _hiperror(err);
  if (err != hipSuccess){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::RegisterCallback(int stream, CallBackType callback_fn, void *data, int flags) 
{
    _trace(" stream:%d data:%p flags:%d", stream, data, flags);
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    ASSERT(data != NULL && "Data shouldn't be null");
    //TODO: hipStreamAddCallback supports only flags = 0, it is reserved in future for nonblocking
    hipError_t err = ld_->hipStreamAddCallback(streams_[stream], (hipStreamCallback_t)callback_fn, data, iris_stream_default);
    _hiperror(err);
    if (err != hipSuccess){
     worker_->platform()->IncrementErrorCount();
     return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}

float DeviceHIP::GetEventTime(void *event, int stream) 
{ 
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    float elapsed=0.0f;
    if (event != NULL) {
        hipError_t err = ld_->hipEventElapsedTime(&elapsed, single_start_time_event_, (hipEvent_t)event);
        //printf("Elapsed:%f start_time_event:%p event:%p\n", elapsed, single_start_time_event_, event);
        if (err != 0) {
            _event_prof_debug("Error:%d dev:[%d][%s] Elapsed:%f start_time_event:%p event:%p stream:%d\n", err, devno(), name(), elapsed, single_start_time_event_, event, stream);
            
        }
        _hiperror(err);
    }
    return elapsed; 
}
void DeviceHIP::CreateEvent(void **event, int flags)
{
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    hipError_t err = ld_->hipEventCreateWithFlags((hipEvent_t *)event, flags);   
    _hiperror(err);
    _trace(" event:%p flags:%d", event, flags);
    if (err != hipSuccess)
        worker_->platform()->IncrementErrorCount();
    //printf("Create dev:%d event:%p\n", devno(), *event);
}
void DeviceHIP::RecordEvent(void **event, int stream, int event_creation_flag)
{
    _trace(" event:%p stream:%d", *event, stream);
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    if (*event == NULL)
        CreateEvent(event, event_creation_flag);
    ASSERT(event != NULL && "Event shouldn't be null");
    hipError_t err;
    if (stream == -1)
        err = ld_->hipEventRecord(*((hipEvent_t*)event), 0);
    else
        err = ld_->hipEventRecord(*((hipEvent_t*)event), streams_[stream]);
    _hiperror(err);
    if (err != hipSuccess)
        worker_->platform()->IncrementErrorCount();
    _event_debug("Recorded dev:[%d]:[%s] event:%p stream:%d err:%d", devno(), name(), *event, stream, err);
}
void DeviceHIP::WaitForEvent(void *event, int stream, int flags)
{
    _trace(" event:%p stream:%d flags:%d", event, stream, flags);
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    ASSERT(event != NULL && "Event shouldn't be null");
    hipError_t err = ld_->hipStreamWaitEvent(streams_[stream], (hipEvent_t)event, flags);
    _hiperror(err);
    if (err != hipSuccess)
        worker_->platform()->IncrementErrorCount();
}
void DeviceHIP::DestroyEvent(void *event)
{
    _trace(" event:%p ", event);
    ASSERT(event != NULL && "Event shouldn't be null");
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    //printf("Destroy dev:%d event:%p\n", devno(), event);
    hipError_t err = ld_->hipEventDestroy((hipEvent_t) event);
    _hiperror(err);
    if (err != hipSuccess)
        worker_->platform()->IncrementErrorCount();
}
void DeviceHIP::EventSynchronize(void *event)
{
    _trace(" event:%p ", event);
    ASSERT(event != NULL && "Event shouldn't be null");
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    hipError_t err = ld_->hipEventSynchronize((hipEvent_t) event);
    _hiperror(err);
    if (err != hipSuccess)
        worker_->platform()->IncrementErrorCount();
}


} /* namespace rt */
} /* namespace iris */


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
  ld_default_ = NULL;
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
    if (ld_default_ != NULL)
        ld_default_->finalize(devno());
    if (ld_default_ != NULL) delete ld_default_;
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
bool DeviceHIP::IsD2DPossible(Device *target)
{
  if (peer_access_ == NULL) return true;
  if (peer_access_[((DeviceHIP *)target)->dev_]) return true;
  return false;
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
        platform_obj()->IncrementErrorCount();
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
  if (platform_obj()->is_default_kernels_load()) {
      char flags[128];
      sprintf(flags, "-shared -g -fPIC -DENABLE_IRIS_HOST2HIP_APIS ");
      string ld_default_path = CompileDefaultKernelLibrary("DEFAULT_HIP_KERNELS", flags);
      if (Utils::Exist(ld_default_path.c_str())) {
          ld_default_ = new LoaderHost2HIP(ld_default_path);
          if (ld_default_->Load() != IRIS_SUCCESS) {
            _trace("%s", "skipping Host2HIP wrapper calls");
          }
      }
  }

  err = ld_->hipGetDevice(&devid_);
  _hiperror(err);
  //host2hip_ld_->set_dev(devno(), model());
  host2hip_ld_->init(devno());
  if (ld_default_ != NULL)
    ld_default_->init(devno());
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
    platform_obj()->IncrementErrorCount();
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
        err = ld_->hipCtxSetCurrent(ctx_);
        _hiperror(err);
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
           platform_obj()->IncrementErrorCount();
           return IRIS_ERROR;
        }
    }
    else if (ld_default_ != NULL) {
        pair<bool, int8_t> out = mem->IsResetPossibleWithMemset(reset_data);
        if (out.first) {
            if (async)
                err = ld_->hipMemsetAsync(mem->arch(this), out.second, mem->size(), streams_[stream_index]);
            else
                err = ld_->hipMemset(mem->arch(this), out.second, mem->size());
            _hiperror(err);
            if (err != hipSuccess) {
                platform_obj()->IncrementErrorCount();
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
    hipError_t err = ld_->hipCtxSetCurrent(ctx_);
    _hiperror(err);
    err = ld_->hipHostRegister(host, size, hipHostRegisterDefault);
    //_printf("Registering mem:%p size:%lu err:%d\n", host, size, err);
    _hipwarning(err);
    //ld_->hipHostRegister(host, size, hipHostRegisterMapped);
}

void DeviceHIP::UnRegisterPin(void *host)
{
    hipError_t err = ld_->hipCtxSetCurrent(ctx_);
    _hiperror(err);
    err = ld_->hipHostUnregister(host);
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
    //_hiperror(err);
    err = ld_->hipHostGetDevicePointer((void **)&hipmem, mem, 0); 
    _hiperror(err);
    ASSERT(hipmem != NULL);
    return hipmem; 
}
int DeviceHIP::MemAlloc(void** mem_addr, size_t size, void *stream_v, bool reset) {
    if (IsContextChangeRequired()) {
        hipError_t err = ld_->hipCtxSetCurrent(ctx_);
        _hiperror(err);
    }
    hipStream_t stream = (hipStream_t) stream_v;
    void** hipmem = mem_addr;
    bool async = (is_async(false) && stream != NULL);
    bool l_async = platform_obj_->is_malloc_async() && async;
    hipError_t err;
    if (l_async)
       err = ld_->hipMallocAsync(hipmem, size, stream);
    else
       err = ld_->hipMalloc(hipmem, size);
    _hiperror(err);
    if (err != hipSuccess) {
       platform_obj()->IncrementErrorCount();
       return IRIS_ERROR;
    }
    if (reset)  {
        if (l_async)
            err = ld_->hipMemsetAsync(*hipmem, 0, size, stream);
        else
            err = ld_->hipMemset(*hipmem, 0, size);
        _hiperror(err);
    }
    return IRIS_SUCCESS;
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
     platform_obj()->IncrementErrorCount();
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
              platform_obj()->IncrementErrorCount();
              return IRIS_ERROR;
          }
      }
      else if (ld_default_ != NULL) {
          pair<bool, int8_t> out = mem->IsResetPossibleWithMemset();
          if (out.first) {
              if (l_async)
                  err = ld_->hipMemsetAsync(*hipmem, out.second, size, streams_[stream]);
              else
                  err = ld_->hipMemset(*hipmem, out.second, size);
              _hiperror(err);
              if (err != hipSuccess) {
                  platform_obj()->IncrementErrorCount();
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

int DeviceHIP::MemFree(void* mem_addr, void *stream_v) {
  void* hipmem = mem_addr;
  hipStream_t stream = (hipStream_t) stream_v;
  bool async = (is_async(false) && stream != NULL);
  bool l_async = platform_obj_->is_malloc_async() && async;
  hipError_t err;
  //printf("Addr: %p free async:%d\n", mem_addr, async);
  if (l_async) 
      err = ld_->hipFreeAsync(hipmem, stream);
  else
      err = ld_->hipFree(hipmem);
  _hiperror(err);
  if (err != hipSuccess){
    platform_obj()->IncrementErrorCount();
    return IRIS_ERROR;
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
    platform_obj()->IncrementErrorCount();
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
      platform_obj()->IncrementErrorCount();
      return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}
void DeviceHIP::MemCpy3DDirect(void *dev, uint8_t *host, size_t *off, 
        size_t *dev_sizes, size_t *host_sizes, 
        size_t elem_size, bool host_2_dev, bool async, int stream_index)
{
    hipError_t err;
    hipMemcpy3DParms copyParams; 
    copyParams.srcArray = NULL;
    copyParams.dstArray = NULL;
    // Source is DEVICE
    size_t *src_sizes = host_sizes;
    size_t *dst_sizes = dev_sizes;
    void *src = host, *dst = (void *)dev;

    if (!host_2_dev) {
        src_sizes = dev_sizes;
        dst_sizes = host_sizes;
        src = (void *)dev;
        dst = host;
    }
    
    copyParams.srcPtr = { 
        src,
        src_sizes[0] * elem_size,  // pitch (row size in bytes)
        src_sizes[0],              // width in elements
        src_sizes[1] };             // height
    //copyParams.srcPtr = srcPtr;

    // Destination is HOST
    copyParams.dstPtr = { 
        dst,
        dst_sizes[0] * elem_size, // pitch (row size in bytes)
        dst_sizes[0],             // width in elements
        dst_sizes[1] };            // height
    //copyParams.dstPtr = dstPtr;

    // Set offsets
    if (host_2_dev) {
        copyParams.srcPos = { off[0] * elem_size, off[1], off[2] };
        copyParams.dstPos = { 0, 0, 0 };
    }
    else {
        copyParams.srcPos = { 0, 0, 0 };
        copyParams.dstPos = { off[0] * elem_size, off[1], off[2] };
    }

    // Define the tile size
    copyParams.extent = { 
        dev_sizes[0] * elem_size,   // Width in bytes
        dev_sizes[1],               // Height in elements
        dev_sizes[2] };   // Depth

    if (host_2_dev) 
        copyParams.kind = hipMemcpyHostToDevice;
    else
        copyParams.kind = hipMemcpyDeviceToHost;

    // Initialize copy parameters
    //hipMemcpy3DParms copyParams = {NULL, srcPos, srcPtr, NULL, dstPos, dstPtr, copy_ext, kind};
    //_printf(" copyParams: srcPtr:%p dstPtr:%p api:%p\n", (void *)dev, host, ld_->hipMemcpy3D);
    // Perform the copy
    if (async)
        err = ld_->hipMemcpy3DAsync(&copyParams, streams_[stream_index]);
    else
        err = ld_->hipMemcpy3D(&copyParams);
    _hiperror(err);
}
void DeviceHIP::MemCpy3D(void * dev, uint8_t *host, size_t *off, 
        size_t *dev_sizes, size_t *host_sizes, 
        size_t elem_size, bool host_2_dev, bool async, int stream_index)
{
    size_t host_row_pitch = elem_size * host_sizes[0];
    size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
    size_t dev_row_pitch = elem_size * dev_sizes[0];
    size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
    uint8_t *host_start = host + off[0]*elem_size + off[1] * host_row_pitch + off[2] * host_slice_pitch;
    size_t dev_off[3] = {  0, 0, 0 };
    void * dev_start = static_cast<char*>(dev) + dev_off[0] * elem_size + dev_off[1] * dev_row_pitch + dev_off[2] * dev_slice_pitch;
    //printf("Host:%p Dest:%p\n", host_start, dev_start);
    for(size_t i=0; i<dev_sizes[2]; i++) {
        uint8_t *z_host = host_start + i * host_slice_pitch;
        void * z_dev = static_cast<char*>(dev_start) + i * dev_slice_pitch;
        for(size_t j=0; j<dev_sizes[1]; j++) {
            uint8_t *y_host = z_host + j * host_row_pitch;
            void * d_dev = static_cast<char*>(z_dev) + j * dev_row_pitch;
            if (host_2_dev) {
                //printf("(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                if (async) {
                    hipError_t err = ld_->hipMemcpyHtoDAsync(d_dev, y_host, dev_sizes[0]*elem_size, streams_[stream_index]); _hiperror(err);
                } else {
                    hipError_t err = ld_->hipMemcpyHtoD(d_dev, y_host, dev_sizes[0]*elem_size); _hiperror(err);
                }
            }
            else {
                //printf("(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                if (async) {
                    hipError_t err = ld_->hipMemcpyDtoHAsync(y_host, d_dev, dev_sizes[0]*elem_size, streams_[stream_index]); _hiperror(err);
                } else {
                    hipError_t err = ld_->hipMemcpyDtoH(y_host, d_dev, dev_sizes[0]*elem_size); _hiperror(err);
                }
            }
        }
    }
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
  if (off[0] == 0 && off[1] == 0 && off[2] == 0 && 
          host_sizes[0] == dev_sizes[0] && 
          host_sizes[1] == dev_sizes[1] && host_sizes[2] == dev_sizes[2]) {
    // Consider it as a dim = 1
    dim = 1;
  }
  if (dim == 3) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host, stream_index);
      MemCpy3DDirect(hipmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true, async, stream_index);
  }
  else if (dim == 2) {
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
    platform_obj()->IncrementErrorCount();
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
  if (off[0] == 0 && off[1] == 0 && off[2] == 0 && 
          host_sizes[0] == dev_sizes[0] && 
          host_sizes[1] == dev_sizes[1] && host_sizes[2] == dev_sizes[2]) {
    // Consider it as a dim = 1
    dim = 1;
  } 
  if (dim == 3) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)hipmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host, stream_index);
      MemCpy3DDirect(hipmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, false, async, stream_index);
  }
  else if (dim == 2) {
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
    platform_obj()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::MemCpy(Task *task, BaseMem *mem, int dst_dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size, int src_type, int dst_type, const char *tag) 
{
  int dim = IRIS_MAX(dst_dim, src_dim);
  if (mem->is_usm(devno()) || (dst == src) ) return IRIS_SUCCESS;
  if (IsContextChangeRequired()) {
      _trace("HIP context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      hipError_t err = ld_->hipCtxSetCurrent(ctx_);
      _hiperror(err);
  }
  size_t tile_size0 = tile_size[0];
  size_t tile_size1 = tile_size[1];
  size_t tile_size2 = tile_size[2];
  size_t src_off0 = src_off[0];
  size_t src_off1 = src_off[1];
  size_t src_off2 = src_off[2];
  size_t dst_off0 = dst_off[0];
  size_t dst_off1 = dst_off[1];
  size_t dst_off2 = dst_off[2];
  size_t src_strides0 = src_strides[0];
  size_t src_strides1 = src_strides[1];
  size_t src_strides2 = src_strides[2];
  size_t dst_strides0 = dst_strides[0];
  size_t dst_strides1 = dst_strides[1];
  size_t dst_strides2 = dst_strides[2];
  int stream_index = 0;
  if ( src_size[0] != dst_size[0] || src_size[1] != dst_size[1] || 
          src_size[2] != dst_size[2] ) {
      bool is_src_continuous = Utils::IsTileContiguous(src_strides, src_size, dim);
      bool is_dst_continuous = Utils::IsTileContiguous(dst_strides, dst_size, dim);
      //printf(" is continuous:%d,%d == %d,%d\n", src_dim, is_src_continuous, dst_dim, is_dst_continuous);
      // tile_size is with respect to destination device size
      if (!is_src_continuous && !is_dst_continuous) {
        ((DataMem *)mem)->Print();
        string src_size_str = Utils::GetString(src_size, dim);
        string dst_size_str = Utils::GetString(dst_size, dim);
        string src_off_str = Utils::GetString(src_off, dim);
        string dst_off_str = Utils::GetString(dst_off, dim);
        string src_strides_str = Utils::GetString(src_strides, dim+1);
        string dst_strides_str = Utils::GetString(dst_strides, dim+1);
        string tile_size_str = Utils::GetString(tile_size, dim);
        _error("\n\n %s couldn't handle the dim:%d tile copy for DMEM dev[%d][%s] task[%ld:%s] elem_size:%lu mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] src_size:%s dst_size:%s src_off:%s dst_off:%s src_strides:%s dst_strides:%s tile_size:%s [%d]", tag, dim, devno_, name_, task->uid(), task->name(), elem_size, mem->uid(), dst, src, src_size_str.c_str(), dst_size_str.c_str(), src_off_str.c_str(), dst_off_str.c_str(), src_strides_str.c_str(), dst_strides_str.c_str(), tile_size_str.c_str(), stream_index);
        Utils::PrintStackTrace();
      }
      else if (is_src_continuous) {
        size_t offset = 0;
        for (int i=0; i<dim; i++) {
            offset += src_off[i] * src_strides[i];
        }
        src = reinterpret_cast<void *>(reinterpret_cast<uint8_t*>(src) + offset * elem_size);
        src_off0 = 0;
        src_off1 = 0;
        src_off2 = 0;
        src_strides0 = 1;    
        src_strides1 = tile_size0;
        src_strides2 = tile_size0 * tile_size1;
      }
      else { // is_dst_continuous 
        size_t offset = 0;
        for (int i=0; i<dim; i++) {
            offset += dst_off[i] * dst_strides[i];
        }
        dst = reinterpret_cast<void *>(reinterpret_cast<uint8_t*>(dst) + offset * elem_size);
        dst_off0 = 0;
        dst_off1 = 0;
        dst_off2 = 0;
        tile_size0 = src_size[0];
        tile_size1 = src_size[1];
        tile_size2 = src_size[2];
        dst_strides0 = 1;    
        dst_strides1 = tile_size0;
        dst_strides2 = tile_size0 * tile_size1;
      }
  }
  hipDeviceptr_t src_hipmem = (hipDeviceptr_t) src;
  hipDeviceptr_t  dst_hipmem = (hipDeviceptr_t) dst;
  bool error_occured = false;
  hipError_t err = hipSuccess;
  _hiperror(err);
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  _event_debug("dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] q[%d] async:%d", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, stream_index, async);
  ASSERT(src_strides0 == 1);
  ASSERT(dst_strides0 == 1);
  if (dim == 3) {
      //if (src_off0 == 0 && src_off1 == 0 && src_off2 == 0 && dst_off0 == 0 & dst_off1 == 0 & dst_off2 ==0) {

      //}
      HIP_MEMCPY3D cd = {};
      memset(&cd, 0, sizeof(HIP_MEMCPY3D));
      cd.srcLOD           = cd.dstLOD = 0;
      cd.srcArray = nullptr;
      cd.dstArray = nullptr;
      cd.srcDevice = (hipDeviceptr_t)nullptr;
      cd.dstDevice = (hipDeviceptr_t)nullptr;
      //cd.srcMemoryType    = HIP_MEMORYTYPE_DEVICE;
      cd.srcMemoryType    = (hipMemoryType) src_type;
      if (src_type == hipMemoryTypeDevice) {
          cd.srcDevice    = src_hipmem;
          cd.srcHost      = NULL;
      }
      else {
          cd.srcHost      = src;
          cd.srcDevice = (hipDeviceptr_t) NULL;
      }
      cd.srcPitch         = src_strides1 * elem_size;
      cd.srcHeight        = src_strides2 / src_strides1; //src_size[1]; 
      cd.srcXInBytes      = src_off0 * elem_size;
      cd.srcY             = src_off1;
      cd.srcZ             = src_off2;

      //cd.dstMemoryType    = HIP_MEMORYTYPE_DEVICE;
      cd.dstMemoryType    = (hipMemoryType) dst_type;
      if (dst_type == hipMemoryTypeDevice) {
          cd.dstDevice    = dst_hipmem;
      }
      else {
          cd.dstHost      = dst;
      }
      cd.dstPitch         = dst_strides1 * elem_size;
      cd.dstHeight        = dst_strides2 / dst_strides1; //dst_size[1];
      cd.dstXInBytes      = dst_off0 * elem_size;
      cd.dstY             = dst_off1;
      cd.dstZ             = dst_off2;

      cd.WidthInBytes     = tile_size0 * elem_size;
      cd.Height           = tile_size1;
      cd.Depth            = tile_size2;

      // perform the copy
      hipError_t err;
      if (async) {
          assert(stream_index >= 0);
          err = ld_->hipDrvMemcpy3DAsync(&cd, streams_[stream_index]);
      } else {
          err = ld_->hipDrvMemcpy3D(&cd);
      }
      _hiperror(err);
      if (err != hipSuccess) error_occured = true;
      if (err != hipSuccess) {
          ((DataMem *)mem)->Print();
          string src_size_str = Utils::GetString(src_size, dim);
          string dst_size_str = Utils::GetString(dst_size, dim);
          string src_off_str = Utils::GetString(src_off, dim);
          string dst_off_str = Utils::GetString(dst_off, dim);
          string src_strides_str = Utils::GetString(src_strides, dim+1);
          string dst_strides_str = Utils::GetString(dst_strides, dim+1);
          string tile_size_str = Utils::GetString(tile_size, dim);
          _error(" %s couldn't handle the dim:%d ptr:%p tile copy for DMEM dev[%d][%s] task[%ld:%s] elem_size:%lu mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] src_size:%s dst_size:%s src_off:%s dst_off:%s src_strides:%s dst_strides:%s tile_size:%s elem_size:%lu [%d]", tag, dim, ld_->hipDrvMemcpy3D, devno_, name_, task->uid(), task->name(), elem_size, mem->uid(), dst, src, src_size_str.c_str(), dst_size_str.c_str(), src_off_str.c_str(), dst_off_str.c_str(), src_strides_str.c_str(), dst_strides_str.c_str(), tile_size_str.c_str(), elem_size, stream_index);
          /*if (src_type == HIP_MEMORYTYPE_HOST) {
              iris::PrintArrayFull<int64_t>((int64_t *)src, src_size[2], src_size[1], src_size[0], " Src:");
          }
          if (dst_type == HIP_MEMORYTYPE_HOST) {
              iris::PrintArrayFull<int64_t>((int64_t *)dst, dst_size[2], dst_size[1], dst_size[0], " Dst:");
          }*/
          Utils::PrintStackTrace();
      }
  }
  else if (dim == 2) {
      // 6) Build the 2D copy descriptor
      HIP_MEMCPY2D cd = {};
      cd.srcArray = nullptr;
      cd.dstArray = nullptr;
      cd.srcMemoryType  = (hipMemoryType) src_type; 
      if (src_type == hipMemoryTypeDevice ) {
          cd.srcDevice  = src_hipmem;
          cd.srcHost    = nullptr;
      }
      else {
          cd.srcHost    = src;
          cd.srcDevice = (hipDeviceptr_t) NULL;
      }
      cd.srcPitch       = src_strides1 * elem_size;
      cd.srcXInBytes    = src_off0 * elem_size;
      cd.srcY           = src_off1;

      cd.dstMemoryType  = (hipMemoryType) dst_type;
      if (dst_type == hipMemoryTypeDevice) {
          cd.dstDevice  = dst_hipmem;
          cd.dstHost    = nullptr;
      }
      else {
          cd.dstHost    = dst;
      }
      cd.dstPitch       = dst_strides1 * elem_size;
      cd.dstXInBytes    = dst_off0 * elem_size;
      cd.dstY           = dst_off1;

      cd.WidthInBytes   = tile_size0 * elem_size;
      cd.Height         = tile_size1;
      //_hiperror(hipMemcpy2D(&cd));
      if (async) {
          ASSERT(stream_index >= 0);
          err = ld_->hipMemcpyParam2DAsync(&cd, streams_[stream_index]);
      }
      else {
          err = ld_->hipMemcpyParam2D(&cd);
      }
      _hiperror(err);
      if (err != hipSuccess) error_occured = true;
      if (err != hipSuccess) {
          ((DataMem *)mem)->Print();
          string src_size_str = Utils::GetString(src_size, dim);
          string dst_size_str = Utils::GetString(dst_size, dim);
          string src_off_str = Utils::GetString(src_off, dim);
          string dst_off_str = Utils::GetString(dst_off, dim);
          string src_strides_str = Utils::GetString(src_strides, dim+1);
          string dst_strides_str = Utils::GetString(dst_strides, dim+1);
          string tile_size_str = Utils::GetString(tile_size, dim);
          string src_str = "Host"; 
          string dst_str = "Host"; 
          if (dst_type == hipMemoryTypeDevice) {
              dst_str = "DEV"; 
          }
          if (src_type == hipMemoryTypeDevice) {
              src_str = "DEV"; 
          }
          _error("\n\n %s couldn't handle the dim:%d tile copy for DMEM dev[%d][%s] task[%ld:%s] elem_size:%lu mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] src_size:%s dst_size:%s src_off:%s dst_off:%s src_strides:%s dst_strides:%s tile_size:%s [%d] [%s -> %s]", tag, dim, devno_, name_, task->uid(), task->name(), elem_size, mem->uid(), dst, src, src_size_str.c_str(), dst_size_str.c_str(), src_off_str.c_str(), dst_off_str.c_str(), src_strides_str.c_str(), dst_strides_str.c_str(), tile_size_str.c_str(), stream_index, src_str.c_str(), dst_str.c_str());
          Utils::PrintStackTrace();
#if 0
          if (dst_type == HIP_MEMORYTYPE_DEVICE) {
              size_t total_size=128;
              int64_t data[16];
              ld_->hipMemcpyDtoH((uint8_t*)data, dst_hipmem, total_size);
              iris::PrintArrayFull<int64_t>((int64_t *)src, 4, 4, " Src:");
              iris::PrintArrayFull<int64_t>((int64_t *)data, 4, 4, " Dst:");
          }
          else if (src_type == HIP_MEMORYTYPE_DEVICE) {
            iris::PrintArrayFull<float>((float*)dst, 4, 8, " Dst:");
          }
#endif      
      }
  }
  else if (dim == 1) {
      hipMemcpyKind kind = hipMemcpyDeviceToDevice;
      if (src_type == hipMemoryTypeDevice && dst_type == hipMemoryTypeDevice)
         kind = hipMemcpyDeviceToDevice;
      else if (src_type == hipMemoryTypeHost && dst_type == hipMemoryTypeDevice)
          kind = hipMemcpyHostToDevice;
      else if (src_type == hipMemoryTypeDevice && dst_type == hipMemoryTypeHost)
         kind = hipMemcpyDeviceToHost;
      else if (src_type == hipMemoryTypeHost && dst_type == hipMemoryTypeHost)
         kind = hipMemcpyHostToHost;
      if (async) {
          ASSERT(stream_index >= 0);
          err = ld_->hipMemcpyAsync(static_cast<char*>(dst_hipmem) + dst_off0*elem_size, static_cast<char*>(src_hipmem)+src_off0*elem_size, tile_size0 * elem_size, kind, streams_[stream_index]);
      }
      else {
          err = ld_->hipMemcpy(static_cast<char*>(dst_hipmem) + dst_off0*elem_size, static_cast<char*>(src_hipmem)+src_off0*elem_size, tile_size0 * elem_size, kind);
      }
      _hiperror(err);
      if (err != hipSuccess) error_occured = true;
      if (err != hipSuccess) {
          ((DataMem *)mem)->Print();
          string src_size_str = Utils::GetString(src_size, dim);
          string dst_size_str = Utils::GetString(dst_size, dim);
          string src_off_str = Utils::GetString(src_off, dim);
          string dst_off_str = Utils::GetString(dst_off, dim);
          string src_strides_str = Utils::GetString(src_strides, dim+1);
          string dst_strides_str = Utils::GetString(dst_strides, dim+1);
          string tile_size_str = Utils::GetString(tile_size, dim);
          _error(" %s couldn't handle the dim:%d tile copy for DMEM dev[%d][%s] task[%ld:%s] elem_size:%lu mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] src_size:%s dst_size:%s src_off:%s dst_off:%s src_strides:%s dst_strides:%s tile_size:%s [%d]", tag, dim, devno_, name_, task->uid(), task->name(), elem_size, mem->uid(), dst, src, src_size_str.c_str(), dst_size_str.c_str(), src_off_str.c_str(), dst_off_str.c_str(), src_strides_str.c_str(), dst_strides_str.c_str(), tile_size_str.c_str(), stream_index);
          Utils::PrintStackTrace();
        
      }
  }
  else {
      string src_size_str = Utils::GetString(src_size, dim);
      string dst_size_str = Utils::GetString(dst_size, dim);
      string src_off_str = Utils::GetString(src_off, dim);
      string dst_off_str = Utils::GetString(dst_off, dim);
      string src_strides_str = Utils::GetString(src_strides, dim+1);
      string dst_strides_str = Utils::GetString(dst_strides, dim+1);
      string tile_size_str = Utils::GetString(tile_size, dim);
      _error(" %s couldn't handle the dim:%d tile copy for DMEM dev[%d][%s] task[%ld:%s] elem_size:%lu mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] src_size:%s dst_size:%s src_off:%s dst_off:%s src_strides:%s dst_strides:%s tile_size:%s [%d]", tag, dim, devno_, name_, task->uid(), task->name(), elem_size, mem->uid(), dst, src, src_size_str.c_str(), dst_size_str.c_str(), src_off_str.c_str(), dst_off_str.c_str(), src_strides_str.c_str(), dst_strides_str.c_str(), tile_size_str.c_str(), stream_index);
      Utils::PrintStackTrace();
  }
  _trace("dev[%d][%s] task[%ld:%s] elem_size:%lu mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), elem_size, mem->uid(), dst, src, size, stream_index);
  if (error_occured) {
  _event_debug("Error dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, mem->size(), stream_index);
  }
  //ASSERT(!error_occured && "HIP Error occured");
  if (error_occured) {
      platform_obj()->IncrementErrorCount();
      return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}
int DeviceHIP::MemD2D(Task *task, BaseMem *mem, int dst_dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size) 
{
    return MemCpy(task, mem, dst_dim, dst, dst_size, dst_off, dst_strides, src_dim, src, src_size, src_off, src_strides, tile_size, elem_size, hipMemoryTypeDevice, hipMemoryTypeDevice, "D2D"); 
}
int DeviceHIP::MemH2D(Task *task, BaseMem *mem, int dst_dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size) 
{
    return MemCpy(task, mem, dst_dim, dst, dst_size, dst_off, dst_strides, src_dim, src, src_size, src_off, src_strides, tile_size, elem_size, hipMemoryTypeHost, hipMemoryTypeDevice, "H2D"); 
}
int DeviceHIP::MemD2H(Task *task, BaseMem *mem, int dst_dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size) 
{
    return MemCpy(task, mem, dst_dim, dst, dst_size, dst_off, dst_strides, src_dim, src, src_size, src_off, src_strides, tile_size, elem_size, hipMemoryTypeDevice, hipMemoryTypeHost, "D2H"); 
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
  if (ld_default_ != NULL && 
          kernel->is_default_kernel(devno_) && 
          ld_default_->host_kernel(&kernel_idx, name) == IRIS_SUCCESS) {
      *kernel_bin = ld_default_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (julia_if_ != NULL && kernel->enable_julia_if()) {
      return IRIS_SUCCESS;
  }
  if (IsContextChangeRequired()) {
      _trace("Changed Context for HIP resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  if (native_kernel_not_exists()) {
      if (report_error) {
          _error("HIP kernel:%s not found !", name);
          platform_obj()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  hipFunction_t* hipkernel = (hipFunction_t*) kernel_bin;
  vector<string> kernel_parts;
  Utils::ReadVector(kernel_parts, name, " :");
  if (kernel_parts.size() <= 1) {
      hipError_t err = ld_->hipModuleGetFunction(hipkernel, module_, name);
      if (report_error) _hiperror(err);
      if (err != hipSuccess){
          if (report_error) {
              _error("HIP kernel:%s not found !", name);
              platform_obj()->IncrementErrorCount();
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
  }
  else {
    char *runtime_path = NULL;
    Platform::GetPlatform()->EnvironmentGet("RUNTIME_KERNELS_PATH", &runtime_path, NULL);
    string mod_name = runtime_path + kernel_parts[0] + ".hip";
    string kname = kernel_parts[1];
    if (runtime_modules_.find(mod_name) == runtime_modules_.end()) {
        char* path = (char *)mod_name.c_str();
        char* src = NULL;
        size_t srclen = 0;
        if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
          _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
          //err = ld_->cuMemFree(0);
          //_cuerror(err);
          return IRIS_SUCCESS;
        }
        hipModule_t run_module;
        _trace("loading dev[%d][%s] kernels[%s]", devno_, name_, path);
        hipError_t err = ld_->hipModuleLoad(&run_module, path);
        if (err != hipSuccess) {
            if (report_error) _hiperror(err);
            return IRIS_ERROR;
        }
        runtime_modules_.insert(make_pair(mod_name, run_module ));
    }
    hipModule_t run_module = runtime_modules_[mod_name];
    hipError_t err = ld_->hipModuleGetFunction(hipkernel, run_module, kname.c_str());
    if (report_error) _hiperror(err);
    if (err != hipSuccess) {
        if (report_error) {
            _error("HIP kernel:%s not found in runtime module:%s!", kname.c_str(), mod_name.c_str());
            platform_obj()->IncrementErrorCount();
        }
        return IRIS_ERROR;
    }
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
  else if (ld_default_ != NULL && kernel->is_default_kernel(devno_)) {
     ld_default_->setarg(
            kernel->GetParamWrapperMemory(), kindex, size, value);
  }
  else if (julia_if_ != NULL && kernel->enable_julia_if()) {
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
    else if (ld_default_ != NULL && kernel->is_default_kernel(devno_)) {
        ld_default_->setmem(
                kernel->GetParamWrapperMemory(), kindex, dev_ptr, size);
    }
    else if (julia_if_ != NULL && kernel->enable_julia_if()) {
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
  else if (ld_default_ != NULL && 
          ld_default_->host_kernel(
              kernel->GetParamWrapperMemory(),
              kernel->name())==IRIS_SUCCESS) {
      kernel->set_default_kernel(devno_, true);
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
    if (kernel->is_vendor_specific_kernel(devno_))
        host2hip_ld_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
    //printf(" Is task julia enabled:%d param:%p\n", kernel->enable_julia_if(), kernel->GetParamWrapperMemory());
    else if (ld_default_ != NULL && kernel->is_default_kernel(devno_))
        ld_default_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
    if (julia_if_ != NULL && kernel->enable_julia_if()) {
        julia_if_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
        julia_if_->set_julia_kernel_type(kernel->GetParamWrapperMemory(), kernel->julia_kernel_type());
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
     if (host2hip_ld_->host_launch((void **)kstream, stream_index, 
                nstreams, kernel->name(), 
                 kernel->GetParamWrapperMemory(), devno(), 
                 dim, off, gws) == IRIS_SUCCESS) {
         if (!async) {
             err = ld_->hipDeviceSynchronize();
             _hiperror(err);
             if (err != hipSuccess){
               platform_obj()->IncrementErrorCount();
               return IRIS_ERROR;
             }
         }
         return IRIS_SUCCESS;
     }
     platform_obj()->IncrementErrorCount();
     return IRIS_ERROR;
  }
  if (kernel->is_default_kernel(devno_)) {
     if (ld_default_->host_launch((void **)kstream, 
                 stream_index, nstreams, kernel->name(), 
                 kernel->GetParamWrapperMemory(), devno(), 
                 dim, off, gws) == IRIS_SUCCESS) {
         if (!async) {
             err = ld_->hipDeviceSynchronize();
             _hiperror(err);
             if (err != hipSuccess){
               platform_obj()->IncrementErrorCount();
               return IRIS_ERROR;
             }
         }
         return IRIS_SUCCESS;
     }
     platform_obj()->IncrementErrorCount();
     return IRIS_ERROR;
  }
  _trace("native kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, stream_index);
  hipFunction_t func = (hipFunction_t) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    if (max_compute_units_ != 0) while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0] && max_block_dims_[0] !=0) block[0] >>= 1;
  }
  int grid[3] = { (int) ((gws[0]+block[0]-1) / block[0]), (int) ((gws[1]+block[1]-1) / block[1]), (int) ((gws[2]+block[2]-1) / block[2]) };
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
      platform_obj()->IncrementErrorCount();
    } else {
      func = kernels_offs_[func];
      _trace("off0[%lu] running %s_with_offsets kernel. max_arg_idx:%d", off[0], kernel->name(), max_arg_idx_);
    }
  }

  _trace("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] off[%ld,%ld,%ld] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], off[0], off[1], off[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index);
  if (julia_if_ != NULL && kernel->enable_julia_if()) {
      size_t grid_s[3] =  { (size_t)grid[0],  (size_t)grid[1],  (size_t)grid[2] };
      size_t block_s[3] = { (size_t)block[0], (size_t)block[1], (size_t)block[2] };
      julia_if_->host_launch(kernel->task()->uid(), (void **)kstream, stream_index, (void *)&ctx_, async,
                  nstreams, kernel->name(), 
                  kernel->GetParamWrapperMemory(), ordinal_,
                  dim, gws, grid_s, block_s, kernel->shared_mem_size());
      return IRIS_SUCCESS;
  }
  if (!async) {
      err = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
      _hiperror(err);
      if (err != hipSuccess){
        platform_obj()->IncrementErrorCount();
        return IRIS_ERROR;
      }
      err = ld_->hipDeviceSynchronize();
      _hiperror(err);
      if (err != hipSuccess){
          platform_obj()->IncrementErrorCount();
          return IRIS_ERROR;
      }
  }
  else {
      err = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, streams_[stream_index], params_, NULL);
      _hiperror(err);
      if (err != hipSuccess){
          platform_obj()->IncrementErrorCount();
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
    platform_obj()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::Custom(int tag, char* params) {
  if (!cmd_handlers_.count(tag)) {
    _error("unknown tag[0x%x]", tag);
    platform_obj()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  command_handler handler = cmd_handlers_[tag];
  handler(params, this);
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
     platform_obj()->IncrementErrorCount();
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
        platform_obj()->IncrementErrorCount();
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
        platform_obj()->IncrementErrorCount();
    _event_debug("Recorded dev:[%d]:[%s] event:%p stream:%d err:%d", devno(), name(), *event, stream, err);
}
int DeviceHIP::WaitForEvent(void *event, int stream, int flags)
{
    _trace(" event:%p stream:%d flags:%d", event, stream, flags);
    if (IsContextChangeRequired()) {
        ld_->hipCtxSetCurrent(ctx_);
    }
    ASSERT(event != NULL && "Event shouldn't be null");
    hipError_t err = ld_->hipStreamWaitEvent(streams_[stream], (hipEvent_t)event, flags);
    _hiperror(err);
    if (err != hipSuccess) {
        platform_obj()->IncrementErrorCount();
        Utils::PrintStackTrace();
        return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
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
        platform_obj()->IncrementErrorCount();
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
        platform_obj()->IncrementErrorCount();
}


} /* namespace rt */
} /* namespace iris */


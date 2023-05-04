#include "DeviceHIP.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderHIP.h"
#include "BaseMem.h"
#include "Worker.h"
#include "Mem.h"
#include "Reduction.h"
#include "Task.h"
#include "Utils.h"
#include <iostream>
#include <string>

namespace iris {
namespace rt {

DeviceHIP::DeviceHIP(LoaderHIP* ld, LoaderHost2HIP *host2hip_ld, hipDevice_t dev, int ordinal, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  atleast_one_command_ = false;
  host2hip_ld_ = host2hip_ld;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  ordinal_ = ordinal;
  peers_count_ = 0;
  enableD2D();
  dev_ = dev;
  strcpy(vendor_, "Advanced Micro Devices");
  err_ = ld_->hipDeviceGetName(name_, sizeof(name_), dev_);
  _hiperror(err_);
  if (strlen(name_) == 0) {
      hipDeviceProp_t props;
      ld_->hipGetDeviceProperties(&props, dev_);
      strcpy(name_, props.gcnArchName);
  }
  std::string  name_str = name_;
  if (name_str.find("gfx") != std::string::npos)
      name_str = name_str.replace(name_str.find("gfx"), 3, "GFX");
  strcpy(name_, name_str.c_str());
  type_ = iris_amd;
  model_ = iris_hip;
  err_ = ld_->hipDriverGetVersion(&driver_version_);
  _hiperror(err_);
  sprintf(version_, "AMD HIP %d", driver_version_);
  _info("device[%d] platform[%d] vendor[%s] device[%s] ordinal[%d] type[%d] version[%s]", devno_, platform_, vendor_, name_, ordinal_, type_, version_);
}

DeviceHIP::~DeviceHIP() {
    if (host2hip_ld_->iris_host2hip_finalize){
        host2hip_ld_->iris_host2hip_finalize();
    }
    if (host2hip_ld_->iris_host2hip_finalize_handles){
        host2hip_ld_->iris_host2hip_finalize_handles(ordinal_);
    }
}
void DeviceHIP::EnablePeerAccess()
{
    for(int i=0; i<peers_count_; i++) {
        hipDevice_t target_dev = peers_[i];
        if (target_dev == dev_) continue;
        _hiperror(err_);
        int can_access=0;
        err_ = ld_->hipDeviceCanAccessPeer(&can_access, dev_, target_dev);
        if (can_access) {
            //printf("Can access dev:%d -> %d = %d\n", dev_, target_dev, can_access);
            err_ = ld_->hipDeviceEnablePeerAccess(target_dev, 0);
            _hiperror(err_);
        }
    }
}
int DeviceHIP::Compile(char* src) {
  char cmd[1024];
  memset(cmd, 0, 256);
  sprintf(cmd, "hipcc --genco %s -o %s", src, kernel_path_);
  if (system(cmd) != EXIT_SUCCESS) {
    _error("cmd[%s]", cmd);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

void DeviceHIP::SetPeerDevices(int *peers, int count)
{
    std::copy(peers, peers+count, peers_);
    peers_count_ = count;
}
int DeviceHIP::Init() {
  int tb=0, mc=0, bx=0, by=0, bz=0, dx=0, dy=0, dz=0, ck=0; //, ae;
  err_ = ld_->hipSetDevice(ordinal_);
  err_ = ld_->hipCtxCreate(&ctx_, hipDeviceScheduleAuto, ordinal_);
  EnablePeerAccess();
  if (host2hip_ld_->iris_host2hip_init != NULL) {
    host2hip_ld_->iris_host2hip_init();
  }
  if (host2hip_ld_->iris_host2hip_init_handles != NULL) {
    host2hip_ld_->iris_host2hip_init_handles(ordinal_);
  }
  _hiperror(err_);
  err_ = ld_->hipGetDevice(&devid_);
  _hiperror(err_);
  err_ = ld_->hipDeviceGetAttribute(&tb, hipDeviceAttributeMaxThreadsPerBlock, devid_);
  err_ = ld_->hipDeviceGetAttribute(&mc, hipDeviceAttributeMultiprocessorCount, devid_);
  err_ = ld_->hipDeviceGetAttribute(&bx, hipDeviceAttributeMaxBlockDimX, devid_);
  err_ = ld_->hipDeviceGetAttribute(&by, hipDeviceAttributeMaxBlockDimY, devid_);
  err_ = ld_->hipDeviceGetAttribute(&bz, hipDeviceAttributeMaxBlockDimZ, devid_);
  err_ = ld_->hipDeviceGetAttribute(&dx, hipDeviceAttributeMaxGridDimX, devid_);
  err_ = ld_->hipDeviceGetAttribute(&dy, hipDeviceAttributeMaxGridDimY, devid_);
  err_ = ld_->hipDeviceGetAttribute(&dz, hipDeviceAttributeMaxGridDimZ, devid_);
  err_ = ld_->hipDeviceGetAttribute(&ck, hipDeviceAttributeConcurrentKernels, devid_);
  max_work_group_size_ = tb;
  max_compute_units_ = mc;
  max_block_dims_[0] = bx;
  max_block_dims_[1] = by;
  max_block_dims_[2] = bz;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;

  _info("devid[%d] max_compute_units[%zu] max_work_group_size_[%zu] max_work_item_sizes[%zu,%zu,%zu] max_block_dims[%d,%d,%d] concurrent_kernels[%d]", devid_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], max_block_dims_[0], max_block_dims_[1], max_block_dims_[2], ck);

  char* path = kernel_path_;
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
    _trace("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return IRIS_SUCCESS;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  ld_->Lock();
  err_ = ld_->hipModuleLoad(&module_, path);
  ld_->Unlock();
  if (err_ != hipSuccess) {
    _hiperror(err_);
    _error("srclen[%zu] src\n%s", srclen, src);
    if (src) free(src);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceHIP::ResetMemory(BaseMem *mem, uint8_t reset_value) {
    err_ = ld_->hipMemset(mem->arch(this), reset_value, mem->size());
    _hiperror(err_);
    if (err_ != hipSuccess) {
       worker_->platform()->IncrementErrorCount();
       return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}

void DeviceHIP::RegisterPin(void *host, size_t size)
{
    ld_->hipHostRegister(host, size, hipHostRegisterDefault);
    //ld_->hipHostRegister(host, size, hipHostRegisterMapped);
}

int DeviceHIP::MemAlloc(void** mem, size_t size, bool reset) {
  void** hipmem = mem;
  err_ = ld_->hipMalloc(hipmem, size);
  _hiperror(err_);
  if (err_ != hipSuccess) {
     worker_->platform()->IncrementErrorCount();
     return IRIS_ERROR;
  }
  if (reset) err_ = ld_->hipMemset(*hipmem, 0, size);
  _hiperror(err_);
  return IRIS_SUCCESS;
}

int DeviceHIP::MemFree(void* mem) {
  void* hipmem = mem;
  err_ = ld_->hipFree(hipmem);
  _hiperror(err_);
  if (err_ != hipSuccess){
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
    hipCtx_t ctx;
    ld_->hipCtxGetCurrent(&ctx);
    _trace("HIP resetting context switch dev[%d][%s] self:%p thread:%p", devno_, name_, (void *)worker()->self(), (void *)worker()->thread());
    _trace("Resetting Context Switch: %p %p", ctx, ctx_);
    ld_->hipCtxSetCurrent(ctx_);
}
int DeviceHIP::MemD2D(Task *task, BaseMem *mem, void *dst, void *src, size_t size) {
  atleast_one_command_ = true;
  if (IsContextChangeRequired()) {
      _trace("HIP context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
#ifndef IRIS_SYNC_EXECUTION
  q_ = task->uid() % nqueues_; 
  err_ = ld_->hipMemcpyDtoDAsync(dst, src, size, streams_[q_]);
#else
  err_ = ld_->hipMemcpyDtoD(dst, src, size);
#endif
  _hiperror(err_);
  _trace("dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, size, q_);
  if (err_ != hipSuccess) return IRIS_ERROR;
  return IRIS_SUCCESS;
}
int DeviceHIP::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  atleast_one_command_ = true;
  if (IsContextChangeRequired()) {
      _trace("HIP context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  void* hipmem = mem->arch(this);
  if (dim == 2) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      size_t host_row_pitch = elem_size * host_sizes[0];
      void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
      err_ = ld_->hipMemcpy2D((char*) hipmem, dev_sizes[0]*elem_size, host_start,
              host_row_pitch, dev_sizes[0]*elem_size, dev_sizes[1], 
              hipMemcpyHostToDevice);
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
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], size, host, q_);
      err_ = ld_->hipMemcpyHtoD((char*) hipmem + off[0], host, size);
#if 0
      printf("H2D: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), hipmem+off[0], host, host, elem_size);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _trace("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, size, host);
  _hiperror(err_);
  if (err_ != hipSuccess){
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
  void* hipmem = mem->arch(this);
  if (dim == 2) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)hipmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      size_t host_row_pitch = elem_size * host_sizes[0];
      void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
      err_ = ld_->hipMemcpy2D((char*) host_start, host_sizes[0]*elem_size, hipmem,
              dev_sizes[0]*elem_size, dev_sizes[0]*elem_size, dev_sizes[1], 
              hipMemcpyDeviceToHost);
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
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, off[0], size, host, q_);
      err_ = ld_->hipMemcpyDtoH(host, (char*) hipmem + off[0], size);
#if 0
      printf("D2H: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), hipmem+off[0], host, host, elem_size);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _trace("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), hipmem, size, host);
  _hiperror(err_);
  //for(int i=0; i<size/4; i++) {
  //   printf("D:%d (%f) ", i, *(((float *)host)+i));
  //}
  //printf("\n");
  if (err_ != hipSuccess){
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
  if (IsContextChangeRequired()) {
      _trace("Changed Context for HIP resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->hipCtxSetCurrent(ctx_);
  }
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  int kernel_idx = -1;
  if (kernel->is_vendor_specific_kernel(devno_) && 
          host2hip_ld_->iris_host2hip_kernel_with_obj &&
        host2hip_ld_->iris_host2hip_kernel_with_obj(&kernel_idx, name)==IRIS_SUCCESS) {
      *kernel_bin = host2hip_ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_kernel) {
      *kernel_bin = host2hip_ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (native_kernel_not_exists()) {
      if (report_error) {
          _error("HIP kernel:%s not found !", name);
          worker_->platform()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  hipFunction_t* hipkernel = (hipFunction_t*) kernel_bin;
  err_ = ld_->hipModuleGetFunction(hipkernel, module_, name);
  if (report_error) _hiperror(err_);
  if (err_ != hipSuccess){
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
  err_ = ld_->hipModuleGetFunction(&hipkernel_off, module_, name_off);
  if (err_ == hipSuccess) {
      kernels_offs_.insert(std::pair<hipFunction_t, hipFunction_t>(*hipkernel, hipkernel_off));
  }
  else {
      return IRIS_ERROR;
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
  if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_setarg_with_obj)
      host2hip_ld_->iris_host2hip_setarg_with_obj(kernel->GetParamWrapperMemory(), kindex, size, value);
  else if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_setarg)
      host2hip_ld_->iris_host2hip_setarg(kindex, size, value);
  return IRIS_SUCCESS;
}

int DeviceHIP::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  mem->arch(this);
  void *dev_ptr = mem->arch(devno_);
  params_[idx] = mem->arch_ptr(devno_);
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_setmem_with_obj) 
      host2hip_ld_->iris_host2hip_setmem_with_obj(kernel->GetParamWrapperMemory(), kindex, dev_ptr);
  else if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_setmem) 
      host2hip_ld_->iris_host2hip_setmem(kindex, dev_ptr);
  return IRIS_SUCCESS;
}

void DeviceHIP::CheckVendorSpecificKernel(Kernel *kernel) {
  kernel->set_vendor_specific_kernel(devno_, false);
  if (host2hip_ld_->iris_host2hip_kernel_with_obj) {
      int status = host2hip_ld_->iris_host2hip_kernel_with_obj(kernel->GetParamWrapperMemory(), kernel->name());
      if (status == IRIS_SUCCESS &&
              host2hip_ld_->IsFunctionExists(kernel->name())) {
          kernel->set_vendor_specific_kernel(devno_, true);
      }
  }
  else if (host2hip_ld_->iris_host2hip_kernel) {
      if (host2hip_ld_->iris_host2hip_kernel(kernel->name()) == IRIS_SUCCESS &&
              host2hip_ld_->IsFunctionExists(kernel->name())) {
          kernel->set_vendor_specific_kernel(devno_, true);
      }
  }
  kernel->set_vendor_specific_kernel_check(devno_, true);
}

int DeviceHIP::KernelLaunchInit(Kernel* kernel) {
    return IRIS_SUCCESS;
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
  if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_launch_with_obj) {
      _trace("dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, q_);
      host2hip_ld_->SetKernelPtr(kernel->GetParamWrapperMemory(), kernel->name());
      int status = host2hip_ld_->iris_host2hip_launch_with_obj(kernel->GetParamWrapperMemory(), ordinal_, dim, off[0], gws[0]);
      err_ = ld_->hipDeviceSynchronize();
      _hiperror(err_);
      return status;
  }
  else if (kernel->is_vendor_specific_kernel(devno_) && host2hip_ld_->iris_host2hip_launch) {
      _trace("dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, q_);
      int status = host2hip_ld_->iris_host2hip_launch(dim, off[0], gws[0]);
      err_ = ld_->hipDeviceSynchronize();
      _hiperror(err_);
      return status;
  }
  _trace("native kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, q_);
  hipFunction_t func = (hipFunction_t) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    if (max_compute_units_ != 0) while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0] && max_block_dims_[0] !=0) block[0] >>= 1;
  }
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };

  if (off[0] != 0 || off[1] != 0 || off[2] != 0) {
    size_t blockOff_x = off[0] / block[0];
    size_t blockOff_y = off[1] / block[1];
    size_t blockOff_z = off[2] / block[2];
    params_[max_arg_idx_ + 1] = &blockOff_x;
    params_[max_arg_idx_ + 2] = &blockOff_y;
    params_[max_arg_idx_ + 3] = &blockOff_z;
    if (kernels_offs_.find(func) == kernels_offs_.end()) {
      _trace("off0[%lu] cannot find %s_with_offsets kernel. ignore offsets", off[0], kernel->name());
    } else {
      func = kernels_offs_[func];
      _trace("off0[%lu] running %s_with_offsets kernel.", off[0], kernel->name());
    }
  }

  _trace("dev[%d] kernel[%s:%s] dim[%d] grid[%d,%d,%d] block[%d,%d,%d] shared_mem_bytes[%u] q[%d]", devno_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, q_);
  err_ = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
  _hiperror(err_);
  if (err_ != hipSuccess){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
#ifdef IRIS_SYNC_EXECUTION
  err_ = ld_->hipDeviceSynchronize();
  _hiperror(err_);
  if (err_ != hipSuccess){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
#endif
  for (int i = 0; i < IRIS_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  return IRIS_SUCCESS;
}

int DeviceHIP::Synchronize() {
  if (! atleast_one_command_) return IRIS_SUCCESS;
  err_ = ld_->hipDeviceSynchronize();
  _hiperror(err_);
  if (err_ != hipSuccess){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHIP::AddCallback(Task* task) {
  task->Complete();
  return task->Ok();
}

} /* namespace rt */
} /* namespace iris */


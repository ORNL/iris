#include "DeviceHIP.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderHIP.h"
#include "Mem.h"
#include "Reduction.h"
#include "Task.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceHIP::DeviceHIP(LoaderHIP* ld, hipDevice_t dev, int ordinal, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  ordinal_ = ordinal;
  dev_ = dev;
  strcpy(vendor_, "Advanced Micro Devices");
  err_ = ld_->hipDeviceGetName(name_, sizeof(name_), dev_);
  _hiperror(err_);
  type_ = brisbane_amd;
  model_ = brisbane_hip;
  err_ = ld_->hipDriverGetVersion(&driver_version_);
  _hiperror(err_);
  sprintf(version_, "AMD HIP %d", driver_version_);
  _info("device[%d] platform[%d] vendor[%s] device[%s] ordinal[%d] type[%d] version[%s]", devno_, platform_, vendor_, name_, ordinal_, type_, version_);
}

DeviceHIP::~DeviceHIP() {

}
int DeviceHIP::Compile(char* src) {
  char cmd[256];
  memset(cmd, 0, 256);
  sprintf(cmd, "hipcc --genco %s -o %s", src, kernel_path_);
  if (system(cmd) == EXIT_SUCCESS) {
    _error("cmd[%s]", cmd);
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int DeviceHIP::Init() {
  int tb, mc, bx, by, bz, dx, dy, dz, ck, ae;
  err_ = ld_->hipSetDevice(ordinal_);
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
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return BRISBANE_OK;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  ld_->Lock();
  err_ = ld_->hipModuleLoad(&module_, path);
  ld_->Unlock();
  if (err_ != hipSuccess) {
    _hiperror(err_);
    _error("srclen[%zu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceHIP::MemAlloc(void** mem, size_t size) {
  void** hipmem = mem;
  err_ = ld_->hipMalloc(hipmem, size);
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceHIP::MemFree(void* mem) {
  void* hipmem = mem;
  err_ = ld_->hipFree(hipmem);
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceHIP::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  void* hipmem = mem->arch(this);
  err_ = ld_->hipMemcpyHtoD((char*) hipmem + off, host, size);
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceHIP::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  void* hipmem = mem->arch(this);
  err_ = ld_->hipMemcpyDtoH(host, (char*) hipmem + off, size);
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceHIP::KernelGet(void** kernel, const char* name) {
  hipFunction_t* hipkernel = (hipFunction_t*) kernel;
  err_ = ld_->hipModuleGetFunction(hipkernel, module_, name);
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;

  char name_off[256];
  memset(name_off, 0, sizeof(name_off));
  sprintf(name_off, "%s_with_offsets", name);
  hipFunction_t hipkernel_off;
  err_ = ld_->hipModuleGetFunction(&hipkernel_off, module_, name_off);
  if (err_ == hipSuccess)
    kernels_offs_.insert(std::pair<hipFunction_t, hipFunction_t>(*hipkernel, hipkernel_off));

  return BRISBANE_OK;
}

int DeviceHIP::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  if (value) params_[idx] = value;
  else {
    shared_mem_offs_[idx] = shared_mem_bytes_;
    params_[idx] = shared_mem_offs_ + idx;
    shared_mem_bytes_ += size;
  }
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceHIP::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  mem->arch(this);
  params_[idx] = mem->archs() + devno_;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceHIP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  hipFunction_t func = (hipFunction_t) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0]) block[0] >>= 1;
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

  _trace("kernel[%s] dim[%d] grid[%d,%d,%d] block[%d,%d,%d] shared_mem_bytes[%u] q[%d]", kernel->name(), dim, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, q_);
  err_ = ld_->hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
#ifdef BRISBANE_SYNC_EXECUTION
  err_ = ld_->hipDeviceSynchronize();
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
#endif
  for (int i = 0; i < BRISBANE_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  return BRISBANE_OK;
}

int DeviceHIP::Synchronize() {
  err_ = ld_->hipDeviceSynchronize();
  _hiperror(err_);
  if (err_ != hipSuccess) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceHIP::AddCallback(Task* task) {
  task->Complete();
  return task->Ok();
}

} /* namespace rt */
} /* namespace brisbane */


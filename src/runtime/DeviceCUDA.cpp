#include "DeviceCUDA.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderCUDA.h"
#include "Mem.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceCUDA::DeviceCUDA(LoaderCUDA* ld, CUdevice cudev, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  max_arg_idx_ = 0;
  ngarbage_ = 0;
  shared_mem_bytes_ = 0;
  dev_ = cudev;
  strcpy(vendor_, "NVIDIA Corporation");
  err_ = ld_->cuDeviceGetName(name_, sizeof(name_), dev_);
  _cuerror(err_);
  type_ = brisbane_nvidia;
  model_ = brisbane_cuda;
  err_ = ld_->cuDriverGetVersion(&driver_version_);
  _cuerror(err_);
  sprintf(version_, "NVIDIA CUDA %d", driver_version_);
  int tb, mc, bx, by, bz, dx, dy, dz, ck, ae;
  err_ = ld_->cuDeviceGetAttribute(&tb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev_);
  err_ = ld_->cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_);
  err_ = ld_->cuDeviceGetAttribute(&bx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev_);
  err_ = ld_->cuDeviceGetAttribute(&by, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev_);
  err_ = ld_->cuDeviceGetAttribute(&bz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev_);
  err_ = ld_->cuDeviceGetAttribute(&ck, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev_);
  err_ = ld_->cuDeviceGetAttribute(&ae, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev_);
  max_work_group_size_ = tb;
  max_compute_units_ = mc;
  max_block_dims_[0] = bx;
  max_block_dims_[1] = by;
  max_block_dims_[2] = bz;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%zu] max_work_group_size_[%zu] max_work_item_sizes[%zu,%zu,%zu] max_block_dims[%d,%d,%d] concurrent_kernels[%d] async_engines[%d]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], max_block_dims_[0], max_block_dims_[1], max_block_dims_[2], ck, ae);
}

DeviceCUDA::~DeviceCUDA() {
}

int DeviceCUDA::Compile(char* src) {
  char cmd[256];
  memset(cmd, 0, 256);
  sprintf(cmd, "nvcc -ptx %s -o %s", src, kernel_path_);
  if (system(cmd) != EXIT_SUCCESS) {
    _error("cmd[%s]", cmd);
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int DeviceCUDA::Init() {
  err_ = ld_->cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, dev_);
  _cuerror(err_);
  for (int i = 0; i < nqueues_; i++) {
    err_ = ld_->cuStreamCreate(streams_ + i, CU_STREAM_DEFAULT);
    _cuerror(err_);
  }

  char* path = kernel_path_;
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return BRISBANE_OK;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  err_ = ld_->cuModuleLoad(&module_, path);
  if (err_ != CUDA_SUCCESS) {
    _cuerror(err_);
    _error("srclen[%zu] src\n%s", srclen, src);
    if (src) free(src);
    return BRISBANE_ERR;
  }
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceCUDA::MemAlloc(void** mem, size_t size) {
  CUdeviceptr* cumem = (CUdeviceptr*) mem;
  err_ = ld_->cuMemAlloc(cumem, size);
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceCUDA::MemFree(void* mem) {
  CUdeviceptr cumem = (CUdeviceptr) mem;
  if (ngarbage_ >= BRISBANE_MAX_GABAGES) _error("ngarbage[%d]", ngarbage_);
  else garbage_[ngarbage_++] = cumem;
  /*
  _trace("dptr[%p]", cumem);
  err_ = ld_->cuMemFree(cumem);
  _cuerror(err_);
  */
  return BRISBANE_OK;
}

void DeviceCUDA::ClearGarbage() {
  if (ngarbage_ == 0) return;
  for (int i = 0; i < ngarbage_; i++) {
    CUdeviceptr cumem = garbage_[i];
    err_ = ld_->cuMemFree(cumem);
    _cuerror(err_);
  }
  ngarbage_ = 0;
}

int DeviceCUDA::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this);
  _trace("dev[%d][%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", devno_, name_, mem->uid(), cumem, off, size, host, q_);
#ifdef BRISBANE_SYNC_EXECUTION
  err_ = ld_->cuMemcpyHtoD(cumem + off, host, size);
#else
  err_ = ld_->cuMemcpyHtoDAsync(cumem + off, host, size, streams_[q_]);
#endif
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceCUDA::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this);
  _trace("dev[%d][%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", devno_, name_, mem->uid(), cumem, off, size, host, q_);
#ifdef BRISBANE_SYNC_EXECUTION
  err_ = ld_->cuMemcpyDtoH(host, cumem + off, size);
#else
  err_ = ld_->cuMemcpyDtoHAsync(host, cumem + off, size, streams_[q_]);
#endif
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelGet(void** kernel, const char* name) {
  CUfunction* cukernel = (CUfunction*) kernel;
  err_ = ld_->cuModuleGetFunction(cukernel, module_, name);
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;

  char name_off[256];
  memset(name_off, 0, sizeof(name_off));
  sprintf(name_off, "%s_with_offsets", name);
  CUfunction cukernel_off;
  err_ = ld_->cuModuleGetFunction(&cukernel_off, module_, name_off);
  if (err_ == CUDA_SUCCESS) {
    kernels_offs_.insert(std::pair<CUfunction, CUfunction>(*cukernel, cukernel_off));
  }

  return BRISBANE_OK;
}

int DeviceCUDA::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  if (value) params_[idx] = value;
  else {
    shared_mem_offs_[idx] = shared_mem_bytes_;
    params_[idx] = shared_mem_offs_ + idx;
    shared_mem_bytes_ += size;
  }
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  mem->arch(this);
  if (off) {
    *(mem->archs_off() + devno_) = (void*) ((CUdeviceptr) mem->arch(this) + off);
    params_[idx] = mem->archs_off() + devno_;
  } else params_[idx] = mem->archs() + devno_;
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  return BRISBANE_OK;
}

int DeviceCUDA::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  CUfunction cukernel = (CUfunction) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0]) block[0] >>= 1;
  }
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };
  size_t blockOff_x = off[0] / block[0];
  size_t blockOff_y = off[1] / block[1];
  size_t blockOff_z = off[2] / block[2];

  if (off[0] != 0 || off[1] != 0 || off[2] != 0) {
    params_[max_arg_idx_ + 1] = &blockOff_x;
    params_[max_arg_idx_ + 2] = &blockOff_y;
    params_[max_arg_idx_ + 3] = &blockOff_z;
    if (kernels_offs_.find(cukernel) == kernels_offs_.end()) {
      _trace("off0[%lu] cannot find %s_with_offsets kernel. ignore offsets", off[0], kernel->name());
    } else {
      cukernel = kernels_offs_[cukernel];
      _trace("off0[%lu] running %s_with_offsets kernel.", off[0], kernel->name());
    }
  }
  _trace("kernel[%s] dim[%d] grid[%d,%d,%d] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", kernel->name(), dim, grid[0], grid[1], grid[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, q_);
#ifdef BRISBANE_SYNC_EXECUTION
  err_ = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
//  err_ = ld_->cuStreamSynchronize(0);
#else
  err_ = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, streams_[q_], params_, NULL);
#endif
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;
  for (int i = 0; i < BRISBANE_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  return BRISBANE_OK;
}

int DeviceCUDA::Synchronize() {
  err_ = ld_->cuCtxSynchronize();
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceCUDA::AddCallback(Task* task) {
  err_ = ld_->cuStreamAddCallback(streams_[q_], DeviceCUDA::Callback, task, 0);
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceCUDA::Custom(int tag, char* params) {
  if (!cmd_handlers_.count(tag)) {
    _error("unknown tag[0x%x]", tag);
    return BRISBANE_ERR;
  }
  command_handler handler = cmd_handlers_[tag];
  handler(params, this);
  return BRISBANE_OK;
}

void DeviceCUDA::Callback(CUstream stream, CUresult status, void* data) {
  Task* task = (Task*) data;
  task->Complete();
}

void DeviceCUDA::TaskPre(Task* task) {
  ClearGarbage();
}

} /* namespace rt */
} /* namespace brisbane */


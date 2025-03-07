#include "DeviceHexagon.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderHexagon.h"
#include "BaseMem.h"
#include "Task.h"
#include "Utils.h"
#include "Worker.h"
#include <iris/hexagon/rpcmem.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

namespace iris {
namespace rt {

DeviceHexagon::DeviceHexagon(LoaderHexagon* ld, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  type_ = iris_hexagon;
  model_ = iris_hexagon;
  strcpy(name_, "Hexagon DSP");
  _info("device[%d] platform[%d] device[%s] type[%d]", devno_, platform_, name_, type_);
}

DeviceHexagon::~DeviceHexagon() {
  ld_->iris_hexagon_finalize();
}

int DeviceHexagon::Init() {
  ld_->iris_hexagon_init();
  return IRIS_SUCCESS;
}

int DeviceHexagon::ResetMemory(Task *task, Command *cmd, BaseMem *mem) {
    _error("Reset memory is not implemented yet !");
    return IRIS_ERROR;
}

int DeviceHexagon::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
  void** hxgmem = mem_addr;
  *hxgmem = ld_->iris_hexagon_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int) size);
  if (*hxgmem == 0) {
    _error("hxgmem[%p]", hxgmem);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (reset) {
    _error("Hexagon not supported with reset for size:%lu", size);
  }
  return IRIS_SUCCESS;
}

int DeviceHexagon::MemFree(BaseMem *mem, void* mem_addr) {
  void* hxgmem = mem_addr;
  if (hxgmem) ld_->iris_hexagon_rpcmem_free(hxgmem);
  return IRIS_SUCCESS;
}

int DeviceHexagon::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* hxgmem = mem->arch(this);
  if (dim == 2 || dim == 3) {
      Utils::MemCpy3D((uint8_t *)hxgmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
  }
  else {
      memcpy((char*) hxgmem, (uint8_t *)host + off[0]*elem_size, size);
  }
  return IRIS_SUCCESS;
}

int DeviceHexagon::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* hxgmem = mem->arch(this);
  if (dim == 2 || dim == 3) {
      Utils::MemCpy3D((uint8_t *)hxgmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
  }
  else {
      memcpy((uint8_t *)host + off[0]*elem_size, (char*) hxgmem, size);
  }
  return IRIS_SUCCESS;
}

int DeviceHexagon::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  return IRIS_SUCCESS;
}

int DeviceHexagon::KernelLaunchInit(Command *cmd, Kernel* kernel) {
  if (ld_->iris_hexagon_kernel_with_obj) {
      if (ld_->iris_hexagon_kernel_with_obj(
              kernel->GetParamWrapperMemory(), kernel->name())==IRIS_SUCCESS) {
          return IRIS_SUCCESS;
      }
  }
  return ld_->iris_hexagon_kernel(kernel->name());
}

int DeviceHexagon::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  if (ld_->iris_hexagon_setarg_with_obj)
      return ld_->iris_hexagon_setarg_with_obj(
              kernel->GetParamWrapperMemory(), kindex, size, value);
  return ld_->iris_hexagon_setarg(kindex, size, value);
}

int DeviceHexagon::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  void* hxgmem = mem->arch(this);
  if (ld_->iris_hexagon_setmem_with_obj)
      return ld_->iris_hexagon_setmem_with_obj(
              kernel->GetParamWrapperMemory(), kindex, hxgmem, (int) mem->size());
  return ld_->iris_hexagon_setmem(kindex, hxgmem, (int) mem->size());
}

int DeviceHexagon::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] off[%zu] gws[%zu]", kernel->name(), dim, off[0], gws[0]);
  if (ld_->iris_hexagon_launch_with_obj) {
      ld_->SetKernelPtr(kernel->GetParamWrapperMemory(), kernel->name());
      return ld_->iris_hexagon_launch_with_obj(NULL,
              kernel->GetParamWrapperMemory(), 0, dim, off[0], gws[0]);
  }
  return ld_->iris_hexagon_launch(dim, off[0], gws[0]);
}

int DeviceHexagon::Synchronize() {
  return IRIS_SUCCESS;
}

int DeviceHexagon::AddCallback(Task* task) {
  task->Complete();
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


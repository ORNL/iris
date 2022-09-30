#include "DeviceHexagon.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderHexagon.h"
#include "Mem.h"
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

int DeviceHexagon::MemAlloc(void** mem, size_t size) {
  void** hxgmem = mem;
  *hxgmem = ld_->iris_hexagon_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int) size);
  if (*hxgmem == 0) {
    _error("hxgmem[%p]", hxgmem);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceHexagon::MemFree(void* mem) {
  void* hxgmem = mem;
  if (hxgmem) ld_->iris_hexagon_rpcmem_free(hxgmem);
  return IRIS_SUCCESS;
}

int DeviceHexagon::MemH2D(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host) {
  void* hxgmem = mem->arch(this);
  if (dim == 2 || dim == 3) {
      Utils::MemCpy3D((uint8_t *)hxgmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
  }
  else {
      memcpy((char*) hxgmem + off[0], host, size);
  }
  return IRIS_SUCCESS;
}

int DeviceHexagon::MemD2H(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host) {
  void* hxgmem = mem->arch(this);
  if (dim == 2 || dim == 3) {
      Utils::MemCpy3D((uint8_t *)hxgmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
  }
  else {
      memcpy(host, (char*) hxgmem + off[0], size);
  }
  return IRIS_SUCCESS;
}

int DeviceHexagon::KernelGet(void** kernel, const char* name) {
  return IRIS_SUCCESS;
}

int DeviceHexagon::KernelLaunchInit(Kernel* kernel) {
  return ld_->iris_hexagon_kernel(kernel->name());
}

int DeviceHexagon::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  return ld_->iris_hexagon_setarg(idx, size, value);
}

int DeviceHexagon::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  void* hxgmem = mem->arch(this);
  return ld_->iris_hexagon_setmem(idx, hxgmem, (int) mem->size());
}

int DeviceHexagon::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] off[%zu] gws[%zu]", kernel->name(), dim, off[0], gws[0]);
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


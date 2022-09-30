#include "Kernel.h"
#include "Debug.h"
#include "Device.h"
#include "History.h"
#include "Mem.h"
#include "Worker.h"
#include <string.h>

namespace iris {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
  size_t len = strlen(name);
  strncpy(name_, name, len);
  name_[len] = 0;
  platform_ = platform;
  history_ = new History(this);
  for (int i = 0; i < IRIS_MAX_NDEVS; i++) {
    archs_[i] = NULL;
    archs_devs_[i] = NULL;
  }
}

Kernel::~Kernel() {
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I)
    delete I->second;
}

int Kernel::SetArg(int idx, size_t size, void* value) {
  KernelArg* arg = new KernelArg;
  arg->size = size;
  if (value) memcpy(arg->value, value, size);
  arg->mem = NULL;
  arg->off = 0ULL;
  args_[idx] = arg;
  return IRIS_SUCCESS;
}

int Kernel::SetMem(int idx, Mem* mem, size_t off, int mode) {
  KernelArg* arg = new KernelArg;
  if(!mem) {
    _error("no mem[%p] for the kernel parameter %d", mem, idx);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  arg->mem = mem;
  arg->off = off;
  arg->mode = mode;
  arg->mem_off = 0;
  arg->mem_size = mem->size();
  args_[idx] = arg;
  return IRIS_SUCCESS;
}

KernelArg* Kernel::ExportArgs() {
  KernelArg* new_args = new KernelArg[args_.size()];
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I) {
    KernelArg* new_arg = new_args + I->first;
    KernelArg* arg = I->second;
    if (arg->mem) {
      new_arg->mem = arg->mem;
      new_arg->off = arg->off;
      new_arg->mode = arg->mode;
      new_arg->mem_off = arg->mem_off;
      new_arg->mem_size = arg->mem_size;
    } else {
      new_arg->size = arg->size; 
      memcpy(new_arg->value, arg->value, arg->size);
      new_arg->mem = NULL;
      new_arg->off = 0ULL;
    }
  }
  return new_args;
}

void* Kernel::arch(Device* dev) {
  int devno = dev->devno();
  if (archs_[devno] == NULL) dev->KernelGet(archs_ + devno, (const char*) name_);
  return archs_[devno];
}

} /* namespace rt */
} /* namespace iris */


#include "Kernel.h"
#include "Debug.h"
#include "Device.h"
#include "History.h"
#include <string.h>

namespace brisbane {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
  size_t len = strlen(name);
  strncpy(name_, name, len);
  name_[len] = 0;
  platform_ = platform;
  history_ = new History(this);
  for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) {
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
  arg->off = 0;
  args_[idx] = arg;
  return BRISBANE_OK;
}

int Kernel::SetMem(int idx, Mem* mem, size_t off, int mode) {
  KernelArg* arg = new KernelArg;
  arg->mem = mem;
  arg->off = off;
  arg->mode = mode;
  args_[idx] = arg;
  return BRISBANE_OK;
}

std::map<int, KernelArg*>* Kernel::ExportArgs() {
  std::map<int, KernelArg*>* new_args = new std::map<int, KernelArg*>();
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I) {
    KernelArg* arg = I->second;
    KernelArg* new_arg = new KernelArg;
    if (arg->mem) {
      new_arg->mem = arg->mem;
      new_arg->off = arg->off;
      new_arg->mode = arg->mode;
    } else {
      new_arg->size = arg->size; 
      memcpy(new_arg->value, arg->value, arg->size);
      new_arg->mem = NULL;
      new_arg->off = 0;
    }
    (*new_args)[I->first] = new_arg;
  }
  return new_args;
}

void* Kernel::arch(Device* dev) {
  int devno = dev->devno();
  if (archs_[devno] == NULL) dev->KernelGet(archs_ + devno, (const char*) name_);
  return archs_[devno];
}

} /* namespace rt */
} /* namespace brisbane */


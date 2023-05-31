#include "Kernel.h"
#include "Debug.h"
#include "Device.h"
#include "History.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include <string.h>

using namespace std;
namespace iris {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
  size_t len = strlen(name);
  strncpy(name_, name, len);
  strcpy(task_name_, name);
  name_[len] = 0;
  profile_data_transfers_ = false;
  platform_ = platform;
  history_ = platform->CreateHistory(name);
  for (int i = 0; i < IRIS_MAX_NDEVS; i++) {
    archs_[i] = NULL;
    archs_devs_[i] = NULL;
    set_vendor_specific_kernel(i, false);
    vendor_specific_kernel_check_flag_[i] = false;
  }
  set_object_track(Platform::GetPlatform()->kernel_track_ptr());
  Platform::GetPlatform()->kernel_track().TrackObject(this, uid());
}

Kernel::~Kernel() {
  Platform::GetPlatform()->kernel_track().UntrackObject(this, uid());
  data_mems_in_.clear();
  data_mems_in_order_.clear();
  data_mem_regions_in_.clear();
  history_ = nullptr;
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I)
    delete I->second;
  _trace(" kernel:%lu:%s is destroyed", uid(), name_);
}

int Kernel::set_order(int *order) {
  for(int index = 0; index < data_mems_in_.size(); index++) {
    data_mems_in_order_.push_back(order[index]);
  }
  return IRIS_SUCCESS;
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

int Kernel::SetMem(int idx, BaseMem* mem, size_t off, int mode) {
  KernelArg* arg = new KernelArg;
  if(!mem) {
    _error("no mem[%p] for the kernel parameter %d", mem, idx);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (mem->GetMemHandlerType() == IRIS_DMEM) add_dmem((DataMem *)mem, idx, mode);
  if (mem->GetMemHandlerType() == IRIS_DMEM_REGION) add_dmem_region((DataMemRegion *)mem, idx, mode);
  arg->mem = mem;
  arg->off = off;
  arg->mode = mode;
  arg->mem_off = 0;
  arg->mem_size = mem->size();
  args_[idx] = arg;
  return IRIS_SUCCESS;
}

void Kernel::add_dmem(DataMem *mem, int idx, int mode)
{
    if (mode == iris_r) {
        data_mems_in_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
    }
    else if (mode == iris_w)  {
        data_mems_out_.insert(make_pair(idx, mem));
    }
    else if (mode == iris_rw)  {
        data_mems_in_.insert(make_pair(idx, mem));
        data_mems_out_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
    }
}
void  Kernel::add_dmem_region(DataMemRegion *mem, int idx, int mode)
{
    if (mode == iris_r) {
        data_mem_regions_in_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
    }
    else if (mode == iris_w)  {
        data_mem_regions_out_.insert(make_pair(idx, mem));
    }
    else if (mode == iris_rw)  {
        data_mem_regions_in_.insert(make_pair(idx, mem));
        data_mem_regions_out_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
    }
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

int Kernel::isSupported(Device* dev) {
  int devno = dev->devno();
  if (archs_[devno] != NULL) return true;
  int result = dev->KernelGet(this, archs_ + devno, (const char*) name_, false);
  return (result == IRIS_SUCCESS);
}

void* Kernel::arch(Device* dev, bool report_error) {
  int devno = dev->devno();
  if (archs_[devno] == NULL) dev->KernelGet(this, archs_ + devno, (const char*) name_, report_error);
  return archs_[devno];
}

} /* namespace rt */
} /* namespace iris */


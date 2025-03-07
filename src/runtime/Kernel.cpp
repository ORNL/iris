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
  name_ = string(name);
  //size_t len = strlen(name);
  //strncpy(name_, name, len);
  task_name_ = name_;
  task_ = NULL;
  n_mems_ = 0;
  //name_[len] = 0;
  profile_data_transfers_ = false;
  platform_ = platform;
  Retain();
  history_ = platform->CreateHistory(name);
  async_data_.Init(this, -1);
  for (size_t i = 0; i < IRIS_MAX_NDEVS; i++) {
    archs_[i] = NULL;
    archs_devs_[i] = NULL;
    set_vendor_specific_kernel(i, false);
    set_default_kernel(i, false);
    vendor_specific_kernel_check_flag_[i] = false;
  }
  host_if_data_ = NULL;
  set_object_track(Platform::GetPlatform()->kernel_track_ptr());
  Platform::GetPlatform()->kernel_track().TrackObject(this, uid());
}

Kernel::~Kernel() {
  //Platform::GetPlatform()->kernel_track().UntrackObject(this, uid());
  data_mems_in_.clear();
  data_mems_in_order_.clear();
  data_mem_regions_in_.clear();
  mem_track_.clear();
  out_mem_track_.clear();
  in_mem_track_.clear();
  data_mems_in_.clear();
  data_mems_out_.clear();
  data_mem_regions_in_.clear();
  data_mem_regions_out_.clear();
  history_ = nullptr;
  if (host_if_data_ != NULL) free(host_if_data_); 
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I)
    delete I->second;
  _trace(" kernel:%lu:%s is destroyed", uid(), name());
}

int Kernel::set_order(int *order) {
  for(size_t index = 0; index < data_mems_in_.size(); index++) {
    data_mems_in_order_.push_back(order[index]);
  }
  return IRIS_SUCCESS;
}

int Kernel::SetArg(int idx, size_t size, void* value) {
  KernelArg* arg = new KernelArg;
  arg->size = size & 0xFFFF;
  arg->data_type = size & 0xFFFF0000;
  if (value) memcpy(arg->value, value, size);
  arg->mem = NULL;
  arg->off = 0ULL;
  args_[idx] = arg;
  return IRIS_SUCCESS;
}

int Kernel::SetMem(int idx, BaseMem* mem, size_t off, int mode) {
  KernelArg* arg = new KernelArg;
  _debug2("IDX:%d", idx);
  if(!mem) {
    _error("no mem[%p] for the kernel parameter %d", mem, idx);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (mem->GetMemHandlerType() == IRIS_MEM) add_mem((Mem*)mem, idx, mode);
  if (mem->GetMemHandlerType() == IRIS_DMEM) add_dmem((DataMem *)mem, idx, mode);
  if (mem->GetMemHandlerType() == IRIS_DMEM_REGION) add_dmem_region((DataMemRegion *)mem, idx, mode);
  arg->mem = mem;
  arg->off = off;
  arg->mode = mode;
  arg->mem_index = n_mems_; n_mems_++;
  arg->mem_off = 0;
  arg->mem_size = mem->size();
  args_[idx] = arg;
  return IRIS_SUCCESS;
}

void Kernel::add_mem(Mem *mem, int idx, int mode)
{
    if (mode == iris_r) {
        in_mem_track_.insert(make_pair((BaseMem *)mem, idx));
    }
    else if (mode == iris_w)  {
        out_mem_track_.insert(make_pair((BaseMem *)mem, idx));
    }
    else if (mode == iris_rw)  {
        in_mem_track_.insert(make_pair((BaseMem *)mem, idx));
        out_mem_track_.insert(make_pair((BaseMem *)mem, idx));
    }
    mem_track_.insert(make_pair((BaseMem *)mem, idx));
}
void Kernel::add_dmem(DataMem *mem, int idx, int mode)
{
    if (mode == iris_r) {
        data_mems_in_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
        in_mem_track_.insert(make_pair(mem, idx));
    }
    else if (mode == iris_w)  {
        data_mems_out_.insert(make_pair(idx, mem));
        out_mem_track_.insert(make_pair(mem, idx));
    }
    else if (mode == iris_rw)  {
        data_mems_in_.insert(make_pair(idx, mem));
        data_mems_out_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
        in_mem_track_.insert(make_pair(mem, idx));
        out_mem_track_.insert(make_pair(mem, idx));
    }
    mem_track_.insert(make_pair(mem, idx));
}
void  Kernel::add_dmem_region(DataMemRegion *mem, int idx, int mode)
{
    if (mode == iris_r) {
        data_mem_regions_in_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
        in_mem_track_.insert(make_pair(mem, idx));
    }
    else if (mode == iris_w)  {
        data_mem_regions_out_.insert(make_pair(idx, mem));
        out_mem_track_.insert(make_pair(mem, idx));
    }
    else if (mode == iris_rw)  {
        data_mem_regions_in_.insert(make_pair(idx, mem));
        data_mem_regions_out_.insert(make_pair(idx, mem));
        all_data_mems_in_.push_back(mem);
        in_mem_track_.insert(make_pair(mem, idx));
        out_mem_track_.insert(make_pair(mem, idx));
    }
    mem_track_.insert(make_pair(mem, idx));
}
/*
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
void Kernel::replace_with_shadow_dmem(DataMem *mem, int idx, int mode){
    auto it = data_mem_out_.find(idx);
    if (it == data_mem_out_.end()) 
        _error("Data mem does not exists in the map:%s\n",name());

    if (mode == iris_r)  {
        _error("Reading only varible does not need shadow:%s\n",name());
    }
    else if (mode == iris_w) {
        data_mem_out_.at(idx) = mem;
    }
    else if (mode == iris_rw)  {
        data_mem_out_.at(idx) = mem;
        data_mem_in_.at(idx) = mem;
    }
}
#endif
#endif
*/
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
      new_arg->mem_index = arg->mem_index;
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
  int result = dev->KernelGet(this, archs_ + devno, name(), false);
  return (result == IRIS_SUCCESS);
}

void* Kernel::arch(Device* dev, bool report_error) {
  int devno = dev->devno();
  if (archs_[devno] == NULL) dev->KernelGet(this, archs_ + devno, name(), report_error);
  return archs_[devno];
}

} /* namespace rt */
} /* namespace iris */


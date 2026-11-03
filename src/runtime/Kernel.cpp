#include "Kernel.h"
#include "Task.h"
#include "Debug.h"
#include "Device.h"
#include "History.h"
#include "Command.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include <string.h>
#include <vector>
#include <algorithm> 

using namespace std;
namespace iris {
namespace rt {

Kernel::Kernel(const char* name, Platform* platform) {
  name_ = string(name);
  //size_t len = strlen(name);
  //strncpy(name_, name, len);
  cmd_ = NULL;
  task_name_ = name_;
  shared_mem_size_ = 0;
  task_ = NULL;
  n_mems_ = 0;
  gws_dynamic_from_dmem_ = NULL;
  //name_[len] = 0;
  enable_julia_if_ = false;
  enable_python_if_ = false;
  julia_kernel_type_ = iris_julia_native;
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
  mem_track_.clear();
  out_mem_track_.clear();
  in_mem_track_.clear();
  data_mems_out_.clear();
  data_mem_regions_in_.clear();
  data_mem_regions_out_.clear();
  history_ = nullptr;
  if (host_if_data_ != NULL) free(host_if_data_); 
  for (std::map<int, KernelArg*>::iterator I = args_.begin(), E = args_.end(); I != E; ++I)
    delete I->second;
  _trace(" kernel:%lu:%s is destroyed", uid(), name());
  //_printf(" kernel:%lu:%s is destroyed", uid(), name());
}

int Kernel::set_order(int *order) {
  for(size_t index = 0; index < data_mems_in_.size(); index++) {
    data_mems_in_order_.push_back(order[index]);
  }
  return IRIS_SUCCESS;
}

void Kernel::set_gws_dynamic_from_dmem(DataMem *mem)
{
    gws_dynamic_from_dmem_ = mem;
}

void Kernel::update_arg(int index, void *val)
{
    if (cmd_ != NULL)  {
        cmd_->update_arg(index, val);
    }
    else {
        _error("This case is not handled");
    }
}
void Kernel::update_arg_core(int index, void *val)
{
    if (args_.find(index) != args_.end()) {
        KernelArg *arg = args_[index];
        if (arg != NULL) 
            memcpy(arg->value, val, arg->size);
    }
}
void Kernel::update_mem(int index, BaseMem *mem)
{
    if (cmd_ != NULL)  {
        cmd_->update_mem(index, mem);
    }
    else {
        _error("This case is not handled");
    }
}
void Kernel::update_map(int index, BaseMem *mem, std::map<BaseMem*, int> & mem2index)
{
    bool found = false;
    for (auto it = mem2index.begin(); it != mem2index.end();) {
        if (it->second == index) {
            auto next = it; ++next;   // save next before erase
            int val = it->second;     // copy or move value
            mem2index.erase(it);
            it = next;
            found = true;
        } else {
            ++it;
        }
    }
    if (found)
        mem2index.insert(std::make_pair(mem, index));
}
void Kernel::update_mem_core(int index, BaseMem *mem)
{
    BaseMem *old_mem = NULL;
    if (args_.find(index) != args_.end()) {
        KernelArg *arg = args_[index];
        //string kname = name();
        //    _printf(" kname:%s index:%d arg:%p mem:%lu:%s\n", kname, index, 
        //            arg, mem->uid(), mem->name());
        if (arg != NULL) {
            old_mem = arg->mem;
            arg->mem = mem;
        }
    }
    update_map(index, mem, in_mem_track_);
    update_map(index, mem, out_mem_track_);
    update_map(index, mem, mem_track_);
    if (data_mems_in_.find(index) != data_mems_in_.end()) {
        old_mem = data_mems_in_[index];
        data_mems_in_[index] = (DataMem *)mem;
    }
    if (data_mems_out_.find(index) != data_mems_out_.end()) {
        data_mems_out_[index] = (DataMem *)mem;
    }
    if (data_mem_regions_in_.find(index) != data_mem_regions_in_.end()) {
        old_mem = data_mem_regions_in_[index];
        data_mem_regions_in_[index] = (DataMemRegion *)mem;
    }
    if (data_mem_regions_out_.find(index) != data_mem_regions_out_.end()) {
        data_mem_regions_out_[index] = (DataMemRegion *)mem;
    }
    auto it = std::find(all_data_mems_in_.begin(), all_data_mems_in_.end(), old_mem);
    if (it != all_data_mems_in_.end())
        *it = mem;
    if (old_mem != NULL && gws_dynamic_from_dmem_ != NULL 
            && old_mem == gws_dynamic_from_dmem_) {
        gws_dynamic_from_dmem_ = (DataMem*)mem;
    }
}
void Kernel::set_task(Task *task) { 
    task_ = task; 
    if (task->enable_julia_if()) { 
        set_enable_julia_if();
        set_julia_kernel_type(task->julia_kernel_type());
    }
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


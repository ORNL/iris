#ifndef IRIS_SRC_RT_KERNEL_H
#define IRIS_SRC_RT_KERNEL_H

#include "Config.h"
#include "Retainable.h"
#include "Platform.h"
#include "History.h"
#include "AsyncData.h"
#include <stdint.h>
#include <map>
#include <vector>
#include <memory>
#include <string>

#define KERNEL_ARGS_MEM_SIZE 8*128
using namespace std;
namespace iris {
namespace rt {

class History;
class MemHistory;
class DataMem;
class DataMemRegion;
typedef struct _KernelArg {
  size_t size;
  char value[1024];
  BaseMem* mem;
  size_t mem_off;
  size_t mem_size;
  size_t off;
  int mode;
  int data_type;
  int mem_index;
  int proactive;
  void proactive_enabled() { proactive = true; }
  void proactive_disabled() { proactive = false; }
} KernelArg;

using KernelAsyncData = AsyncData<Kernel>;
class Kernel: public Retainable<struct _iris_kernel, Kernel> {
public:
  Kernel(const char* name, Platform* platform);
  virtual ~Kernel();

  int SetArg(int idx, size_t size, void* value);
  int SetMem(int idx, BaseMem* mem, size_t off, int mode);
  KernelArg* ExportArgs();
  void* GetJuliadata() { return host_if_data_; }
  void CreateJuliadata(size_t size) { host_if_data_ = malloc(size); }
  void* GetFFIdata() { return host_if_data_; }
  void CreateFFIdata(size_t size) { host_if_data_ = malloc(size); }
  void* GetParamWrapperMemory() { return (void *)param_wrapper_mem_; }

  const char* name() { return name_.c_str(); }
  void set_profile_data_transfers(bool flag=true) { profile_data_transfers_ = flag; }
  bool is_profile_data_transfers() { return profile_data_transfers_; }
  void AddInDataObjectProfile(DataObjectProfile hist) { in_dataobject_profiles.push_back(hist); }
  void ClearMemInProfile() { in_dataobject_profiles.clear(); }
  vector<DataObjectProfile> & in_mem_profiles() { return in_dataobject_profiles; }
  bool vendor_specific_kernel_check_flag(int devno) { return vendor_specific_kernel_check_flag_[devno]; }
  void set_vendor_specific_kernel_check(int devno, bool flag=true) { vendor_specific_kernel_check_flag_[devno] = flag; }
  bool is_vendor_specific_kernel(int devno) { return is_vendor_specific_kernel_[devno]; }
  void set_vendor_specific_kernel(int devno, bool flag=true) { is_vendor_specific_kernel_[devno] = flag; }
  bool is_default_kernel(int devno) { return is_default_kernel_[devno]; }
  void set_default_kernel(int devno, bool flag=true) { is_default_kernel_[devno] = flag; }
  void set_task_name(const char *name) { task_name_= string(name); }
  void set_task(Task *task) { task_ = task; }
  Task *task() { return task_; }
  const char *get_task_name() { return task_name_.c_str(); }
  Platform* platform() { return platform_; }
  shared_ptr<History> history() { return history_; }
  map<BaseMem*, int> & in_mems() { return in_mem_track_; }
  map<BaseMem*, int> & out_mems() { return out_mem_track_; }
  map<BaseMem*, int> & mems() { return mem_track_; }
  map<int, DataMem *> & data_mems_in() { return data_mems_in_; }
  map<int, DataMem *> & data_mems_out() { return data_mems_out_; }
  map<int, DataMemRegion *> & data_mem_regions_in() { return data_mem_regions_in_; }
  map<int, DataMemRegion *> & data_mem_regions_out() { return data_mem_regions_out_; }
  vector<int> & data_mems_in_order() { return data_mems_in_order_; }
  vector<BaseMem*> & all_data_mems_in() { return all_data_mems_in_; }
  int get_mem_karg_index(BaseMem *mem) { 
    return (mem_track_.find(mem) != mem_track_.end()) ? mem_track_[mem] : -1;
  }
  KernelArg *get_mem_karg(BaseMem *mem) { 
    return (mem_track_.find(mem) != mem_track_.end()) ? karg(mem_track_[mem]) : NULL;
  }
  KernelArg *get_in_mem_karg(BaseMem *mem) { 
    return (in_mem_track_.find(mem) != in_mem_track_.end()) ? karg(in_mem_track_[mem]) : NULL;
  }
  bool is_in_mem_exist(BaseMem *mem) { 
    return in_mem_track_.find(mem) != in_mem_track_.end();
  }
  void** archs() { return archs_; }
  size_t nargs() { return args_.size(); }
  KernelArg *karg(int index) { return args_[index]; }
  void* arch(Device* dev, bool report_error=true);
  int set_order(int *order);
  int isSupported(Device* dev);
  void add_mem(Mem *mem, int idx, int mode);
  void add_dmem(DataMem *mem, int idx, int mode);
  void add_dmem_region(DataMemRegion *mem, int idx, int mode);
  void **GetCompletionEventPtr(bool new_entry=false) { return async_data_.GetCompletionEventPtr(new_entry); }
  void *GetCompletionEvent() { return async_data_.GetCompletionEvent(); }
/*
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
  void replace_with_shadow_dmem(DataMem *mem, int idx, int mode);
#endif
#endif
*/

private:
  int n_mems_;
  std::string name_;
  std::string task_name_;
  Task *task_;
  std::map<int, KernelArg*> args_;
  void* archs_[IRIS_MAX_NDEVS];
  Device* archs_devs_[IRIS_MAX_NDEVS];
  void *host_if_data_;
  uint8_t param_wrapper_mem_[KERNEL_ARGS_MEM_SIZE];
  Platform* platform_;
  shared_ptr<History> history_;
  bool is_vendor_specific_kernel_[IRIS_MAX_NDEVS];
  bool is_default_kernel_[IRIS_MAX_NDEVS];
  bool vendor_specific_kernel_check_flag_[IRIS_MAX_NDEVS];
  bool profile_data_transfers_;
  vector<DataObjectProfile>       in_dataobject_profiles;
  vector<int> data_mems_in_order_;
  vector<BaseMem *> all_data_mems_in_;
  std::map<BaseMem *, int> in_mem_track_;
  std::map<BaseMem *, int> out_mem_track_;
  std::map<BaseMem *, int> mem_track_;
  std::map<int, DataMem *> data_mems_in_;
  std::map<int, DataMemRegion *> data_mem_regions_in_;
  std::map<int, DataMem *> data_mems_out_;
  std::map<int, DataMemRegion *> data_mem_regions_out_;
  KernelAsyncData async_data_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_KERNEL_H */

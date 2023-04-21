#ifndef IRIS_SRC_RT_KERNEL_H
#define IRIS_SRC_RT_KERNEL_H

#include "Config.h"
#include "Retainable.h"
#include "Platform.h"
#include "History.h"
#include <stdint.h>
#include <map>
#include <vector>
#include <memory>

using namespace std;
namespace iris {
namespace rt {

class History;
class MemHistory;
class DataMem;
class DataMemRegion;
typedef struct _KernelArg {
  size_t size;
  char value[256];
  BaseMem* mem;
  size_t mem_off;
  size_t mem_size;
  size_t off;
  int mode;
} KernelArg;

class Kernel: public Retainable<struct _iris_kernel, Kernel> {
public:
  Kernel(const char* name, Platform* platform);
  virtual ~Kernel();

  int SetArg(int idx, size_t size, void* value);
  int SetMem(int idx, BaseMem* mem, size_t off, int mode);
  KernelArg* ExportArgs();
  void* GetParamWrapperMemory() { return (void *)param_wrapper_mem_; }

  char* name() { return name_; }
  void set_profile_data_transfers(bool flag=true) { profile_data_transfers_ = flag; }
  bool is_profile_data_transfers() { return profile_data_transfers_; }
  void AddInDataObjectProfile(DataObjectProfile hist) { in_dataobject_profiles.push_back(hist); }
  void ClearMemInProfile() { in_dataobject_profiles.clear(); }
  vector<DataObjectProfile> & in_mem_profiles() { return in_dataobject_profiles; }
  bool vendor_specific_kernel_check_flag(int devno) { return vendor_specific_kernel_check_flag_[devno]; }
  void set_vendor_specific_kernel_check(int devno, bool flag=true) { vendor_specific_kernel_check_flag_[devno] = flag; }
  bool is_vendor_specific_kernel(int devno) { return is_vendor_specific_kernel_[devno]; }
  void set_vendor_specific_kernel(int devno, bool flag=true) { is_vendor_specific_kernel_[devno] = flag; }
  void set_task_name(const char *name) { strcpy(task_name_, name); }
  char *get_task_name() { return task_name_; }
  Platform* platform() { return platform_; }
  shared_ptr<History> history() { return history_; }
  map<int, DataMem *> & data_mems_in() { return data_mems_in_; }
  map<int, DataMem *> & data_mems_out() { return data_mems_out_; }
  map<int, DataMemRegion *> & data_mem_regions_in() { return data_mem_regions_in_; }
  map<int, DataMemRegion *> & data_mem_regions_out() { return data_mem_regions_out_; }
  vector<int> & data_mems_in_order() { return data_mems_in_order_; }
  vector<BaseMem*> & all_data_mems_in() { return all_data_mems_in_; }
  void** archs() { return archs_; }
  size_t nargs() { return args_.size(); }
  void* arch(Device* dev, bool report_error=true);
  int set_order(int *order);
  int isSupported(Device* dev);
  void add_dmem(DataMem *mem, int idx, int mode);
  void add_dmem_region(DataMemRegion *mem, int idx, int mode);

private:
  char name_[256];
  char task_name_[256];
  std::map<int, KernelArg*> args_;
  void* archs_[IRIS_MAX_NDEVS];
  Device* archs_devs_[IRIS_MAX_NDEVS];
  uint8_t param_wrapper_mem_[8*128];
  Platform* platform_;
  shared_ptr<History> history_;
  bool is_vendor_specific_kernel_[IRIS_MAX_NDEVS];
  bool vendor_specific_kernel_check_flag_[IRIS_MAX_NDEVS];
  bool profile_data_transfers_;
  vector<DataObjectProfile>       in_dataobject_profiles;
  vector<int> data_mems_in_order_;
  vector<BaseMem *> all_data_mems_in_;
  std::map<int, DataMem *> data_mems_in_;
  std::map<int, DataMemRegion *> data_mem_regions_in_;
  std::map<int, DataMem *> data_mems_out_;
  std::map<int, DataMemRegion *> data_mem_regions_out_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_KERNEL_H */

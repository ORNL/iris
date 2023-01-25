#ifndef IRIS_SRC_RT_KERNEL_H
#define IRIS_SRC_RT_KERNEL_H

#include "Config.h"
#include "Retainable.h"
#include "Platform.h"
#include <stdint.h>
#include <map>
#include <vector>

using namespace std;
namespace iris {
namespace rt {

class History;
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
  bool is_vendor_specific_kernel() { return is_vendor_specific_kernel_; }
  void set_vendor_specific_kernel(bool flag=true) { is_vendor_specific_kernel_ = flag; }
  void set_task_name(char *name) { strcpy(task_name_, name); }
  char *get_task_name() { return task_name_; }
  Platform* platform() { return platform_; }
  History* history() { return history_; }
  map<int, DataMem *> & data_mems_in() { return data_mems_in_; }
  map<int, DataMem *> & data_mems_out() { return data_mems_out_; }
  map<int, DataMemRegion *> & data_mem_regions_in() { return data_mem_regions_in_; }
  map<int, DataMemRegion *> & data_mem_regions_out() { return data_mem_regions_out_; }
  void** archs() { return archs_; }
  void* arch(Device* dev, bool report_error=true);
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
  History* history_;
  bool is_vendor_specific_kernel_;
  std::map<int, DataMem *> data_mems_in_;
  std::map<int, DataMemRegion *> data_mem_regions_in_;
  std::map<int, DataMem *> data_mems_out_;
  std::map<int, DataMemRegion *> data_mem_regions_out_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_KERNEL_H */

#ifndef IRIS_SRC_RT_COMMAND_H
#define IRIS_SRC_RT_COMMAND_H

#include <iris/iris_poly_types.h>
#include "BaseMem.h"
#include "Kernel.h"
#include <stddef.h>

#define IRIS_CMD_NOP            0x1000
#define IRIS_CMD_INIT           0x1001
#define IRIS_CMD_KERNEL         0x1002
#define IRIS_CMD_MALLOC         0x1003
#define IRIS_CMD_H2D            0x1004
#define IRIS_CMD_H2DNP          0x1005
#define IRIS_CMD_D2H            0x1006
#define IRIS_CMD_MAP            0x1007
#define IRIS_CMD_MAP_TO         0x1008
#define IRIS_CMD_MAP_FROM       0x1009
#define IRIS_CMD_RELEASE_MEM    0x100a
#define IRIS_CMD_HOST           0x100b
#define IRIS_CMD_MEM_FLUSH      0x100c
#define IRIS_CMD_CUSTOM         0x100d
#define IRIS_CMD_RESET_INPUT    0x100e
#define IRIS_CMD_H2BROADCAST    0x100f
#define IRIS_CMD_D2D            0x1010

#define IRIS_CMD_KERNEL_NARGS_MAX   16

namespace iris {
namespace rt {

class DataMem;
class Mem;
class DataMem;
class Task;

class Command {
public:
  Command();
  Command(Task* task, int type);
  ~Command();

  void Set(Task* task, int type);

  int type() { return type_; }
  bool type_init() { return type_ == IRIS_CMD_INIT; }
  bool type_h2d() { return type_ == IRIS_CMD_H2D; }
  bool type_h2broadcast() { return type_ == IRIS_CMD_H2BROADCAST; }
  bool type_h2dnp() { return type_ == IRIS_CMD_H2DNP; }
  bool type_d2h() { return type_ == IRIS_CMD_D2H; }
  bool type_memflush() { return type_ == IRIS_CMD_MEM_FLUSH; }
  bool type_kernel() { return type_ == IRIS_CMD_KERNEL; }
  size_t size() { return size_; }
  void* host() { return host_; }
  int dim() { return dim_; }
  size_t* off() { return off_; }
  size_t off(int i) { return off_[i]; }
  size_t ws() { return gws_[0] * gws_[1] * gws_[2]; }
  size_t* gws() { return gws_; }
  size_t gws(int i) { return gws_[i]; }
  size_t* lws() { return lws_; }
  size_t lws(int i) { return lws_[i]; }
  size_t elem_size() { return elem_size_; }
  Kernel* kernel() { return kernel_; }
  KernelArg* kernel_args() { return kernel_args_; }
  KernelArg* kernel_arg(int i) { return kernel_args_ + i; }
  int kernel_nargs() { return kernel_nargs_; }
  DataMem* datamem() { return (DataMem *)mem_; }
  Mem* mem() { return (Mem *)mem_; }
  Task* task() { return task_; }
  bool last() { return last_; }
  void set_last() { last_ = true; }
  bool exclusive() { return exclusive_; }
  iris_poly_mem* polymems() { return polymems_; }
  int npolymems() { return npolymems_; }
  int tag() { return tag_; }
  void set_selector_kernel(iris_selector_kernel func, void* params, size_t params_size);
  void* selector_kernel_params() { return selector_kernel_params_; }
  iris_selector_kernel selector_kernel() { return selector_kernel_; }
  bool is_internal_memory_transfer() { return internal_memory_transfer_;}
  void set_internal_memory_transfer() { internal_memory_transfer_ = true;}
  iris_host_task func() { return func_; }
  void* func_params() { return func_params_; }
  char* params() { return params_; }
  void set_params_map(int *pmap);
  int *get_params_map() { return params_map_; }
  char* type_name() { return type_name_; }
  const char* name() { return name_; }
  void set_name(const char* name) { name_ = name; }
  uint8_t reset_value() { return reset_value_; }
  double SetTime(double t, bool incr=true);
  double time() { return time_; }
  void set_time_start(double d) { time_start_ = d; }
  void set_time_end(double d) { time_end_ = d; }
  double time_start() { return time_start_; }
  double time_end() { return time_end_; }
  double time_duration() { return time_end_-time_start_; }
  int get_access_index() { return access_index_; }
  int src_dev() { return src_dev_; }
  void set_src_dev(int devno) { src_dev_ = devno; }
private:
  void Clear(bool init);

private:
  int type_;
  int src_dev_;
  size_t size_;
  void* host_;
  int *params_map_;
  int dim_;
  size_t off_[3];
  size_t gws_[3];
  size_t lws_[3];
  size_t elem_size_;
  Kernel* kernel_;
  BaseMem* mem_;
  Task* task_;
  Platform* platform_;
  double time_;
  double time_start_;
  double time_end_;
  bool last_;
  bool exclusive_;
  KernelArg* kernel_args_;
  int kernel_nargs_;
  int kernel_nargs_max_;
  iris_poly_mem* polymems_;
  int npolymems_;
  int tag_;
  iris_selector_kernel selector_kernel_;
  void* selector_kernel_params_;
  iris_host_task func_;
  void* func_params_;
  char* params_;
  char* type_name_;
  const char* name_;
  int access_index_;
  bool internal_memory_transfer_;
  uint8_t reset_value_;

public:
  static Command* Create(Task* task, int type);
  static Command* CreateInit(Task* task);
  static Command* CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  static Command* CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges);
  static Command* CreateKernelPolyMem(Task* task, Command* cmd, size_t* off, size_t* gws, iris_poly_mem* polymems, int npolymems);
  static Command* CreateMalloc(Task* task, Mem* mem);
  static Command* CreateMemResetInput(Task* task, BaseMem *mem, uint8_t reset_value=0);
  static Command* CreateMemIn(Task* task, DataMem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host);
  static Command* CreateH2Broadcast(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateH2Broadcast(Task* task, Mem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host);
  static Command* CreateD2D(Task* task, Mem* mem, size_t off, size_t size, void* host, int src_dev);
  static Command* CreateH2D(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateH2D(Task* task, Mem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host);
  static Command* CreateH2DNP(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateD2H(Task* task, Mem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host);
  static Command* CreateD2H(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateMemFlushOut(Task* task, DataMem* mem);
  static Command* CreateMap(Task* task, void* host, size_t size);
  static Command* CreateMapTo(Task* task, void* host);
  static Command* CreateMapFrom(Task* task, void* host);
  static Command* CreateReleaseMem(Task* task, Mem* mem);
  static Command* CreateHost(Task* task, iris_host_task func, void* params);
  static Command* CreateCustom(Task* task, int tag, void* params, size_t params_size);
  static void Release(Command* cmd);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_COMMAND_H */


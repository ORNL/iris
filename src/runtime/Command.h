#ifndef BRISBANE_SRC_RT_COMMAND_H
#define BRISBANE_SRC_RT_COMMAND_H

#include <iris/brisbane_poly_types.h>
#include "Kernel.h"
#include <stddef.h>

#define BRISBANE_CMD_NOP            0x1000
#define BRISBANE_CMD_INIT           0x1001
#define BRISBANE_CMD_KERNEL         0x1002
#define BRISBANE_CMD_MALLOC         0x1003
#define BRISBANE_CMD_H2D            0x1004
#define BRISBANE_CMD_H2DNP          0x1005
#define BRISBANE_CMD_D2H            0x1006
#define BRISBANE_CMD_MAP            0x1007
#define BRISBANE_CMD_MAP_TO         0x1008
#define BRISBANE_CMD_MAP_FROM       0x1009
#define BRISBANE_CMD_RELEASE_MEM    0x100a
#define BRISBANE_CMD_HOST           0x100b
#define BRISBANE_CMD_CUSTOM         0x100c

#define BRISBANE_CMD_KERNEL_NARGS_MAX   16

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
#define IRIS_CMD_CUSTOM         0x100c

#define IRIS_CMD_KERNEL_NARGS_MAX   16

namespace brisbane {
namespace rt {

class Mem;
class Task;

class Command {
public:
  Command();
  Command(Task* task, int type);
  ~Command();

  void Set(Task* task, int type);

  int type() { return type_; }
  bool type_init() { return type_ == BRISBANE_CMD_INIT; }
  bool type_h2d() { return type_ == BRISBANE_CMD_H2D; }
  bool type_h2dnp() { return type_ == BRISBANE_CMD_H2DNP; }
  bool type_d2h() { return type_ == BRISBANE_CMD_D2H; }
  bool type_kernel() { return type_ == BRISBANE_CMD_KERNEL; }
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
  Kernel* kernel() { return kernel_; }
  KernelArg* kernel_args() { return kernel_args_; }
  KernelArg* kernel_arg(int i) { return kernel_args_ + i; }
  int kernel_nargs() { return kernel_nargs_; }
  Mem* mem() { return mem_; }
  Task* task() { return task_; }
  bool last() { return last_; }
  void set_last() { last_ = true; }
  bool exclusive() { return exclusive_; }
  brisbane_poly_mem* polymems() { return polymems_; }
  int npolymems() { return npolymems_; }
  int tag() { return tag_; }
  void set_selector_kernel(brisbane_selector_kernel func, void* params, size_t params_size);
  void* selector_kernel_params() { return selector_kernel_params_; }
  brisbane_selector_kernel selector_kernel() { return selector_kernel_; }
  brisbane_host_task func() { return func_; }
  void* func_params() { return func_params_; }
  char* params() { return params_; }
  double time() { return time_; }
  double SetTime(double t);

private:
  void Clear(bool init);

private:
  int type_;
  size_t size_;
  void* host_;
  int dim_;
  size_t off_[3];
  size_t gws_[3];
  size_t lws_[3];
  Kernel* kernel_;
  Mem* mem_;
  Task* task_;
  Platform* platform_;
  double time_;
  bool last_;
  bool exclusive_;
  KernelArg* kernel_args_;
  int kernel_nargs_;
  int kernel_nargs_max_;
  brisbane_poly_mem* polymems_;
  int npolymems_;
  int tag_;
  brisbane_selector_kernel selector_kernel_;
  void* selector_kernel_params_;
  brisbane_host_task func_;
  void* func_params_;
  char* params_;

public:
  static Command* Create(Task* task, int type);
  static Command* CreateInit(Task* task);
  static Command* CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  static Command* CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges);
  static Command* CreateKernelPolyMem(Task* task, Command* cmd, size_t* off, size_t* gws, brisbane_poly_mem* polymems, int npolymems);
  static Command* CreateMalloc(Task* task, Mem* mem);
  static Command* CreateH2D(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateH2DNP(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateD2H(Task* task, Mem* mem, size_t off, size_t size, void* host);
  static Command* CreateMap(Task* task, void* host, size_t size);
  static Command* CreateMapTo(Task* task, void* host);
  static Command* CreateMapFrom(Task* task, void* host);
  static Command* CreateReleaseMem(Task* task, Mem* mem);
  static Command* CreateHost(Task* task, brisbane_host_task func, void* params);
  static Command* CreateCustom(Task* task, int tag, void* params, size_t params_size);
  static void Release(Command* cmd);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_COMMAND_H */


#ifndef IRIS_SRC_RT_DEVICE_LEVEL_ZERO_H
#define IRIS_SRC_RT_DEVICE_LEVEL_ZERO_H

#include "Device.h"
#include "LoaderLevelZero.h"

namespace iris {
namespace rt {

class DeviceLevelZero : public Device {
public:
  DeviceLevelZero(LoaderLevelZero* ld, ze_device_handle_t zedev, ze_context_handle_t zectx, ze_driver_handle_t zedriver, int devno, int platform);
  ~DeviceLevelZero();

  int Compile(char* src, const char *out=NULL, const char *flags=NULL);
  int Init();
  int ResetMemory(Task *task, Command *cmd, BaseMem *mem);
  int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset=false);
  int MemFree(BaseMem *mem, void* mem_addr);
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *tile_sizes,  size_t *full_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *tile_sizes,  size_t *full_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);

  const char* kernel_src() { return "KERNEL_SRC_SPV"; }
  const char* kernel_bin() { return "KERNEL_BIN_SPV"; }

private:
  LoaderLevelZero* ld_;
  ze_driver_handle_t zedriver_;
  ze_device_handle_t zedev_;
  ze_context_handle_t zectx_;
  ze_command_queue_handle_t zecmq_;
  ze_command_list_handle_t zecml_;
  ze_module_handle_t zemod_;
  ze_event_pool_handle_t zeevtpool_;
  ze_result_t err_;

  size_t align_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_LEVEL_ZERO_H */


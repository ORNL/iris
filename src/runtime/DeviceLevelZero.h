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

  int Compile(char* src);
  int Init();
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t *off, size_t *tile_sizes,  size_t *full_sizes, size_t elem_size, int dim, size_t size, void* host);
  int MemD2H(Mem* mem, size_t *off, size_t *tile_sizes,  size_t *full_sizes, size_t elem_size, int dim, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
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


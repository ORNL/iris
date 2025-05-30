#ifndef IRIS_SRC_RT_DEVICE_HEXAGON_H
#define IRIS_SRC_RT_DEVICE_HEXAGON_H

#include "Device.h"
#include "LoaderHexagon.h"

namespace iris {
namespace rt {

class DeviceHexagon : public Device {
public:
  DeviceHexagon(LoaderHexagon* ld, int devno, int platform);
  ~DeviceHexagon();

  int Init();
  int ResetMemory(Task *task, Command *cmd, BaseMem *mem);
  int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset=false);
  int MemFree(BaseMem *mem, void* mem_addr);
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  int KernelLaunchInit(Command *cmd, Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  bool SupportJIT() { return false; }

private:
  LoaderHexagon* ld_;

  //float* Z;
  //float* X;
  //float* Y;
  //float A;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_HEXAGON_H */


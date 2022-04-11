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
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host);
  int MemD2H(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelLaunchInit(Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  bool SupportJIT() { return false; }

private:
  LoaderHexagon* ld_;

  float* Z;
  float* X;
  float* Y;
  float A;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_HEXAGON_H */


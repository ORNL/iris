#ifndef IRIS_SRC_RT_DEVICE_OPENMP_H
#define IRIS_SRC_RT_DEVICE_OPENMP_H

#include "Device.h"
#include "LoaderOpenMP.h"

namespace iris {
namespace rt {

class DeviceOpenMP : public Device {
public:
  DeviceOpenMP(LoaderOpenMP* ld, int devno, int platform);
  ~DeviceOpenMP();

  int Init();
  void TaskPre(Task* task);
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t off, size_t size, void* host);
  int MemD2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelLaunchInit(Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  bool SupportJIT() { return false; }

private:
  LoaderOpenMP* ld_;
  int GetProcessorNameIntel(char* cpuinfo);
  int GetProcessorNamePower(char* cpuinfo);
  int GetProcessorNameAMD(char* cpuinfo);
  int GetProcessorNameARM(char* cpuinfo);
  int GetProcessorNameQualcomm(char* cpuinfo);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_OPENMP_H */


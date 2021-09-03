#ifndef BRISBANE_SRC_RT_DEVICE_OPENCL_H
#define BRISBANE_SRC_RT_DEVICE_OPENCL_H

#include "Device.h"
#include "LoaderOpenCL.h"
#include <string>

namespace brisbane {
namespace rt {

class DeviceOpenCL : public Device {
public:
  DeviceOpenCL(LoaderOpenCL* ld, cl_device_id cldev, cl_context clctx, int devno, int platform);
  ~DeviceOpenCL();

  int Init();
  int BuildProgram(char* path);
  int MemAlloc(void** mem, size_t size);
  int MemFree(void* mem);
  int MemH2D(Mem* mem, size_t off, size_t size, void* host);
  int MemD2H(Mem* mem, size_t off, size_t size, void* host);
  int KernelGet(void** kernel, const char* name);
  int KernelSetArg(Kernel* kernel, int idx, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  int RecreateContext();

private:
  int CreateProgram(const char* suffix, char** src, size_t* srclen);

private:
  LoaderOpenCL* ld_;
  cl_device_id cldev_;
  cl_context clctx_;
  cl_command_queue clcmdq_;
  cl_program clprog_;
  cl_device_type cltype_;
  cl_bool compiler_available_;
  cl_int err_;
  std::string fpga_bin_suffix_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEVICE_OPENCL_H */


#include "DeviceOpenMP.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderOpenMP.h"
#include "Mem.h"
#include "Task.h"
#include "Utils.h"
#include <dlfcn.h>
#include <stdlib.h>

namespace brisbane {
namespace rt {

DeviceOpenMP::DeviceOpenMP(LoaderOpenMP* ld, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  type_ = brisbane_cpu;
  model_ = brisbane_openmp;
  FILE* fd = fopen("/proc/cpuinfo", "rb");
  char* arg = 0;
  size_t size = 0;
  while (getdelim(&arg, &size, 0, fd) != -1) {
    if (GetProcessorNameIntel(arg) == BRISBANE_OK) break;
    if (GetProcessorNamePower(arg) == BRISBANE_OK) break;
    if (GetProcessorNameAMD(arg) == BRISBANE_OK) break;
    if (GetProcessorNameARM(arg) == BRISBANE_OK) break;
    if (GetProcessorNameQualcomm(arg) == BRISBANE_OK) break;
    strcpy(name_, "Unknown CPU"); break;
  }
  free(arg);
  fclose(fd);
  _info("device[%d] platform[%d] device[%s] type[%d]", devno_, platform_, name_, type_);
}

DeviceOpenMP::~DeviceOpenMP() {
  ld_->brisbane_openmp_finalize();
}

int DeviceOpenMP::GetProcessorNameIntel(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return BRISBANE_ERR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, "GHz");
  if (!c3) return BRISBANE_ERR;
  strncpy(name_, c2, c3 - c2 + 3);
  return BRISBANE_OK;
}

int DeviceOpenMP::GetProcessorNamePower(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "cpu\t\t: ");
  if (!c1) return BRISBANE_ERR;
  char* c2 = c1 + strlen("cpu\t\t: ");
  char* c3 = strstr(c2, "clock");
  if (!c3) return BRISBANE_ERR;
  strncpy(name_, c2, c3 - c2 - 1);
  return BRISBANE_OK;
}

int DeviceOpenMP::GetProcessorNameAMD(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return BRISBANE_ERR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, "\n");
  if (!c3) return BRISBANE_ERR;
  strncpy(name_, c2, c3 - c2);
  return BRISBANE_OK;
}

int DeviceOpenMP::GetProcessorNameARM(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return BRISBANE_ERR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, ")");
  if (!c3) return BRISBANE_ERR;
  strncpy(name_, c2, c3 - c2 + 1);
  return BRISBANE_OK;
}

int DeviceOpenMP::GetProcessorNameQualcomm(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, ": ");
  if (!c1) return BRISBANE_ERR;
  char* c2 = c1 + strlen(": ");
  char* c3 = strstr(c2, ")");
  if (!c3) return BRISBANE_ERR;
  strncpy(name_, c2, c3 - c2 + 1);
  return BRISBANE_OK;
}

int DeviceOpenMP::Init() {
  ld_->brisbane_openmp_init();
  return BRISBANE_OK;
}

int DeviceOpenMP::MemAlloc(void** mem, size_t size) {
  void** mpmem = mem;
  if (posix_memalign(mpmem, 0x1000, size) != 0) {
    _error("%s", "posix_memalign");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int DeviceOpenMP::MemFree(void* mem) {
  void* mpmem = mem;
  if (mpmem) free(mpmem);
  return BRISBANE_OK;
}

int DeviceOpenMP::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  void* mpmem = mem->arch(this);
  memcpy((char*) mpmem + off, host, size);
  return BRISBANE_OK;
}

int DeviceOpenMP::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  void* mpmem = mem->arch(this);
  memcpy(host, (char*) mpmem + off, size);
  return BRISBANE_OK;
}

int DeviceOpenMP::KernelGet(void** kernel, const char* name) {
  return BRISBANE_OK;
}

int DeviceOpenMP::KernelLaunchInit(Kernel* kernel) {
  return ld_->brisbane_openmp_kernel(kernel->name());
}

int DeviceOpenMP::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  return ld_->brisbane_openmp_setarg(idx, size, value);
}

int DeviceOpenMP::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  void* mpmem = (char*) mem->arch(this) + off;
  return ld_->brisbane_openmp_setmem(idx, mpmem);
}

int DeviceOpenMP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] off[%lu] gws[%lu]", kernel->name(), dim, off[0], gws[0]);
  return ld_->brisbane_openmp_launch(dim, off[0], gws[0]);
}

int DeviceOpenMP::Synchronize() {
  return BRISBANE_OK;
}

int DeviceOpenMP::AddCallback(Task* task) {
  task->Complete();
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


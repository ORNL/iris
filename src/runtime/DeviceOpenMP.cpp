#include "DeviceOpenMP.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderOpenMP.h"
#include "Mem.h"
#include "Task.h"
#include "Utils.h"
#include <dlfcn.h>
#include <stdlib.h>

namespace iris {
namespace rt {

DeviceOpenMP::DeviceOpenMP(LoaderOpenMP* ld, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  type_ = iris_cpu;
  model_ = iris_openmp;
  FILE* fd = fopen("/proc/cpuinfo", "rb");
  char* arg = 0;
  size_t size = 0;
  while (getdelim(&arg, &size, 0, fd) != -1) {
    if (GetProcessorNameIntel(arg) == IRIS_OK) break;
    if (GetProcessorNamePower(arg) == IRIS_OK) break;
    if (GetProcessorNameAMD(arg) == IRIS_OK) break;
    if (GetProcessorNameARM(arg) == IRIS_OK) break;
    if (GetProcessorNameQualcomm(arg) == IRIS_OK) break;
    strcpy(name_, "Unknown CPU"); break;
  }
  free(arg);
  fclose(fd);
  _info("device[%d] platform[%d] device[%s] type[%d]", devno_, platform_, name_, type_);
}

DeviceOpenMP::~DeviceOpenMP() {
  ld_->iris_openmp_finalize();
}

void DeviceOpenMP::TaskPre(Task *task) {
    if (!is_shared_memory_buffers()) return;
    for (int i = 0; i < task->ncmds(); i++) {
        Command* cmd = task->cmd(i);
        if (hook_command_pre_) hook_command_pre_(cmd);
        switch (cmd->type()) {
            case IRIS_CMD_D2H:          
                {
                    Mem* mem = cmd->mem();
                    void* host = cmd->host();
                    mem->arch(this, host);
                    break;
                }
            default: break;
        }
        if (hook_command_post_) hook_command_post_(cmd);
#ifndef IRIS_SYNC_EXECUTION
        if (cmd->last()) AddCallback(task);
#endif
    }
}

int DeviceOpenMP::GetProcessorNameIntel(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return IRIS_ERR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, "GHz");
  if (!c3) return IRIS_ERR;
  strncpy(name_, c2, c3 - c2 + 3);
  return IRIS_OK;
}

int DeviceOpenMP::GetProcessorNamePower(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "cpu\t\t: ");
  if (!c1) return IRIS_ERR;
  char* c2 = c1 + strlen("cpu\t\t: ");
  char* c3 = strstr(c2, "clock");
  if (!c3) return IRIS_ERR;
  strncpy(name_, c2, c3 - c2 - 1);
  return IRIS_OK;
}

int DeviceOpenMP::GetProcessorNameAMD(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return IRIS_ERR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, "\n");
  if (!c3) return IRIS_ERR;
  strncpy(name_, c2, c3 - c2);
  return IRIS_OK;
}

int DeviceOpenMP::GetProcessorNameARM(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return IRIS_ERR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, ")");
  if (!c3) return IRIS_ERR;
  strncpy(name_, c2, c3 - c2 + 1);
  return IRIS_OK;
}

int DeviceOpenMP::GetProcessorNameQualcomm(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, ": ");
  if (!c1) return IRIS_ERR;
  char* c2 = c1 + strlen(": ");
  char* c3 = strstr(c2, ")");
  if (!c3) return IRIS_ERR;
  strncpy(name_, c2, c3 - c2 + 1);
  return IRIS_OK;
}

int DeviceOpenMP::Init() {
  ld_->iris_openmp_init();
  return IRIS_OK;
}

int DeviceOpenMP::MemAlloc(void** mem, size_t size) {
  void** mpmem = mem;
  if (posix_memalign(mpmem, 0x1000, size) != 0) {
    _error("%s", "posix_memalign");
    return IRIS_ERR;
  }
  return IRIS_OK;
}

int DeviceOpenMP::MemFree(void* mem) {
  void* mpmem = mem;
  if (mpmem && !is_shared_memory_buffers()) free(mpmem);
  return IRIS_OK;
}

int DeviceOpenMP::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  void* mpmem = mem->arch(this, host);
  if (!is_shared_memory_buffers())
      memcpy((char*) mpmem + off, host, size);
  return IRIS_OK;
}

int DeviceOpenMP::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  void* mpmem = mem->arch(this, host);
  if (!is_shared_memory_buffers())
      memcpy(host, (char*) mpmem + off, size);
  return IRIS_OK;
}

int DeviceOpenMP::KernelGet(void** kernel, const char* name) {
  return IRIS_OK;
}

int DeviceOpenMP::KernelLaunchInit(Kernel* kernel) {
  return ld_->iris_openmp_kernel(kernel->name());
}

int DeviceOpenMP::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  return ld_->iris_openmp_setarg(idx, size, value);
}

int DeviceOpenMP::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  void* mpmem = (char*) mem->arch(this) + off;
  return ld_->iris_openmp_setmem(idx, mpmem);
}

int DeviceOpenMP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("dev[%d] kernel[%s] dim[%d] off[%lu] gws[%lu]", devno_, kernel->name(), dim, off[0], gws[0]);
  return ld_->iris_openmp_launch(dim, off[0], gws[0]);
}

int DeviceOpenMP::Synchronize() {
  return IRIS_OK;
}

int DeviceOpenMP::AddCallback(Task* task) {
  task->Complete();
  return IRIS_OK;
}

} /* namespace rt */
} /* namespace iris */


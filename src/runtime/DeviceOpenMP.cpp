#include "DeviceOpenMP.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderOpenMP.h"
#include "BaseMem.h"
#include "Mem.h"
#include "Task.h"
#include "Utils.h"
#include "Worker.h"
#include <dlfcn.h>
#include <stdlib.h>

namespace iris {
namespace rt {

DeviceOpenMP::DeviceOpenMP(LoaderOpenMP* ld, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  type_ = iris_cpu;
  model_ = iris_openmp;
  can_share_host_memory_ = true;
  FILE* fd = fopen("/proc/cpuinfo", "rb");
  char* arg = 0;
  size_t size = 0;
  while (getdelim(&arg, &size, 0, fd) != -1) {
    if (GetProcessorNameIntel(arg) == IRIS_SUCCESS) break;
    if (GetProcessorNamePower(arg) == IRIS_SUCCESS) break;
    if (GetProcessorNameAMD(arg) == IRIS_SUCCESS) break;
    if (GetProcessorNameARM(arg) == IRIS_SUCCESS) break;
    if (GetProcessorNameQualcomm(arg) == IRIS_SUCCESS) break;
    strcpy(name_, "Unknown CPU"); break;
  }
  free(arg);
  fclose(fd);
  _info("device[%d] platform[%d] device[%s] type[%d]", devno_, platform_, name_, type_);
}

DeviceOpenMP::~DeviceOpenMP() {
  if (ld_->iris_openmp_finalize)
      ld_->iris_openmp_finalize();
  if (ld_->iris_openmp_finalize_handles)
      ld_->iris_openmp_finalize_handles(devno_);
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
  if (!c1) return IRIS_ERROR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, "GHz");
  if (!c3) return IRIS_ERROR;
  strncpy(name_, c2, c3 - c2 + 3);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::GetProcessorNamePower(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "cpu\t\t: ");
  if (!c1) return IRIS_ERROR;
  char* c2 = c1 + strlen("cpu\t\t: ");
  char* c3 = strstr(c2, "clock");
  if (!c3) return IRIS_ERROR;
  strncpy(name_, c2, c3 - c2 - 1);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::GetProcessorNameAMD(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return IRIS_ERROR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, "\n");
  if (!c3) return IRIS_ERROR;
  strncpy(name_, c2, c3 - c2);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::GetProcessorNameARM(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, "model name\t: ");
  if (!c1) return IRIS_ERROR;
  char* c2 = c1 + strlen("model name\t: ");
  char* c3 = strstr(c2, ")");
  if (!c3) return IRIS_ERROR;
  strncpy(name_, c2, c3 - c2 + 1);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::GetProcessorNameQualcomm(char* cpuinfo) {
  char* c1 = strstr(cpuinfo, ": ");
  if (!c1) return IRIS_ERROR;
  char* c2 = c1 + strlen(": ");
  char* c3 = strstr(c2, ")");
  if (!c3) return IRIS_ERROR;
  strncpy(name_, c2, c3 - c2 + 1);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::Init() {
  if (ld_->iris_openmp_init) 
      ld_->iris_openmp_init();
  if (ld_->iris_openmp_init_handles) 
      ld_->iris_openmp_init_handles(devno_);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::ResetMemory(BaseMem *mem, uint8_t reset_value)
{
    memset(mem->arch(this), reset_value, mem->size());
    return IRIS_SUCCESS;
}

int DeviceOpenMP::MemAlloc(void** mem, size_t size, bool reset) {
  void** mpmem = mem;
  if (posix_memalign(mpmem, 0x1000, size) != 0) {
    _error("%s", "posix_memalign");
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (reset) memset(*mpmem, 0, size);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::MemFree(void* mem) {
  void* mpmem = mem;
  if (mpmem && !is_shared_memory_buffers()) free(mpmem);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* mpmem = mem->arch(this, host);
  if (!is_shared_memory_buffers()) {
      if (dim == 2 || dim == 3) {
          _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), mpmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
          Utils::MemCpy3D((uint8_t *)mpmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
#if 0
          printf("H2D: ");
          float *A = (float *) host;
          for(int i=0; i<dev_sizes[1]; i++) {
             int ai = off[1] + i;
             for(int j=0; j<dev_sizes[0]; j++) {
                 int aj = off[0] + j;
                 printf("%10.1lf ", A[ai*host_sizes[0]+aj]);
             }
          }
          printf("\n");
#endif
      }
      else {
          _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), mpmem, off[0], size, host);
          memcpy((char*) mpmem + off[0], host, size);
#if 0
          printf("H2D: ");
          float *A = (float *) host;
          for(int i=0; i<size/4; i++) {
              printf("%10.1lf ", A[i]);
          }
          printf("\n");
#endif
      }
  }
  return IRIS_SUCCESS;
}

int DeviceOpenMP::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* mpmem = mem->arch(this, host);
  if (!is_shared_memory_buffers()) {
      if (dim == 2 || dim == 3) {
          _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), mpmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
          Utils::MemCpy3D((uint8_t *)mpmem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, false);
#if 0
          printf("D2H: ");
          float *A = (float *) host;
          for(int i=0; i<dev_sizes[1]; i++) {
             int ai = off[1] + i;
             for(int j=0; j<dev_sizes[0]; j++) {
                 int aj = off[0] + j;
                 printf("%10.1lf ", A[ai*host_sizes[0]+aj]);
             }
          }
          printf("\n");
#endif
      }
      else {
          _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), mpmem, off[0], size, host);
          memcpy(host, (char*) mpmem + off[0], size);
#if 0
          printf("D2H: ");
          float *A = (float *) host;
          for(int i=0; i<size/4; i++) {
              printf("%10.1lf ", A[i]);
          }
          printf("\n");
#endif
      }
  }
  return IRIS_SUCCESS;
}

int DeviceOpenMP::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  int status = IRIS_ERROR;
  if (!ld_->IsFunctionExists(name)) {
      if (report_error) {
          _error("Missing function for OpenMP kernel:%s", kernel->name());
          worker_->platform()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  *kernel_bin = ld_->GetFunctionPtr(name);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::KernelLaunchInit(Kernel* kernel) {
  //c_string_array data = ld_->iris_get_kernel_names();
  int status=IRIS_ERROR;
  if (ld_->iris_openmp_kernel_with_obj)
      status = ld_->iris_openmp_kernel_with_obj(kernel->GetParamWrapperMemory(), kernel->name());
  else if (ld_->iris_openmp_kernel)
      status = ld_->iris_openmp_kernel(kernel->name());
  if (status == IRIS_ERROR) {
      _error("Missing iris_openmp_kernel/iris_openmp_kernel_with_obj for OpenMP kernel:%s", kernel->name());
      worker_->platform()->IncrementErrorCount();
  }
  return status;
}

int DeviceOpenMP::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  if (ld_->iris_openmp_setarg_with_obj)
      return ld_->iris_openmp_setarg_with_obj(kernel->GetParamWrapperMemory(), kindex, size, value);
  if (ld_->iris_openmp_setarg)
      return ld_->iris_openmp_setarg(kindex, size, value);
  _error("Missing host iris_openmp_setarg/iris_openmp_setarg_with_obj function for OpenMP kernel:%s", kernel->name());
  worker_->platform()->IncrementErrorCount();
  return IRIS_ERROR;
}

int DeviceOpenMP::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  void* mpmem = (char*) mem->arch(this) + off;
  if (ld_->iris_openmp_setmem_with_obj)
    return ld_->iris_openmp_setmem_with_obj(kernel->GetParamWrapperMemory(), kindex, mpmem);
  if (ld_->iris_openmp_setmem)
    return ld_->iris_openmp_setmem(kindex, mpmem);
  _error("Missing host iris_openmp_setmem/iris_openmp_setmem_with_obj function for OpenMP kernel:%s", kernel->name());
  worker_->platform()->IncrementErrorCount();
  return IRIS_ERROR;
}

int DeviceOpenMP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("dev[%d] kernel[%s:%s] dim[%d] off[%lu] gws[%lu]", devno_, kernel->name(), kernel->get_task_name(), dim, off[0], gws[0]);
  ld_->SetKernelPtr(kernel->GetParamWrapperMemory(), kernel->name());
  if (ld_->iris_openmp_launch_with_obj)
    return ld_->iris_openmp_launch_with_obj(kernel->GetParamWrapperMemory(), 0, dim, off[0], gws[0]);
  if (ld_->iris_openmp_launch)
    return ld_->iris_openmp_launch(dim, off[0], gws[0]);
  _error("Missing host iris_openmp_launch/iris_openmp_launch_with_obj function for OpenMP kernel:%s", kernel->name());
  worker_->platform()->IncrementErrorCount();
  return IRIS_ERROR;
}

int DeviceOpenMP::Synchronize() {
  return IRIS_SUCCESS;
}

int DeviceOpenMP::AddCallback(Task* task) {
  task->Complete();
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


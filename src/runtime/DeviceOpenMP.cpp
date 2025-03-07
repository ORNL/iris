#include "DeviceOpenMP.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderOpenMP.h"
#include "BaseMem.h"
#include "DataMem.h"
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
  ld_->finalize(devno());
  if (julia_if_ != NULL) julia_if_->finalize(devno());
}

void DeviceOpenMP::TaskPre(Task *task) {
}

int DeviceOpenMP::Compile(char* src, const char *out, const char *flags) {
  char default_comp_flags[] = "";
  char cmd[1024];
  memset(cmd, 0, 256);
  if (flags == NULL) 
      flags = default_comp_flags;
  if (out == NULL) 
      out = kernel_path();
  sprintf(cmd, "g++ %s -o %s %s > /dev/null 2>&1", src, out, flags);
  //printf("Cmd: %s\n", cmd);
  if (system(cmd) != EXIT_SUCCESS) {
    int result = system("g++ --version > /dev/null 2>&1");
    if (result == 0) {
        _error("cmd[%s]", cmd);
        worker_->platform()->IncrementErrorCount();
        return IRIS_ERROR;
    }
    else {
        _warning("g++ is not available for JIT compilation of cmd [%s]", cmd);
        return IRIS_WARNING;
    }
  }
  return IRIS_SUCCESS;
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
  //ld_->set_dev(devno(), model());
  ld_->init(devno());
  if (julia_if_ != NULL) julia_if_->init(devno());
  set_first_event_cpu_begin_time(timer_->Now());
  set_first_event_cpu_end_time(timer_->Now());
  char flags[128];
  sprintf(flags, "-shared -fopenmp -fPIC");
  LoadDefaultKernelLibrary("DEFAULT_OMP_KERNELS", flags);

  return IRIS_SUCCESS;
}

int DeviceOpenMP::ResetMemory(Task *task, Command *cmd, BaseMem *mem)
{
    ResetData & reset_data = cmd->reset_data();
    if (cmd->reset_data().reset_type_ == iris_reset_memset) {
        uint8_t reset_value = reset_data.value_.u8;
        memset(mem->arch(this), reset_value, mem->size());
    }
    else {
        pair<bool, int8_t> out = mem->IsResetPossibleWithMemset(reset_data);
        if (out.first) {
            memset(mem->arch(this), out.second, mem->size());
        }
        else if (mem->GetMemHandlerType() == IRIS_DMEM || 
                mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
            size_t elem_size = ((DataMem*)mem)->elem_size();
            CallMemReset(mem, mem->size()/elem_size, cmd->reset_data(), NULL);
        }
        else {
            _error("Unknow reset type for memory:%lu\n", mem->uid());
        }
    }
    return IRIS_SUCCESS;
}

int DeviceOpenMP::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
  void** mpmem = mem_addr;
  if (posix_memalign(mpmem, 0x1000, size) != 0) {
    _error("%s", "posix_memalign");
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  _trace("Allocated mem:%lu addr:%p, size:%lu reset:%d", mem->uid(), *mem_addr, size, reset);
  if (reset) {
      if (mem->reset_data().reset_type_ == iris_reset_memset) {
          memset(*mpmem, 0, size);
      }
      else if (ld_default() != NULL) {
          pair<bool, int8_t> out = mem->IsResetPossibleWithMemset();
          if (out.first) {
              memset(*mpmem, out.second, size);
          }
          else if (mem->GetMemHandlerType() == IRIS_DMEM || 
                  mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
              size_t elem_size = ((DataMem*)mem)->elem_size();
              CallMemReset(mem, size/elem_size, mem->reset_data(), NULL);
          }
          else {
              _error("Unknow reset type for memory:%lu\n", mem->uid());
          }
      }
  }
  return IRIS_SUCCESS;
}

int DeviceOpenMP::MemFree(BaseMem *mem, void* mem_addr) {
  void* mpmem = mem_addr;
  if (mpmem) free(mpmem);
  return IRIS_SUCCESS;
}

int DeviceOpenMP::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* mpmem = mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
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
          _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] elem_size:%ld", tag, devno_, name_, task->uid(), task->name(), mem->uid(), mpmem, off[0], size, host, elem_size);
          memcpy((char*) mpmem, (char *)host+off[0]*elem_size, size);
#if 0
          printf("H2D: ");
          float *A = (float *) host;
          for(int i=0; i<size/4; i++) {
              printf("%10.1lf ", A[i]);
          }
          printf("\n");
#endif
      }
  return IRIS_SUCCESS;
}

int DeviceOpenMP::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* mpmem = mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
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
          _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] elem_size:%ld", tag, devno_, name_, task->uid(), task->name(), mem->uid(), mpmem, off[0], size, host, elem_size);
          memcpy((char *)host + off[0]*elem_size, (char*) mpmem, size);
#if 0
          printf("D2H: ");
          float *A = (float *) host;
          for(int i=0; i<size/4; i++) {
              printf("%10.1lf ", A[i]);
          }
          printf("\n");
#endif
      }
  return IRIS_SUCCESS;
}

int DeviceOpenMP::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  int kernel_idx=-1;
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      return IRIS_SUCCESS;
  }
  if (ld_->host_kernel(&kernel_idx, name) == IRIS_SUCCESS) {
      *kernel_bin = ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (report_error) {
      _error("Missing function for OpenMP kernel:%s", kernel->name());
      worker_->platform()->IncrementErrorCount();
  }
  return IRIS_ERROR;
}

int DeviceOpenMP::KernelLaunchInit(Command *cmd, Kernel* kernel) {
  //c_string_array data = ld_->iris_get_kernel_names();
  int status=IRIS_ERROR;
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      julia_if_->launch_init(model(), devno_, 0, 0, (void **)NULL, kernel->GetParamWrapperMemory(), cmd);
      julia_if_->set_julia_kernel_type(kernel->GetParamWrapperMemory(), kernel->task()->julia_kernel_type());
      return IRIS_SUCCESS;
  }
  else {
      status = ld_->launch_init(model(), devno_, 0, 0, NULL, kernel->GetParamWrapperMemory(), cmd);
      /*
         if (ld_->iris_openmp_kernel_with_obj)
         status = ld_->iris_openmp_kernel_with_obj(kernel->GetParamWrapperMemory(), kernel->name());
         else if (ld_->iris_openmp_kernel)
         status = ld_->iris_openmp_kernel(kernel->name());
       */
      if (status == IRIS_ERROR) {
          _error("Missing iris_openmp_kernel/iris_openmp_kernel_with_obj for OpenMP kernel:%s", kernel->name());
          worker_->platform()->IncrementErrorCount();
      }
  }
  return status;
}

int DeviceOpenMP::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
    if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
       julia_if_->setarg(
              kernel->GetParamWrapperMemory(), kindex, size, value);
       return IRIS_SUCCESS;
    }
    else {
        int status = ld_->setarg(
                kernel->GetParamWrapperMemory(), kindex, size, value);
        /*
           if (ld_->iris_openmp_setarg_with_obj)
           return ld_->iris_openmp_setarg_with_obj(kernel->GetParamWrapperMemory(), kindex, size, value);
           if (ld_->iris_openmp_setarg)
           return ld_->iris_openmp_setarg(kindex, size, value);
         */
        if (status != IRIS_SUCCESS) {
            _error("Missing host iris_openmp_setarg/iris_openmp_setarg_with_obj function for OpenMP kernel:%s", kernel->name());
            worker_->platform()->IncrementErrorCount();
        }
        return status;
    }
  return IRIS_ERROR;
}

int DeviceOpenMP::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  void* mpmem = (char*) mem->arch(this) + off;
  size_t size = mem->size() - off;
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      julia_if_->setmem(
              kernel->GetParamWrapperMemory(), mem, kindex, mpmem, size);
      return IRIS_SUCCESS;
  }
  else {
      int status = ld_->setmem(kernel->GetParamWrapperMemory(), kindex, mpmem, size);
      /*
         if (ld_->iris_openmp_setmem_with_obj)
         return ld_->iris_openmp_setmem_with_obj(kernel->GetParamWrapperMemory(), kindex, mpmem);
         if (ld_->iris_openmp_setmem)
         return ld_->iris_openmp_setmem(kindex, mpmem);
       */
      if (status == IRIS_SUCCESS) return IRIS_SUCCESS;
      _error("Missing host iris_openmp_setmem/iris_openmp_setmem_with_obj function for OpenMP kernel:%s", kernel->name());
      worker_->platform()->IncrementErrorCount();
  }
  return IRIS_ERROR;
}

int DeviceOpenMP::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  //printf(" Task %s\n", kernel->get_task_name());
  _trace("dev[%d] kernel[%s:%s] dim[%d] off[%lu] gws[%lu]", devno_, kernel->name(), kernel->get_task_name(), dim, off[0], gws[0]);
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      julia_if_->host_launch(kernel->task()->uid(), (void **)NULL, 0, NULL, false,
                  0, kernel->name(), 
                  kernel->GetParamWrapperMemory(), devno(),
                  dim, off, gws);
      return IRIS_SUCCESS;
  }
  else if (ld_->host_launch(NULL, 0, 0, kernel->name(), kernel->GetParamWrapperMemory(), devno(),
              dim, off, gws) == IRIS_SUCCESS) 
      return IRIS_SUCCESS;
  /*
  ld_->SetKernelPtr(kernel->GetParamWrapperMemory(), kernel->name());
  if (ld_->iris_openmp_launch_with_obj)
    return ld_->iris_openmp_launch_with_obj(NULL, 
            kernel->GetParamWrapperMemory(), 0, dim, off[0], gws[0]);
  if (ld_->iris_openmp_launch)
    return ld_->iris_openmp_launch(dim, off[0], gws[0]);
    */
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


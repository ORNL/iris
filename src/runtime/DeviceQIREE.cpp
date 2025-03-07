#include "DeviceQIREE.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderQIREE.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "Task.h"
#include "Utils.h"
#include "Worker.h"
//#include <iris/qiree/rpcmem.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

namespace iris {
namespace rt {

DeviceQIREE::DeviceQIREE(LoaderQIREE* ld, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  qiree_nparams_ = 0;
  //type_ = iris_qiree;
  //model_ = iris_qiree;
  strcpy(name_, "QIREE");
  _info("device[%d] platform[%d] device[%s] type[%d]", devno_, platform_, name_, type_);
}

DeviceQIREE::~DeviceQIREE() {
  //ld_->iris_qiree_finalize();
  //return IRIS_SUCCESS;
}

int DeviceQIREE::Init() {
  //ld_->iris_hexagon_init();
  return IRIS_SUCCESS;
}


int DeviceQIREE::ResetMemory(Task *task, Command *cmd, BaseMem *mem) {
    _error("Reset memory is not implemented yet !");
    return IRIS_ERROR;
}

int DeviceQIREE::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
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

int DeviceQIREE::MemFree(BaseMem *mem, void* mem_addr) {
  void* mpmem = mem_addr;
  if (mpmem) free(mpmem);
  return IRIS_SUCCESS;
}

int DeviceQIREE::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
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

int DeviceQIREE::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
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



int DeviceQIREE::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  int kernel_idx=-1;
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      return IRIS_SUCCESS;
  }
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelLaunchInit(Command *cmd, Kernel* kernel) {
  qiree_nparams_ = 0;
  /*if (ld_->iris_qiree_kernel_with_obj) {
      if (ld_->iris_qiree_kernel_with_obj(
              kernel->GetParamWrapperMemory(), kernel->name())==IRIS_SUCCESS) {
          return IRIS_SUCCESS;
      } */
  //}
  //return ld_->iris_qiree_kernel(kernel->name());
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
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
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  void* mpmem = (char*) mem->arch(this) + off;
  size_t size = mem->size() - off;
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      julia_if_->setmem(
              kernel->GetParamWrapperMemory(), mem, kindex, mpmem, size);
      return IRIS_SUCCESS;
  }
  else {
      int status = IRIS_SUCCESS;
      qiree_params_[kindex+1] = mpmem;
      qiree_nparams_ = (kindex+1 >= qiree_nparams_) ? kindex+2 : qiree_nparams_;
      //printf("p%d: %p (%s) n:%d\n", kindex+1, mpmem, (char *)mpmem, qiree_nparams_);
      //int status = ld_->setmem(kernel->GetParamWrapperMemory(), kindex, mpmem, size);
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
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] off[%zu] gws[%zu]", kernel->name(), dim, off[0], gws[0]);
  qiree_params_[0] = (void *)"null";
#if 0
  char* argv[6]; 
  argv[0] = "null";
  argv[1] = "bell.ll";
  argv[2] = "-a";
  argv[3] = "qpp";
  argv[4] = "-s";
  argv[5] = "qpp";
#endif
  // Call the function with an argument
  //parse_input_c(4, (char**)argv);
  //printf("nparams: %d\n", qiree_nparams_);
  ld_->parse_input_c(qiree_nparams_, (char**)qiree_params_);
  //return ld_->parse_input_c(4, (char**)argv);
  return IRIS_SUCCESS;
}

int DeviceQIREE::Synchronize() {
  return IRIS_SUCCESS;
}

int DeviceQIREE::AddCallback(Task* task) {
  task->Complete();
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


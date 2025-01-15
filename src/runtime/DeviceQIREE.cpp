#include "DeviceQIREE.h"
#include "Debug.h"
#include "Kernel.h"
#include "LoaderQIREE.h"
#include "BaseMem.h"
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


int DeviceQIREE::ResetMemory(Task *task, BaseMem *mem, uint8_t reset_value) {
    _error("Reset memory is not implemented yet !");
    return IRIS_ERROR;
}

int DeviceQIREE::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
  return IRIS_SUCCESS;
}

int DeviceQIREE::MemFree(BaseMem *mem, void* mem_addr) {
  return IRIS_SUCCESS;
}

int DeviceQIREE::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  return IRIS_SUCCESS;
}

int DeviceQIREE::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  return IRIS_SUCCESS;
}



int DeviceQIREE::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelLaunchInit(Command *cmd, Kernel* kernel) {
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
  /*
  if (ld_->iris_qiree_setarg_with_obj)
      return ld_->iris_qiree_setarg_with_obj(
              kernel->GetParamWrapperMemory(), kindex, size, value);
  return ld_->iris_qiree_setarg(kindex, size, value); 
  */
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  /*
  void* hxgmem = mem->arch(this);
  if (ld_->iris_qiree_setmem_with_obj)
      return ld_->iris_qiree_setmem_with_obj(
              kernel->GetParamWrapperMemory(), kindex, hxgmem, (int) mem->size());
  return ld_->iris_qiree_setmem(kindex, hxgmem, (int) mem->size());
  */
  return IRIS_SUCCESS;
}

int DeviceQIREE::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] off[%zu] gws[%zu]", kernel->name(), dim, off[0], gws[0]);

  char* argv[4]; 
  argv[0] = "null";
  argv[1] = "bell.ll";
  argv[2] = "-a";
  argv[3] = "qpp";
  // Call the function with an argument
  //parse_input_c(4, (char**)argv);
  ld_->parse_input_c(4, (char**)argv);
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


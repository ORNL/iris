#include "DeviceLevelZero.h"
#include "Debug.h"
#include "Command.h"
#include "LoaderLevelZero.h"
#include "BaseMem.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"
#include "Worker.h"

namespace iris {
namespace rt {

DeviceLevelZero::DeviceLevelZero(LoaderLevelZero* ld, ze_device_handle_t zedev, ze_context_handle_t zectx, ze_driver_handle_t zedriver, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  zedev_ = zedev;
  zectx_ = zectx;
  zedriver_ = zedriver;

  ze_device_properties_t props = {};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  err_ = ld_->zeDeviceGetProperties(zedev_, &props); 
  _zeerror(err_);

  strcpy(name_, props.name);

  type_ = iris_gpu_intel;
  align_ = 0x1000;

  _info("device[%d] platform[%d] device[%s] type[0x%x:%d] align[0x%lx]", devno_, platform_, name_, type_, type_, align_);
}

DeviceLevelZero::~DeviceLevelZero() {
}

int DeviceLevelZero::Compile(char* src) {
  char cmd[1024];
  memset(cmd, 0, 256);
  sprintf(cmd, "clang -cc1 -finclude-default-header -triple spir %s -flto -emit-llvm-bc -o %s.bc", src, kernel_path_);
  if (system(cmd) != EXIT_SUCCESS) {
    _error("cmd[%s]", cmd);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  sprintf(cmd, "llvm-spirv %s.bc -o %s", kernel_path_, kernel_path_);
  if (system(cmd) != EXIT_SUCCESS) {
    _error("cmd[%s]", cmd);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceLevelZero::Init() {
  ze_command_queue_desc_t cmq_desc = {};
  cmq_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  err_ = ld_->zeCommandQueueCreate(zectx_, zedev_, &cmq_desc, &zecmq_);
  _zeerror(err_);

  ze_command_queue_desc_t altdesc = {};
  altdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  err_ = ld_->zeCommandListCreateImmediate(zectx_, zedev_, &altdesc, &zecml_);

  ze_event_pool_desc_t evtpool_desc = {};
  evtpool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  evtpool_desc.count = 1;
  evtpool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  err_ = ld_->zeEventPoolCreate(zectx_, &evtpool_desc, 1, &zedev_, &zeevtpool_);
  _zeerror(err_);

  char* path = kernel_path_;
//  Platform::GetPlatform()->EnvironmentGet("KERNEL_BIN_SPV", &path, NULL);
  uint8_t* src = nullptr;
  size_t srclen = 0;
  if (Utils::ReadFile(path, (char**) &src, &srclen) == IRIS_ERROR) {
    _error("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return IRIS_SUCCESS;
  }

  ze_module_desc_t mod_desc = {}; 
  mod_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mod_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mod_desc.inputSize = srclen;
  mod_desc.pInputModule = src;

  ld_->Lock();
  err_ = ld_->zeModuleCreate(zectx_, zedev_, &mod_desc, &zemod_, nullptr);
  ld_->Unlock();
  _zeerror(err_);

  if (src) free(src);

  return IRIS_SUCCESS;
}

int DeviceLevelZero::ResetMemory(BaseMem *mem, uint8_t reset_value)
{
    _error("Reset Memory is not implemented");
    return IRIS_ERROR;
}
int DeviceLevelZero::MemAlloc(void** mem, size_t size, bool reset) {
  void** dptr = mem;
  ze_device_mem_alloc_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  err_ = ld_->zeMemAllocDevice(zectx_, &desc, size, align_, zedev_, dptr);
  if (reset) {
    _error("Levelzero not supported with reset for size:%lu", size);
  }
  _zeerror(err_);
  return IRIS_SUCCESS;
}

int DeviceLevelZero::MemFree(void* mem) {
  void* dptr = mem;
  err_ = ld_->zeMemFree(zectx_, dptr);
  _zeerror(err_);
  return IRIS_SUCCESS;
}

int DeviceLevelZero::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *tile_sizes,  size_t *full_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* dptr = (void*) ((char*) mem->arch(this) + off[0]);
  _trace("%sdptr[%p] offset[%lu] size[%lu] host[%p]", tag,  dptr, off[0], size, host);

  ze_event_handle_t zeevt;
  ze_event_desc_t zeevt_desc = {};
  zeevt_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  err_ = ld_->zeEventCreate(zeevtpool_, &zeevt_desc, &zeevt);
  _zeerror(err_);

  err_ = ld_->zeCommandListAppendMemoryCopy(zecml_, dptr, (const void*) host, size, zeevt, 0, nullptr);
  _zeerror(err_);

  err_ = ld_->zeCommandListAppendSignalEvent(zecml_, zeevt);
  _zeerror(err_);

  err_ = ld_->zeEventHostSynchronize(zeevt, UINT64_MAX);
  _zeerror(err_);

  /*
  err_ = ld_->zeCommandQueueExecuteCommandLists(zecmq_, 1, &zecml, nullptr);
  _zeerror(err_);

  err_ = ld_->zeCommandQueueSynchronize(zecmq_, 1000000);
  _zeerror(err_);
  */

  err_ = ld_->zeCommandListReset(zecml_);
  _zeerror(err_);

  return IRIS_SUCCESS;
}

int DeviceLevelZero::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *tile_sizes,  size_t *full_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  void* dptr = (void*) ((char*) mem->arch(this) + off[0]);
  _trace("%sdptr[%p] offset[%lu] size[%lu] host[%p]", tag, dptr, off[0], size, host);

  ze_event_handle_t zeevt;
  ze_event_desc_t zeevt_desc = {};
  zeevt_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  err_ = ld_->zeEventCreate(zeevtpool_, &zeevt_desc, &zeevt);
  _zeerror(err_);

  err_ = ld_->zeCommandListAppendMemoryCopy(zecml_, host, (const void*) dptr, size, zeevt, 0, nullptr);
  _zeerror(err_);

  err_ = ld_->zeCommandListAppendSignalEvent(zecml_, zeevt);
  _zeerror(err_);

  err_ = ld_->zeEventHostSynchronize(zeevt, UINT64_MAX);
  _zeerror(err_);

  err_ = ld_->zeCommandListReset(zecml_);
  _zeerror(err_);

  return IRIS_SUCCESS;
}

int DeviceLevelZero::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  ze_kernel_handle_t* zekernel = (ze_kernel_handle_t*) kernel_bin;
  ze_kernel_desc_t kernel_desc = {};
  kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernel_desc.pKernelName = name;
  err_ = ld_->zeKernelCreate(zemod_, &kernel_desc, zekernel);
  if (report_error) _zeerror(err_);
  return IRIS_SUCCESS;
}

int DeviceLevelZero::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  ze_kernel_handle_t zekernel = (ze_kernel_handle_t) kernel->arch(this);
  err_ = ld_->zeKernelSetArgumentValue(zekernel, idx, size, value);
  _zeerror(err_);
  return IRIS_SUCCESS;
}

int DeviceLevelZero::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  ze_kernel_handle_t zekernel = (ze_kernel_handle_t) kernel->arch(this);
  void* dptr = (void*) ((char*) mem->arch(this) + off);
  err_ = ld_->zeKernelSetArgumentValue(zekernel, idx, sizeof(dptr), &dptr);
  _zeerror(err_);
  return IRIS_SUCCESS;
}

int DeviceLevelZero::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] gws[%zu,%zu,%zu] lws[%zu,%zu,%zu]", kernel->name(), dim, gws[0], gws[1], gws[2], lws ? lws[0] : 0, lws ? lws[1] : 0, lws ? lws[2] : 0);

  ze_event_handle_t zeevt;
  ze_event_desc_t zeevt_desc = {};
  zeevt_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  err_ = ld_->zeEventCreate(zeevtpool_, &zeevt_desc, &zeevt);
  _zeerror(err_);

  ze_kernel_handle_t zekernel = (ze_kernel_handle_t) kernel->arch(this);
  ze_group_count_t group_count;
  group_count.groupCountX = gws[0];
  group_count.groupCountY = gws[1];
  group_count.groupCountZ = gws[2];
  err_ = ld_->zeCommandListAppendLaunchKernel(zecml_, zekernel, &group_count, zeevt, 0, nullptr);

  err_ = ld_->zeCommandListAppendSignalEvent(zecml_, zeevt);
  _zeerror(err_);

  err_ = ld_->zeEventHostSynchronize(zeevt, UINT64_MAX);
  _zeerror(err_);

  err_ = ld_->zeCommandListReset(zecml_);
  _zeerror(err_);
  
  return IRIS_SUCCESS;
}

int DeviceLevelZero::Synchronize() {
  return IRIS_SUCCESS;
}

int DeviceLevelZero::AddCallback(Task* task) {
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


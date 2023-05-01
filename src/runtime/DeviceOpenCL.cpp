#include "DeviceOpenCL.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderOpenCL.h"
#include "LoaderHost2OpenCL.h"
#include "BaseMem.h"
#include "Mem.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Utils.h"
#include "Worker.h"

namespace iris {
namespace rt {

std::string DeviceOpenCL::GetLoaderHost2OpenCLSuffix(LoaderOpenCL *ld, cl_device_id cldev)
{
    cl_device_type cltype;
    char vendor[64];
    char version[64];
    char name[64];
    cl_int err = ld->clGetDeviceInfo(cldev, CL_DEVICE_TYPE, sizeof(cltype), &cltype, NULL);
    err = ld->clGetDeviceInfo(cldev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    err = ld->clGetDeviceInfo(cldev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    err = ld->clGetDeviceInfo(cldev, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    std::string fpga_bin_suffix = "xilinx";   
    int type;
    if (cltype == CL_DEVICE_TYPE_CPU) type = iris_cpu;
    else if (cltype == CL_DEVICE_TYPE_GPU) {
        type = iris_gpu;
        if (strcasestr(vendor, "NVIDIA")) type = iris_nvidia;
        else if (strcasestr(vendor, "Advanced Micro Devices")) type = iris_amd;
    }
    else if (cltype == CL_DEVICE_TYPE_ACCELERATOR) {
        if (strstr(vendor, "Intel") != NULL) { type = iris_fpga; fpga_bin_suffix = "intel"; }
        if (strstr(vendor, "Xilinx") != NULL) { type = iris_fpga; fpga_bin_suffix = "xilinx"; }
        else if (strstr(name, "FPGA") != NULL || strstr(version, "FPGA") != NULL) { type = iris_fpga; fpga_bin_suffix = "fpga";}
        else type = iris_phi;
    }
    else type = iris_cpu;
    std::string output_suffix = 
            type == iris_fpga   ? fpga_bin_suffix   : "cl";
    return output_suffix;
}
DeviceOpenCL::DeviceOpenCL(LoaderOpenCL* ld, LoaderHost2OpenCL *host2opencl_ld, cl_device_id cldev, cl_context clctx, int devno, int ocldevno, int platform) : Device(devno, platform) {
  ld_ = ld;
  ocldevno_ = ocldevno;
  host2opencl_ld_ = host2opencl_ld;
  cldev_ = cldev;
  clctx_ = clctx;
  clprog_ = NULL;
  timer_ = new Timer();
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_), &max_compute_units_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size_), &max_work_group_size_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes_), max_work_item_sizes_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available_), &compiler_available_, NULL);
  fpga_bin_suffix_ = "aocx";   

  if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = iris_cpu;
  else if (cltype_ == CL_DEVICE_TYPE_GPU) {
    type_ = iris_gpu;
    if (strcasestr(vendor_, "NVIDIA")) type_ = iris_nvidia;
    else if (strcasestr(vendor_, "Advanced Micro Devices")) type_ = iris_amd;
  }
  else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) {
    if (strstr(vendor_, "Xilinx") != NULL) { type_ = iris_fpga; fpga_bin_suffix_ = "xclbin"; }
    else if (strstr(name_, "FPGA") != NULL || strstr(version_, "FPGA") != NULL) type_ = iris_fpga;
    else type_ = iris_phi;
  }
  else type_ = iris_cpu;
  model_ = iris_opencl;

  _info("device[%d] platform[%d] vendor[%s] device[%s] type[0x%x:%d] version[%s] max_compute_units[%zu] max_work_group_size[%zu] max_work_item_sizes[%zu,%zu,%zu] compiler_available[%d]", devno_, platform_, vendor_, name_, type_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], compiler_available_);
}

DeviceOpenCL::~DeviceOpenCL() {
    if (host2opencl_ld_->iris_host2opencl_finalize)
        host2opencl_ld_->iris_host2opencl_finalize();
    if (host2opencl_ld_->iris_host2opencl_finalize_handles)
        host2opencl_ld_->iris_host2opencl_finalize_handles(ocldevno_);
}

int DeviceOpenCL::Init() {
  clcmdq_ = ld_->clCreateCommandQueue(clctx_, cldev_, 0, &err_);
  if (host2opencl_ld_->iris_host2opencl_init)
      host2opencl_ld_->iris_host2opencl_init();
  if (host2opencl_ld_->iris_host2opencl_init_handles)
      host2opencl_ld_->iris_host2opencl_init_handles(ocldevno_);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }

  cl_int status;
  char* src = NULL;
  size_t len = 0;
  if (CreateProgram("spv", &src, &len) == IRIS_SUCCESS) {
    if (type_ == iris_fpga) clprog_ = ld_->clCreateProgramWithBinary(clctx_, 1, &cldev_, (const size_t*) &len, (const unsigned char**) &src, &status, &err_);
    else clprog_ = ld_->clCreateProgramWithIL(clctx_, (const void*) src, len, &err_);
    _clerror(err_);
    if (err_ != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  } else if (CreateProgram("cl", &src, &len) == IRIS_SUCCESS) {
    clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &len, &err_);
    _clerror(err_);
    if (err_ != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  } else {
    _error("dev[%d][%s] has no kernel file", devno_, name_);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  err_ = ld_->clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) {
    cl_build_status s;
    err_ = ld_->clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_STATUS, sizeof(s), &s, NULL);
    _clerror(err_);
    err_ = ld_->clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* log = (char*) malloc(len + 1);
    err_ = ld_->clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, len + 1, log, NULL);
    _clerror(err_);
    _error("status[%d]  log:%s", s, log);
    _error("srclen[%zu] src\n%s", len, src);
    if (src) free(src);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  size_t nkernels=0;
  ld_->clGetProgramInfo(clprog_, CL_PROGRAM_NUM_KERNELS, sizeof(nkernels), &nkernels, NULL);
  ld_->clGetProgramInfo(clprog_, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &len);
  char* kernel_names = (char*) malloc(len + 1);
  ld_->clGetProgramInfo(clprog_, CL_PROGRAM_KERNEL_NAMES, len + 1, kernel_names, NULL);
  _trace("nkernels[%zu] kernel_names[%s]", nkernels, kernel_names);
  free(kernel_names);
  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceOpenCL::BuildProgram(char* path) {
  if (clprog_) {
    err_ = ld_->clReleaseProgram(clprog_);
    _clerror(err_);
  }

  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
    _error("path[%s]", path);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }

  err_ = ld_->clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }

  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceOpenCL::ResetMemory(BaseMem *mem, uint8_t reset_value) {
    _error("Reset memory is not implemented yet !");
    return IRIS_ERROR;
}

int DeviceOpenCL::MemAlloc(void** mem, size_t size, bool reset) {
  cl_mem* clmem = (cl_mem*) mem;
  *clmem = ld_->clCreateBuffer(clctx_, CL_MEM_READ_WRITE, size, NULL, &err_);
  if (reset) {
    _error("OpenCL not supported with reset for size:%lu", size);
  }
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemFree(void* mem) {
  cl_mem clmem = (cl_mem) mem;
  err_ = ld_->clReleaseMemObject(clmem);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  cl_mem clmem = (cl_mem) mem->arch(this);
  if (dim == 2 || dim ==3) {
      size_t host_row_pitch = elem_size * host_sizes[0];
      size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
      size_t dev_row_pitch = elem_size * dev_sizes[0];
      size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
      size_t buffer_origin[3] = { 0, 0, 0};
      size_t host_origin[3] = {off[0] * elem_size, off[1], off[2]};
      size_t region[3] = { dev_sizes[0]*elem_size, dev_sizes[1], dev_sizes[2] };
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      err_ = ld_->clEnqueueWriteBufferRect(clcmdq_, clmem, CL_TRUE, buffer_origin, host_origin, region, dev_row_pitch, dev_slice_pitch, host_row_pitch, host_slice_pitch, host, 0, NULL, NULL);
#if 0
      float *hostA = new float[dev_sizes[0] * dev_sizes[1]];
      int SIZE = dev_sizes[0]*dev_sizes[1];
      printf("dev[%d] OFF:(%d,%d,%d) DEV:(%d,%d,%d) HOST:(%d,%d,%d) ELEM:%d\n", devno_, off[0], off[1], off[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], host_sizes[0], host_sizes[1], host_sizes[2], elem_size);
      err_ = ld_->clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, 0, dev_sizes[0]*dev_sizes[1]*elem_size, hostA, 0, NULL, NULL);
      int print_size = (SIZE > 8) ? 8: SIZE;
      printf("H2DOffset: dev:%d hostA=\n", devno_);
      for(int i=0; i<print_size; i++) {
          printf("%10.1lf ", hostA[i]);
      }
      printf("\n");
#endif
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], size, host, q_);
      err_ = ld_->clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off[0], size, host, 0, NULL, NULL);
#if 0
      printf("H2D: Dev%d: ", devno_);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  cl_mem clmem = (cl_mem) mem->arch(this);
  if (dim == 2 || dim ==3) {
      size_t host_row_pitch = elem_size * host_sizes[0];
      size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
      size_t dev_row_pitch = elem_size * dev_sizes[0];
      size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
      size_t buffer_origin[3] = { 0, 0, 0};
      size_t host_origin[3] = {off[0] * elem_size, off[1], off[2]};
      size_t region[3] = { dev_sizes[0]*elem_size, dev_sizes[1], dev_sizes[2] };
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      err_ = ld_->clEnqueueReadBufferRect(clcmdq_, clmem, CL_TRUE, buffer_origin, host_origin, region, dev_row_pitch, dev_slice_pitch, host_row_pitch, host_slice_pitch, host, 0, NULL, NULL);
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], size, host, q_);
      err_ = ld_->clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off[0], size, host, 0, NULL, NULL);
#if 0
      printf("D2H: Dev:%d: ", devno_);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  int kernel_idx = -1;
  if (kernel->is_vendor_specific_kernel(devno_)) {
      //_trace("dev[%d][%s] kernel[%s:%s] kernel-get", devno_, name_, kernel->name(), kernel->get_task_name());
      if (host2opencl_ld_->iris_host2opencl_kernel_with_obj) {
          //_trace("dev[%d][%s] kernel[%s:%s] kernel-get-1", devno_, name_, kernel->name(), kernel->get_task_name());
          if (host2opencl_ld_->iris_host2opencl_kernel_with_obj(&kernel_idx, name)==IRIS_SUCCESS) {
              //_trace("dev[%d][%s] kernel[%s:%s] kernel-get-2", devno_, name_, kernel->name(), kernel->get_task_name());
              *kernel_bin = host2opencl_ld_->GetFunctionPtr(name);
              return IRIS_SUCCESS;
          }
      }
  }
  //_trace("dev[%d][%s] kernel[%s:%s] kernel-get-3", devno_, name_, kernel->name(), kernel->get_task_name());
  if (kernel->is_vendor_specific_kernel(devno_) && 
          host2opencl_ld_->iris_host2opencl_kernel)
      return IRIS_SUCCESS;
  //_trace("dev[%d][%s] kernel[%s:%s] kernel-get-4", devno_, name_, kernel->name(), kernel->get_task_name());
  cl_kernel* clkernel = (cl_kernel*) kernel_bin;
  *clkernel = ld_->clCreateKernel(clprog_, name, &err_);
  if (report_error) _clerror(err_);
  if (err_ != CL_SUCCESS){
    if (report_error) worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  if (kernel->is_vendor_specific_kernel(devno_)) {
      //_trace("dev[%d][%s] kernel[%s:%s] kernel-setarg-0", devno_, name_, kernel->name(), kernel->get_task_name());
      if (host2opencl_ld_->iris_host2opencl_setarg_with_obj) {
          //_trace("dev[%d][%s] kernel[%s:%s] kernel-setarg-1", devno_, name_, kernel->name(), kernel->get_task_name());
          host2opencl_ld_->iris_host2opencl_setarg_with_obj(
                  kernel->GetParamWrapperMemory(), kindex, size, value);
      }
  }
  else if (kernel->is_vendor_specific_kernel(devno_) && host2opencl_ld_->iris_host2opencl_setarg)
      host2opencl_ld_->iris_host2opencl_setarg(kindex, size, value);
  else {
    cl_kernel clkernel = (cl_kernel) kernel->arch(this);
    err_ = ld_->clSetKernelArg(clkernel, (cl_uint) idx, size, value);
    _clerror(err_);
    if (err_ != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  cl_mem clmem = (cl_mem) mem->arch(this);
  if (kernel->is_vendor_specific_kernel(devno_)) {
      //_trace("dev[%d][%s] kernel[%s:%s] kernel-setmem-0", devno_, name_, kernel->name(), kernel->get_task_name());
      if (host2opencl_ld_->iris_host2opencl_setmem_with_obj) {
          //_trace("dev[%d][%s] kernel[%s:%s] kernel-setmem-1", devno_, name_, kernel->name(), kernel->get_task_name());
          host2opencl_ld_->iris_host2opencl_setmem_with_obj(
                  kernel->GetParamWrapperMemory(), kindex, clmem);
      }
  }
  else if (kernel->is_vendor_specific_kernel(devno_) && host2opencl_ld_->iris_host2opencl_setmem)
      host2opencl_ld_->iris_host2opencl_setmem(kindex, clmem);
  else {
    err_ = ld_->clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
    _clerror(err_);
    if (err_ != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  }
  return IRIS_SUCCESS;
}

void DeviceOpenCL::CheckVendorSpecificKernel(Kernel *kernel) {
    kernel->set_vendor_specific_kernel(devno_, false);
    //_trace("dev[%d][%s] kernel[%p:%s:%s] launchInit-0", devno_, name_, kernel, kernel->name(), kernel->get_task_name());
    if (host2opencl_ld_->iris_host2opencl_kernel_with_obj) {
        host2opencl_ld_->iris_host2opencl_set_queue_with_obj(
                kernel->GetParamWrapperMemory(), &clcmdq_);
        //_trace("dev[%d][%s] kernel[%s:%s] launchInit-1", devno_, name_, kernel->name(), kernel->get_task_name());
        int status = host2opencl_ld_->iris_host2opencl_kernel_with_obj(
                kernel->GetParamWrapperMemory(), kernel->name());
        if (status == IRIS_SUCCESS && 
                host2opencl_ld_->IsFunctionExists(kernel->name())) { 
            //_trace("dev[%d][%s] kernel[%s:%s] launchInit-2", devno_, name_, kernel->name(), kernel->get_task_name());
                //_trace("dev[%d][%s] kernel[%s:%s] launchInit-3", devno_, name_, kernel->name(), kernel->get_task_name());
            kernel->set_vendor_specific_kernel(devno_, true);
        }
    }
    else if (host2opencl_ld_->iris_host2opencl_kernel) {
        host2opencl_ld_->iris_host2opencl_set_queue(&clcmdq_);
        int status = host2opencl_ld_->iris_host2opencl_kernel(kernel->name());
        if (status == IRIS_SUCCESS && 
                host2opencl_ld_->IsFunctionExists(kernel->name())) {
            kernel->set_vendor_specific_kernel(devno_, true);
        }
    }
    kernel->set_vendor_specific_kernel_check(devno_, true);
}
int DeviceOpenCL::KernelLaunchInit(Kernel* kernel) {
    return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("dev[%d][%s] kernel[%s:%s] dim[%d] gws[%zu,%zu,%zu] lws[%zu,%zu,%zu]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, gws[0], gws[1], gws[2], lws ? lws[0] : 0, lws ? lws[1] : 0, lws ? lws[2] : 0);
  if (kernel->is_vendor_specific_kernel(devno_)) {
      if (host2opencl_ld_->iris_host2opencl_launch_with_obj) {
          host2opencl_ld_->SetKernelPtr(kernel->GetParamWrapperMemory(), kernel->name());
          host2opencl_ld_->iris_host2opencl_launch_with_obj(
                  kernel->GetParamWrapperMemory(), ocldevno_, dim, off[0], gws[0]);
          return IRIS_SUCCESS; 
      }
      else if (host2opencl_ld_->iris_host2opencl_launch) {
          host2opencl_ld_->iris_host2opencl_launch(dim, off[0], gws[0]);
          return IRIS_SUCCESS; 
      }
  }
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  err_ = ld_->clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
#ifdef IRIS_SYNC_EXECUTION
  err_ = ld_->clFinish(clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
#endif
  return IRIS_SUCCESS;
}

int DeviceOpenCL::Synchronize() {
  err_ = ld_->clFinish(clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::AddCallback(Task* task) {
  task->Complete();
  return IRIS_SUCCESS;
}

void DeviceOpenCL::ExecuteKernel(Command* cmd) {
  timer_->Start(IRIS_TIMER_KERNEL);
  Kernel* kernel = ExecuteSelectorKernel(cmd);
  int dim = cmd->dim();
  size_t* off = cmd->off();
  size_t* gws = cmd->gws();
  size_t* lws = cmd->lws();
  //bool reduction = false;
  iris_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  int max_idx = 0;
  int mem_idx = 0;
  kernel->set_vendor_specific_kernel(devno_, false);
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  KernelLaunchInit(kernel);
  KernelArg* args = cmd->kernel_args();
  int *params_map = cmd->get_params_map();
  int arg_idx = 0;
  for (int idx = 0; idx < cmd->kernel_nargs(); idx++) {
    if (idx > max_idx) max_idx = idx;
    KernelArg* arg = args + idx;
    if (params_map != NULL && 
        (params_map[idx] & iris_all) == 0 && 
        !(params_map[idx] & type_) ) continue;
    BaseMem* bmem = arg->mem;
    if (bmem && bmem->GetMemHandlerType() == IRIS_MEM) {
      Mem *mem = (Mem *)bmem;
      if (arg->mode == iris_w || arg->mode == iris_rw) {
        if (npolymems) {
          iris_poly_mem* pm = polymems + mem_idx;
          mem->SetOwner(pm->typesz * pm->w0, pm->typesz * (pm->w1 - pm->w0 + 1), this);
        } else mem->SetOwner(arg->mem_off, arg->mem_size, this);
      }
      if (mem->mode() & iris_reduction) {
        lws = (size_t*) alloca(3 * sizeof(size_t));
        lws[0] = 1;
        lws[1] = 1;
        lws[2] = 1;
        while (max_compute_units_ * lws[0] < gws[0]) lws[0] <<= 1;
        while (max_work_item_sizes_[0] / 4 < lws[0]) lws[0] >>= 1;
        size_t expansion = (gws[0] + lws[0] - 1) / lws[0];
        gws[0] = lws[0] * expansion;
        mem->Expand(expansion);
        KernelSetMem(kernel, arg_idx, idx, mem, arg->off);
        KernelSetArg(kernel, arg_idx + 1, idx, lws[0] * mem->type_size(), NULL);
        //reduction = true;
        if (idx + 1 > max_idx) max_idx = idx + 1;
        idx++;
        arg_idx+=2;
      } else {
          KernelSetMem(kernel, arg_idx, idx, mem, arg->off); arg_idx+=1;
      }
      mem_idx++;
    } else if (bmem) {
        KernelSetMem(kernel, arg_idx, idx, bmem, arg->off); arg_idx+=1; 
        mem_idx++;
    } else {
        KernelSetArg(kernel, arg_idx, idx, arg->size, arg->value);
        arg_idx+=1;
    }
  }
#if 0
  if (reduction) {
    size_t gws0 = gws[0];
    _trace("max_idx+1[%d] gws[%lu]", max_idx + 1, gws0);
    KernelSetArg(kernel, max_idx + 1, sizeof(size_t), &gws0);
  }
#endif
  bool enabled = true;
  if (cmd->task() != NULL && cmd->task()->is_kernel_launch_disabled())
      enabled = false;
  if (enabled)
      errid_ = KernelLaunch(kernel, dim, off, gws, lws[0] > 0 ? lws : NULL);
  if (errid_ != IRIS_SUCCESS) {
    _error("iret[%d]", errid_); 
    worker_->platform()->IncrementErrorCount();
    printf("OpenCL error count = %i",errid_);
  }
  double time = timer_->Stop(IRIS_TIMER_KERNEL);
  cmd->SetTime(time);
  cmd->kernel()->history()->AddKernel(cmd, this, time);
}

int DeviceOpenCL::CreateProgram(const char* suffix, char** src, size_t* srclen) {
  char* p = NULL;
  if (Platform::GetPlatform()->GetFilePath(strcmp("spv", suffix) == 0 ? "KERNEL_BIN_SPV" : "KERNEL_SRC_SPV", &p, NULL) == IRIS_SUCCESS) {
    Utils::ReadFile(p, src, srclen);
  }

  if (*srclen > 0) {
    _trace("dev[%d][%s] kernels[%s]", devno_, name_, p);
    return IRIS_SUCCESS;
  }
  if (type_ == iris_fpga) {
      if (p != NULL) { free(p); p = NULL; }
      if (strcmp("aocx", fpga_bin_suffix_.c_str()) == 0 && 
              Platform::GetPlatform()->GetFilePath("KERNEL_INTEL_AOCX", &p, NULL) == IRIS_SUCCESS) {
          Utils::ReadFile(p, src, srclen);
          if (*srclen > 0) {
              _trace("dev[%d][%s] kernels[%s]", devno_, name_, p);
              return IRIS_SUCCESS;
          }
      }
      if (p != NULL) { free(p); p = NULL; }
      if (strcmp("xclbin", fpga_bin_suffix_.c_str()) == 0 && 
              Platform::GetPlatform()->GetFilePath("KERNEL_XILINX_XCLBIN", &p, NULL) == IRIS_SUCCESS) {
          Utils::ReadFile(p, src, srclen);
          if (*srclen > 0) {
              _trace("dev[%d][%s] kernels[%s]", devno_, name_, p);
              return IRIS_SUCCESS;
          }
      }
      if (p != NULL) { free(p); p = NULL; }
      if (strcmp("xclbin", fpga_bin_suffix_.c_str()) == 0 && 
              Platform::GetPlatform()->GetFilePath("KERNEL_FPGA_XCLBIN", &p, NULL) == IRIS_SUCCESS) {
          Utils::ReadFile(p, src, srclen);
          if (*srclen > 0) {
              _trace("dev[%d][%s] kernels[%s]", devno_, name_, p);
              return IRIS_SUCCESS;
          }
      }
  }

  char path[256];
  sprintf(path, "kernel-%s.%s",
    type_ == iris_cpu    ? "cpu"    :
    type_ == iris_nvidia ? "nvidia" :
    type_ == iris_amd    ? "amd"    :
    type_ == iris_gpu    ? "gpu"    :
    type_ == iris_phi    ? "phi"    :
    type_ == iris_fpga   ? "fpga"   : "default",
    type_ == iris_fpga   ? fpga_bin_suffix_.c_str() : suffix);
  if (Utils::ReadFile(path, src, srclen) == IRIS_ERROR && type_ != iris_fpga) {
    sprintf(path, "kernel.%s", suffix);
    Utils::ReadFile(path, src, srclen);
  }
  if (*srclen > 0) {
    _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
    return IRIS_SUCCESS;
  }
  if (strcmp("cl", suffix) == 0  && type_ != iris_fpga) {
      _trace("dev[%d][%s] has no kernel file [%s]. Hence, using the default kernel", devno_, name_, path);
      char default_str[] = "\
            __kernel void ____process(__global int *out, int A) {\
                size_t id = get_global_id(0);\
                    out[id] = A;\
            }";
      *src = (char *)malloc(strlen(default_str)+1);
      memcpy(*src, default_str, strlen(default_str)+1);
      return IRIS_SUCCESS;
  }
  return IRIS_ERROR;
}

int DeviceOpenCL::RecreateContext(){
  //for the device to interpret environment variables (such as AIWC) -- setenv(name, value, 1);
  cl_int err;
  clctx_ = ld_->clCreateContext(NULL, 1, &cldev_, NULL, NULL, &err);
  Init();
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


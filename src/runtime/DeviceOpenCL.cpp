#include "DeviceOpenCL.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderOpenCL.h"
#include "Mem.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Utils.h"

namespace brisbane {
namespace rt {

DeviceOpenCL::DeviceOpenCL(LoaderOpenCL* ld, cl_device_id cldev, cl_context clctx, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  cldev_ = cldev;
  clctx_ = clctx;
  clprog_ = NULL;
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_), &max_compute_units_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size_), &max_work_group_size_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes_), max_work_item_sizes_, NULL);
  err_ = ld_->clGetDeviceInfo(cldev_, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available_), &compiler_available_, NULL);
  fpga_bin_suffix_ = "aocx";   

  if (cltype_ == CL_DEVICE_TYPE_CPU) type_ = brisbane_cpu;
  else if (cltype_ == CL_DEVICE_TYPE_GPU) {
    type_ = brisbane_gpu;
    if (strcasestr(vendor_, "NVIDIA")) type_ = brisbane_nvidia;
    else if (strcasestr(vendor_, "Advanced Micro Devices")) type_ = brisbane_amd;
  }
  else if (cltype_ == CL_DEVICE_TYPE_ACCELERATOR) {
    if (strstr(vendor_, "Xilinx") != NULL) { type_ = brisbane_fpga; fpga_bin_suffix_ = "xclbin"; }
    else if (strstr(name_, "FPGA") != NULL || strstr(version_, "FPGA") != NULL) type_ = brisbane_fpga;
    else type_ = brisbane_phi;
  }
  else type_ = brisbane_cpu;
  model_ = brisbane_opencl;

  _info("device[%d] platform[%d] vendor[%s] device[%s] type[0x%x:%d] version[%s] max_compute_units[%zu] max_work_group_size[%zu] max_work_item_sizes[%zu,%zu,%zu] compiler_available[%d]", devno_, platform_, vendor_, name_, type_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], compiler_available_);
}

DeviceOpenCL::~DeviceOpenCL() {
}

int DeviceOpenCL::Init() {
  clcmdq_ = ld_->clCreateCommandQueue(clctx_, cldev_, 0, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;

  cl_int status;
  char* src = NULL;
  size_t len = 0;
  if (CreateProgram("spv", &src, &len) == BRISBANE_OK) {
    if (type_ == brisbane_fpga) clprog_ = ld_->clCreateProgramWithBinary(clctx_, 1, &cldev_, (const size_t*) &len, (const unsigned char**) &src, &status, &err_);
    else clprog_ = ld_->clCreateProgramWithIL(clctx_, (const void*) src, len, &err_);
    _clerror(err_);
    if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  } else if (CreateProgram("cl", &src, &len) == BRISBANE_OK) {
    clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &len, &err_);
    _clerror(err_);
    if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  } else {
    _error("dev[%d][%s] has no kernel file", devno_, name_);
    return BRISBANE_ERR;
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
    return BRISBANE_ERR;
  }
  size_t nkernels;
  ld_->clGetProgramInfo(clprog_, CL_PROGRAM_NUM_KERNELS, sizeof(nkernels), &nkernels, NULL);
  ld_->clGetProgramInfo(clprog_, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &len);
  char* kernel_names = (char*) malloc(len + 1);
  ld_->clGetProgramInfo(clprog_, CL_PROGRAM_KERNEL_NAMES, len + 1, kernel_names, NULL);
  _trace("nkernels[%zu] kernel_names[%s]", nkernels, kernel_names);
  free(kernel_names);
  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceOpenCL::BuildProgram(char* path) {
  if (clprog_) {
    err_ = ld_->clReleaseProgram(clprog_);
    _clerror(err_);
  }

  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR) {
    _error("path[%s]", path);
    return BRISBANE_ERR;
  }
  clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;

  err_ = ld_->clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;

  if (src) free(src);
  return BRISBANE_OK;
}

int DeviceOpenCL::MemAlloc(void** mem, size_t size) {
  cl_mem* clmem = (cl_mem*) mem;
  *clmem = ld_->clCreateBuffer(clctx_, CL_MEM_READ_WRITE, size, NULL, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::MemFree(void* mem) {
  cl_mem clmem = (cl_mem) mem;
  err_ = ld_->clReleaseMemObject(clmem);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::MemH2D(Mem* mem, size_t off, size_t size, void* host) {
  cl_mem clmem = (cl_mem) mem->arch(this);
  err_ = ld_->clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::MemD2H(Mem* mem, size_t off, size_t size, void* host) {
  cl_mem clmem = (cl_mem) mem->arch(this);
  err_ = ld_->clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off, size, host, 0, NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelGet(void** kernel, const char* name) {
  cl_kernel* clkernel = (cl_kernel*) kernel;
  *clkernel = ld_->clCreateKernel(clprog_, name, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  err_ = ld_->clSetKernelArg(clkernel, (cl_uint) idx, size, value);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  cl_mem clmem = (cl_mem) mem->arch(this);
  err_ = ld_->clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  _trace("dev[%d] kernel[%s] dim[%d] gws[%zu,%zu,%zu] lws[%zu,%zu,%zu]", devno_, kernel->name(), dim, gws[0], gws[1], gws[2], lws ? lws[0] : 0, lws ? lws[1] : 0, lws ? lws[2] : 0);
  err_ = ld_->clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
#ifdef BRISBANE_SYNC_EXECUTION
//  err_ = ld_->clFinish(clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
#endif
  return BRISBANE_OK;
}

int DeviceOpenCL::Synchronize() {
  err_ = ld_->clFinish(clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return BRISBANE_ERR;
  return BRISBANE_OK;
}

int DeviceOpenCL::AddCallback(Task* task) {
  task->Complete();
  return BRISBANE_OK;
}

int DeviceOpenCL::CreateProgram(const char* suffix, char** src, size_t* srclen) {
  char* p = NULL;
  if (Platform::GetPlatform()->EnvironmentGet(strcmp("spv", suffix) == 0 ? "KERNEL_BIN_SPV" : "KERNEL_SRC_SPV", &p, NULL) == BRISBANE_OK) {
    Utils::ReadFile(p, src, srclen);
  }

  if (*srclen > 0) {
    _trace("dev[%d][%s] kernels[%s]", devno_, name_, p);
    return BRISBANE_OK;
  }

  char path[256];
  sprintf(path, "kernel-%s.%s",
    type_ == brisbane_cpu    ? "cpu"    :
    type_ == brisbane_nvidia ? "nvidia" :
    type_ == brisbane_amd    ? "amd"    :
    type_ == brisbane_gpu    ? "gpu"    :
    type_ == brisbane_phi    ? "phi"    :
    type_ == brisbane_fpga   ? "fpga"   : "default",
    type_ == brisbane_fpga   ? fpga_bin_suffix_.c_str() : suffix);
  if (Utils::ReadFile(path, src, srclen) == BRISBANE_ERR && type_ != brisbane_fpga) {
    sprintf(path, "kernel.%s", suffix);
    Utils::ReadFile(path, src, srclen);
  }
  if (*srclen > 0) {
    _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
    return BRISBANE_OK;
  }
  return BRISBANE_ERR;
}

int DeviceOpenCL::RecreateContext(){
  //for the device to interpret environment variables (such as AIWC) -- setenv(name, value, 1);
  cl_int err;
  clctx_ = ld_->clCreateContext(NULL, 1, &cldev_, NULL, NULL, &err);
  Init();
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


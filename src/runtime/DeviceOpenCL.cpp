#include "DeviceOpenCL.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderOpenCL.h"
#include "LoaderHost2OpenCL.h"
#include "Mem.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Utils.h"

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
DeviceOpenCL::DeviceOpenCL(LoaderOpenCL* ld, LoaderHost2OpenCL *host2opencl_ld, cl_device_id cldev, cl_context clctx, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
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
}

int DeviceOpenCL::Init() {
  clcmdq_ = ld_->clCreateCommandQueue(clctx_, cldev_, 0, &err_);
  if (host2opencl_ld_->iris_host2opencl_init)
      host2opencl_ld_->iris_host2opencl_init();
  if (host2opencl_ld_->iris_host2opencl_set_handle)
      host2opencl_ld_->iris_host2opencl_set_handle(&clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;

  cl_int status;
  char* src = NULL;
  size_t len = 0;
  if (CreateProgram("spv", &src, &len) == IRIS_SUCCESS) {
    if (type_ == iris_fpga) clprog_ = ld_->clCreateProgramWithBinary(clctx_, 1, &cldev_, (const size_t*) &len, (const unsigned char**) &src, &status, &err_);
    else clprog_ = ld_->clCreateProgramWithIL(clctx_, (const void*) src, len, &err_);
    _clerror(err_);
    if (err_ != CL_SUCCESS) return IRIS_ERROR;
  } else if (CreateProgram("cl", &src, &len) == IRIS_SUCCESS) {
    clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &len, &err_);
    _clerror(err_);
    if (err_ != CL_SUCCESS) return IRIS_ERROR;
  } else {
    _error("dev[%d][%s] has no kernel file", devno_, name_);
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
    return IRIS_ERROR;
  }
  size_t nkernels;
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
    return IRIS_ERROR;
  }
  clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;

  err_ = ld_->clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;

  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemAlloc(void** mem, size_t size) {
  cl_mem* clmem = (cl_mem*) mem;
  *clmem = ld_->clCreateBuffer(clctx_, CL_MEM_READ_WRITE, size, NULL, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemFree(void* mem) {
  cl_mem clmem = (cl_mem) mem;
  err_ = ld_->clReleaseMemObject(clmem);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemH2D(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host) {
  cl_mem clmem = (cl_mem) mem->arch(this);
  if (dim == 2 || dim ==3) {
      size_t host_row_pitch = elem_size * host_sizes[0];
      size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
      size_t dev_row_pitch = elem_size * dev_sizes[0];
      size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
      size_t buffer_origin[3] = { 0, 0, 0};
      size_t host_origin[3] = {off[0] * elem_size, off[1], off[2]};
      size_t region[3] = { dev_sizes[0]*elem_size, dev_sizes[1], dev_sizes[2] };
      err_ = ld_->clEnqueueWriteBufferRect(clcmdq_, clmem, CL_TRUE, buffer_origin, host_origin, region, dev_row_pitch, dev_slice_pitch, host_row_pitch, host_slice_pitch, host, 0, NULL, NULL);
  }
  else
      err_ = ld_->clEnqueueWriteBuffer(clcmdq_, clmem, CL_TRUE, off[0], size, host, 0, NULL, NULL);
  _clerror(err_);
  _trace("dev[%d][%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", devno_, name_, mem->uid(), clmem, off[0], size, host, q_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemD2H(Mem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host) {
  cl_mem clmem = (cl_mem) mem->arch(this);
  if (dim == 2 || dim ==3) {
      size_t host_row_pitch = elem_size * host_sizes[0];
      size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
      size_t dev_row_pitch = elem_size * dev_sizes[0];
      size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
      size_t buffer_origin[3] = { 0, 0, 0};
      size_t host_origin[3] = {off[0] * elem_size, off[1], off[2]};
      size_t region[3] = { dev_sizes[0]*elem_size, dev_sizes[1], dev_sizes[2] };
      err_ = ld_->clEnqueueReadBufferRect(clcmdq_, clmem, CL_TRUE, buffer_origin, host_origin, region, dev_row_pitch, dev_slice_pitch, host_row_pitch, host_slice_pitch, host, 0, NULL, NULL);
  }
  else {
      err_ = ld_->clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, off[0], size, host, 0, NULL, NULL);
  }
  _clerror(err_);
  _trace("dev[%d][%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", devno_, name_, mem->uid(), clmem, off[0], size, host, q_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelGet(void** kernel, const char* name) {
  if (is_vendor_specific_kernel() && host2opencl_ld_->iris_host2opencl_kernel) 
        return IRIS_SUCCESS;
  cl_kernel* clkernel = (cl_kernel*) kernel;
  *clkernel = ld_->clCreateKernel(clprog_, name, &err_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelSetArg(Kernel* kernel, int idx, size_t size, void* value) {
  if (is_vendor_specific_kernel() && host2opencl_ld_->iris_host2opencl_setarg)
      host2opencl_ld_->iris_host2opencl_setarg(idx, size, value);
  else {
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  err_ = ld_->clSetKernelArg(clkernel, (cl_uint) idx, size, value);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelSetMem(Kernel* kernel, int idx, Mem* mem, size_t off) {
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  cl_mem clmem = (cl_mem) mem->arch(this);
  if (is_vendor_specific_kernel() && host2opencl_ld_->iris_host2opencl_setmem)
      host2opencl_ld_->iris_host2opencl_setmem(idx, clmem);
  else {
  err_ = ld_->clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelLaunchInit(Kernel* kernel) {
    set_vendor_specific_kernel(false);
    if (host2opencl_ld_->iris_host2opencl_kernel)
        if (host2opencl_ld_->iris_host2opencl_kernel(kernel->name()) == IRIS_SUCCESS)
            set_vendor_specific_kernel(true);
    return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  _trace("kernel[%s] dim[%d] gws[%zu,%zu,%zu] lws[%zu,%zu,%zu]", kernel->name(), dim, gws[0], gws[1], gws[2], lws ? lws[0] : 0, lws ? lws[1] : 0, lws ? lws[2] : 0);
  if (is_vendor_specific_kernel() && host2opencl_ld_->iris_host2opencl_launch) {
      host2opencl_ld_->iris_host2opencl_launch(dim, off[0], gws[0]);
      return IRIS_SUCCESS; 
  }
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  err_ = ld_->clEnqueueNDRangeKernel(clcmdq_, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) lws, 0, NULL, NULL);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
#ifdef IRIS_SYNC_EXECUTION
//  err_ = ld_->clFinish(clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
#endif
  return IRIS_SUCCESS;
}

int DeviceOpenCL::Synchronize() {
  err_ = ld_->clFinish(clcmdq_);
  _clerror(err_);
  if (err_ != CL_SUCCESS) return IRIS_ERROR;
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
  size_t gws0 = gws[0];
  size_t* lws = cmd->lws();
  bool reduction = false;
  iris_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  int max_idx = 0;
  int mem_idx = 0;
  set_vendor_specific_kernel(false);
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
    Mem* mem = arg->mem;
    if (mem) {
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
        KernelSetMem(kernel, arg_idx, mem, arg->off);
        KernelSetArg(kernel, arg_idx + 1, lws[0] * mem->type_size(), NULL);
        reduction = true;
        if (idx + 1 > max_idx) max_idx = idx + 1;
        idx++;
        arg_idx+=2;
      } else {
          KernelSetMem(kernel, arg_idx, mem, arg->off);
          arg_idx+=1;
      }
      mem_idx++;
    } else {
        KernelSetArg(kernel, arg_idx, arg->size, arg->value);
        arg_idx+=1;
    }
  }
#if 0
  if (reduction) {
    _trace("max_idx+1[%d] gws[%lu]", max_idx + 1, gws0);
    KernelSetArg(kernel, max_idx + 1, sizeof(size_t), &gws0);
  }
#endif
  errid_ = KernelLaunch(kernel, dim, off, gws, lws[0] > 0 ? lws : NULL);
  double time = timer_->Stop(IRIS_TIMER_KERNEL);
  cmd->SetTime(time);
  cmd->kernel()->history()->AddKernel(cmd, this, time);
}

int DeviceOpenCL::CreateProgram(const char* suffix, char** src, size_t* srclen) {
  char* p = NULL;
  if (Platform::GetPlatform()->EnvironmentGet(strcmp("spv", suffix) == 0 ? "KERNEL_BIN_SPV" : "KERNEL_SRC_SPV", &p, NULL) == IRIS_SUCCESS) {
    Utils::ReadFile(p, src, srclen);
  }

  if (*srclen > 0) {
    _trace("dev[%d][%s] kernels[%s]", devno_, name_, p);
    return IRIS_SUCCESS;
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
            __kernel void process(__global int *out, int A) {\
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


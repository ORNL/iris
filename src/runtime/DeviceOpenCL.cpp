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
    //cl_int err;
    ld->clGetDeviceInfo(cldev, CL_DEVICE_TYPE, sizeof(cltype), &cltype, NULL);
    ld->clGetDeviceInfo(cldev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    ld->clGetDeviceInfo(cldev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    ld->clGetDeviceInfo(cldev, CL_DEVICE_VERSION, sizeof(version), version, NULL);
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
  set_async(true && Platform::GetPlatform()->is_async()); 
  ocldevno_ = ocldevno;
  host2opencl_ld_ = host2opencl_ld;
  cldev_ = cldev;
  clctx_ = clctx;
  clprog_ = NULL;
  timer_ = new Timer();
  //cl_int err;
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_VENDOR, sizeof(vendor_), vendor_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_NAME, sizeof(name_), name_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_TYPE, sizeof(cltype_), &cltype_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_VERSION, sizeof(version_), version_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_), &max_compute_units_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size_), &max_work_group_size_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes_), max_work_item_sizes_, NULL);
  ld_->clGetDeviceInfo(cldev_, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available_), &compiler_available_, NULL);
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
  clcmdq_ = new cl_command_queue[nqueues_];
  for (int i = 0; i < nqueues_; i++) {
    clcmdq_[i] = NULL;
  }
  default_queue_ = NULL;
  single_start_time_event_ = NULL;
  if (IsDeviceValid()) {
      _info("device[%d] platform[%d] vendor[%s] device[%s] type[0x%x:%d] version[%s] max_compute_units[%zu] max_work_group_size[%zu] max_work_item_sizes[%zu,%zu,%zu] compiler_available[%d]", devno_, platform_, vendor_, name_, type_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], compiler_available_);
  }
}

DeviceOpenCL::~DeviceOpenCL() {
    host2opencl_ld_->finalize(ocldevno_);
    for (int i = 0; i < nqueues_; i++) {
      if (clcmdq_[i]) {
          cl_int err = ld_->clReleaseCommandQueue(clcmdq_[i]);
          _clerror(err);
      }
    }
    delete timer_;
    delete [] clcmdq_;
    if (default_queue_) {
        cl_int err = ld_->clReleaseCommandQueue(default_queue_);
        _clerror(err);
    }
    if (is_async(false) && platform_obj_->is_event_profile_enabled()) 
        DestroyEvent(single_start_time_event_);
    cl_int err;
    if (clprog_) {
        err = ld_->clReleaseProgram(clprog_);
        _clerror(err);
    }
    if (this == root_device()) {
        // Context is shared across different opencl devices
        err = ld_->clReleaseContext(clctx_);
        _clerror(err);
    }
}

int DeviceOpenCL::Init() {
  cl_int err;
  if (is_async(false)) {
      for (int i = 0; i < nqueues_; i++) {
#ifdef CL_VERSION_2_0
          const cl_queue_properties props[] = {
                  //CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                  CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                      0 // Terminate the properties list
          };
          if (ld_->clCreateCommandQueueWithProperties == NULL || type_ == iris_fpga)
              clcmdq_[i] = ld_->clCreateCommandQueue(clctx_, cldev_, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
          else
              clcmdq_[i] = ld_->clCreateCommandQueueWithProperties(clctx_, cldev_, props, &err);
#else
          clcmdq_[i] = ld_->clCreateCommandQueue(clctx_, cldev_, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
#endif
          _clerror(err);
      }
      set_first_event_cpu_begin_time(timer_->Now());
      RecordEvent((void **)(&single_start_time_event_), 0, iris_event_default);
      set_first_event_cpu_end_time(timer_->Now());
      _event_prof_debug("Event start time of device:%f end time of record:%f", first_event_cpu_begin_time(), first_event_cpu_end_time());
  }
  else {
      nqueues_ = 0;
  }
#ifdef CL_VERSION_2_0
  const cl_queue_properties props[] = {
          CL_QUEUE_PROPERTIES, CL_QUEUE_ON_DEVICE_DEFAULT,
              0 // Terminate the properties list
  };
  //printf("clCreateCommandQueue:%p clCreateCommandQueueWithProperties:%p\n", ld_->clCreateCommandQueue, ld_->clCreateCommandQueueWithProperties);
  if (ld_->clCreateCommandQueueWithProperties == NULL || type_ == iris_fpga)
      default_queue_ = ld_->clCreateCommandQueue(clctx_, cldev_, 0, &err);
  else
      default_queue_ = ld_->clCreateCommandQueueWithProperties(clctx_, cldev_, NULL, &err);
#else
  default_queue_ = ld_->clCreateCommandQueue(clctx_, cldev_, 0, &err);
#endif
  _clerror(err);
  host2opencl_ld_->init(ocldevno_);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }

  cl_int status;
  char* src = NULL;
  size_t len = 0;
  if (CreateProgram("spv", &src, &len) == IRIS_SUCCESS) {
    if (type_ == iris_fpga) clprog_ = ld_->clCreateProgramWithBinary(clctx_, 1, &cldev_, (const size_t*) &len, (const unsigned char**) &src, &status, &err);
    else clprog_ = ld_->clCreateProgramWithIL(clctx_, (const void*) src, len, &err);
    _clerror(err);
    if (err != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  } else if (CreateProgram("cl", &src, &len) == IRIS_SUCCESS) {
    if (type_ == iris_fpga) {
        _error("dev[%d][%s] has no binary kernel file", devno_, name_);
        return IRIS_SUCCESS;
    }
    clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &len, &err);
    _clerror(err);
    if (err != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  } else {
    _error("dev[%d][%s] has no kernel file", devno_, name_);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  err = ld_->clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err);
  if (err != CL_SUCCESS) {
    cl_build_status s;
    err = ld_->clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_STATUS, sizeof(s), &s, NULL);
    _clerror(err);
    err = ld_->clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* log = (char*) malloc(len + 1);
    err = ld_->clGetProgramBuildInfo(clprog_, cldev_, CL_PROGRAM_BUILD_LOG, len + 1, log, NULL);
    _clerror(err);
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
  cl_int err;
  if (clprog_) {
    err = ld_->clReleaseProgram(clprog_);
    _clerror(err);
  }

  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
    _error("path[%s]", path);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  clprog_ = ld_->clCreateProgramWithSource(clctx_, 1, (const char**) &src, (const size_t*) &srclen, &err);
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }

  err = ld_->clBuildProgram(clprog_, 1, &cldev_, "", NULL, NULL);
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }

  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceOpenCL::ResetMemory(Task *task, Command *cmd, BaseMem *mem) {
    cl_mem clmem = (cl_mem) mem->arch(this);
    int stream_index = 0;
    cl_command_queue queue = default_queue_;
    if (is_async(task)) {
        stream_index = GetStream(task); //task->uid() % nqueues_; 
        if (stream_index == DEFAULT_STREAM_INDEX) { stream_index = 0; }
        queue = clcmdq_[stream_index];
    }
    cl_int err;
    ResetData & reset_data = cmd->reset_data();
    uint8_t reset_value = reset_data.value_.u8;
    int value_to_fill = (int)reset_value;
    err = ld_->clEnqueueFillBuffer(queue, clmem, &value_to_fill, sizeof(uint8_t), 0, mem->size(), 0, NULL, NULL);
    _clerror(err);
    if (mem->reset_data().reset_type_ != iris_reset_memset) {
        _error("Reset memory is not implemented yet !");
    } 
    return IRIS_ERROR;
}

int DeviceOpenCL::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
  cl_int err;
  cl_mem* clmem = (cl_mem*) mem_addr;
  *clmem = ld_->clCreateBuffer(clctx_, CL_MEM_READ_WRITE, size, NULL, &err);
  if (reset) {
    _error("OpenCL not supported with reset for size:%lu", size);
  }
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemFree(BaseMem *mem, void* mem_addr) {
  cl_mem clmem = (cl_mem) mem_addr;
  cl_int err = ld_->clReleaseMemObject(clmem);
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  cl_mem clmem = (cl_mem) mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
  int stream_index = 0;
  cl_command_queue queue = default_queue_;
  //bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      //async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { /*async = false;*/ stream_index = 0; }
      queue = clcmdq_[stream_index];
  }
  cl_int err;
  if (dim == 2 || dim ==3) {
      size_t host_row_pitch = elem_size * host_sizes[0];
      size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
      size_t dev_row_pitch = elem_size * dev_sizes[0];
      size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
      size_t buffer_origin[3] = { 0, 0, 0};
      size_t host_origin[3] = {off[0] * elem_size, off[1], off[2]};
      size_t region[3] = { dev_sizes[0]*elem_size, dev_sizes[1], dev_sizes[2] };
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      err = ld_->clEnqueueWriteBufferRect(queue, clmem, CL_TRUE, buffer_origin, host_origin, region, dev_row_pitch, dev_slice_pitch, host_row_pitch, host_slice_pitch, host, 0, NULL, NULL);
#if 0
      float *hostA = new float[dev_sizes[0] * dev_sizes[1]];
      int SIZE = dev_sizes[0]*dev_sizes[1];
      printf("dev[%d] OFF:(%d,%d,%d) DEV:(%d,%d,%d) HOST:(%d,%d,%d) ELEM:%d\n", devno_, off[0], off[1], off[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], host_sizes[0], host_sizes[1], host_sizes[2], elem_size);
      err = ld_->clEnqueueReadBuffer(clcmdq_, clmem, CL_TRUE, 0, dev_sizes[0]*dev_sizes[1]*elem_size, hostA, 0, NULL, NULL);
      int print_size = (SIZE > 8) ? 8: SIZE;
      printf("H2DOffset: dev:%d hostA=\n", devno_);
      for(int i=0; i<print_size; i++) {
          printf("%10.1lf ", hostA[i]);
      }
      printf("\n");
#endif
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], size, host, stream_index);
      err = ld_->clEnqueueWriteBuffer(queue, clmem, CL_TRUE, 0, size, (uint8_t *)host+off[0]*elem_size, 0, NULL, NULL);
#if 0
      printf("H2D: Dev%d: ", devno_);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  cl_mem clmem = (cl_mem) mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
  int stream_index = 0;
  //bool async = false;
  cl_command_queue queue = default_queue_;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      //async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { /*async = false;*/ stream_index = 0; }
      queue = clcmdq_[stream_index];
  }
  cl_int err;
  if (dim == 2 || dim ==3) {
      size_t host_row_pitch = elem_size * host_sizes[0];
      size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
      size_t dev_row_pitch = elem_size * dev_sizes[0];
      size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
      size_t buffer_origin[3] = { 0, 0, 0};
      size_t host_origin[3] = {off[0] * elem_size, off[1], off[2]};
      size_t region[3] = { dev_sizes[0]*elem_size, dev_sizes[1], dev_sizes[2] };
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      err = ld_->clEnqueueReadBufferRect(queue, clmem, CL_TRUE, buffer_origin, host_origin, region, dev_row_pitch, dev_slice_pitch, host_row_pitch, host_slice_pitch, host, 0, NULL, NULL);
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d] ref_cnt[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), clmem, off[0], size, host, stream_index, task->ref_cnt());
      err = ld_->clEnqueueReadBuffer(queue, clmem, CL_TRUE, 0, size, (uint8_t *)host+off[0]*elem_size, 0, NULL, NULL);
#if 0
      printf("D2H: Dev:%d: ", devno_);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
  }
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  cl_int err;
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  int kernel_idx = -1;
  if (kernel->is_vendor_specific_kernel(devno_) && host2opencl_ld_->host_kernel(&kernel_idx, name) == IRIS_SUCCESS) {
      *kernel_bin = host2opencl_ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (clprog_ == NULL) 
      return IRIS_ERROR;
  //_trace("dev[%d][%s] kernel[%s:%s] kernel-get-3", devno_, name_, kernel->name(), kernel->get_task_name());
  cl_kernel* clkernel = (cl_kernel*) kernel_bin;
  *clkernel = ld_->clCreateKernel(clprog_, name, &err);
  if (report_error) _clerror(err);
  if (err != CL_SUCCESS){
    if (report_error) worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  if (kernel->is_vendor_specific_kernel(devno_)) {
     host2opencl_ld_->setarg(
            kernel->GetParamWrapperMemory(), kindex, size, value);
  }
  else {
    cl_kernel clkernel = (cl_kernel) kernel->arch(this);
    cl_int err = ld_->clSetKernelArg(clkernel, (cl_uint) idx, size, value);
    _clerror(err);
    if (err != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  cl_int err;
  void **dev_alloc_ptr = mem->arch_ptr(this);
  void *dev_ptr = NULL;
  //void *param;
  if (off) {
      //TODO: Use sub-buffers here. Otherwise, it wouldn't work.
      *(mem->archs_off() + devno_) = (void*) ((uint8_t *) *dev_alloc_ptr + off);
      //param = mem->archs_off() + devno_;
      dev_ptr = *(mem->archs_off() + devno_);
  } else {
      //param = dev_alloc_ptr;
      dev_ptr = *dev_alloc_ptr; 
  }
  size_t size = mem->size() - off;
  _debug2("task:%lu:%s idx:%d::%d off:%lu dev_ptr:%p dev_alloc_ptr:%p", 
          kernel->task()->uid(), kernel->task()->name(),
          idx, kindex, off, dev_ptr, dev_alloc_ptr);
  if (kernel->is_vendor_specific_kernel(devno_)) {
      host2opencl_ld_->setmem(
              kernel->GetParamWrapperMemory(), kindex, dev_ptr, size);
  }
  else {
    cl_kernel clkernel = (cl_kernel) kernel->arch(this);
    cl_mem clmem = (cl_mem) dev_ptr;
    err = ld_->clSetKernelArg(clkernel, (cl_uint) idx, sizeof(clmem), (const void*) &clmem);
    _clerror(err);
    if (err != CL_SUCCESS){
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
    }
  }
  return IRIS_SUCCESS;
}

void DeviceOpenCL::CheckVendorSpecificKernel(Kernel *kernel) {
    kernel->set_vendor_specific_kernel(devno_, false);
    if (host2opencl_ld_->host_kernel(kernel->GetParamWrapperMemory(), kernel->name())==IRIS_SUCCESS) {
            kernel->set_vendor_specific_kernel(devno_, true);
    }
    kernel->set_vendor_specific_kernel_check(devno_, true);
}
int DeviceOpenCL::KernelLaunchInit(Command *cmd, Kernel* kernel) {
    int stream_index = 0;
    cl_command_queue *kstream = &default_queue_;
    int nstreams = 1;
    if (is_async(kernel->task(), false)) {
        stream_index = GetStream(kernel->task()); //task->uid() % nqueues_; 
        if (stream_index == DEFAULT_STREAM_INDEX) { stream_index = 0; }
        kstream = &clcmdq_[stream_index];
        nstreams = nqueues_ - stream_index;
    }
    host2opencl_ld_->launch_init(model(), ocldevno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
    return IRIS_SUCCESS;
}

int DeviceOpenCL::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  cl_int err;
  int stream_index = 0;
  cl_command_queue *kstream = &default_queue_;
  bool async = false;
  int nstreams = 1;
  if (is_async(kernel->task(), false)) { //Disable stream policy check
      stream_index = GetStream(kernel->task()); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
      // Though async is set to false, we still pass all streams to kernel to use it
      kstream = &clcmdq_[stream_index];
      nstreams = nqueues_ - stream_index;
  }
  _debug2("dev[%d][%s] task[%ld:%s] kernel launch::%ld:%s q[%d]", devno_, name_, kernel->task()->uid(), kernel->task()->name(), kernel->uid(), kernel->name(), stream_index);
  if (kernel->is_vendor_specific_kernel(devno_)) {
     if (host2opencl_ld_->host_launch((void **)kstream, stream_index, nstreams, kernel->name(), 
                 kernel->GetParamWrapperMemory(), ocldevno_,
                 dim, off, gws) == IRIS_SUCCESS) {
         if (!async) {
             err = ld_->clFinish(*kstream);
             _clerror(err);
             if (err != CL_SUCCESS){
                 _error("dev[%d][%s] task[%ld:%s] kernel launch::%ld:%s failed q[%d]", devno_, name_, kernel->task()->uid(), kernel->task()->name(), kernel->uid(), kernel->name(), stream_index);
                 worker_->platform()->IncrementErrorCount();
                 return IRIS_ERROR;
             }
         }
         return IRIS_SUCCESS;
     }
     worker_->platform()->IncrementErrorCount();
     return IRIS_ERROR;
  }

  size_t block[3] = { lws ? lws[0] : 1, lws ?  lws[1] : 1, lws ?  lws[2] : 1 };
  _trace("dev[%d][%s] kernel[%s:%s] dim[%d] gws[%zu,%zu,%zu] lws[%zu,%zu,%zu] off[%zu,%zu,%zu] block[%zu, %zu, %zu]", 
          devno_, name_, kernel->name(), kernel->get_task_name(), 
          dim, gws[0], gws[1], gws[2], 
          lws ? lws[0] : 1, lws ? lws[1] : 1, lws ? lws[2] : 1,
          off ? off[0] : 0, off ? off[1] : 0, off ? off[2] : 0,
          block[0], block[1], block[2]
          );
  cl_kernel clkernel = (cl_kernel) kernel->arch(this);
  err = ld_->clEnqueueNDRangeKernel(*kstream, clkernel, (cl_uint) dim, (const size_t*) off, (const size_t*) gws, (const size_t*) block, 0, NULL, NULL);
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (!async) {
      err = ld_->clFinish(*kstream);
      _clerror(err);
      if (err != CL_SUCCESS){
          worker_->platform()->IncrementErrorCount();
          return IRIS_ERROR;
      }
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::Synchronize() {
  for(int i=0; i<nqueues_; i++) {
      cl_int err = ld_->clFinish(clcmdq_[i]);
      _clerror(err);
      if (err != CL_SUCCESS){
        worker_->platform()->IncrementErrorCount();
        return IRIS_ERROR;
      }
  }
  cl_int err = ld_->clFinish(default_queue_);
  _clerror(err);
  if (err != CL_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceOpenCL::RegisterCallback(int stream, CallBackType callback_fn, void *data, int flags) 
{
    // Create an event for the callback
    cl_event callbackEvent;
    //cl_int err = ld_->clEnqueueMarker(clcmdq_[stream], &callbackEvent);
    cl_int err = ld_->clEnqueueMarkerWithWaitList(clcmdq_[stream], 0, NULL, &callbackEvent);

    // Set up callback function
    ld_->clSetEventCallback(callbackEvent, CL_COMPLETE, (OpenCLCallBack)callback_fn, data);
    return IRIS_SUCCESS;
}
/*
int DeviceOpenCL::AddCallback(Task* task) {
  task->Complete();
  return IRIS_SUCCESS;
}*/

#if 0
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
  KernelLaunchInit(cmd, kernel);
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
#endif

bool DeviceOpenCL::IsDeviceValid() { 
    if (type_ == iris_fpga) {
        char* p = NULL;
        if (type_ == iris_fpga) {
            if (strcmp("aocx", fpga_bin_suffix_.c_str()) == 0 && 
                    Platform::GetPlatform()->GetFilePath("KERNEL_INTEL_AOCX", &p, NULL) == IRIS_SUCCESS) {
                if (Utils::Exist(p)) return true;
            }
            if (p != NULL) { free(p); p = NULL; }
            if (strcmp("xclbin", fpga_bin_suffix_.c_str()) == 0 && 
                    Platform::GetPlatform()->GetFilePath("KERNEL_XILINX_XCLBIN", &p, NULL) == IRIS_SUCCESS) {
                if (Utils::Exist(p)) return true;
            }
            if (p != NULL) { free(p); p = NULL; }
            if (strcmp("xclbin", fpga_bin_suffix_.c_str()) == 0 && 
                    Platform::GetPlatform()->GetFilePath("KERNEL_FPGA_XCLBIN", &p, NULL) == IRIS_SUCCESS) {
                if (Utils::Exist(p)) return true;
            }
        }
        return false;
    }
    return true; 
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

float DeviceOpenCL::GetEventTime(void *event, int stream) 
{ 
    float elapsed=0.0f;
    if (event != NULL) {
        cl_ulong start_time, stop_time;
        cl_int err = ld_->clGetEventProfilingInfo(single_start_time_event_, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        _clerror(err);
        err = ld_->clGetEventProfilingInfo((cl_event)event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &stop_time, NULL);
        _clerror(err);
        elapsed = ((float)(stop_time - start_time))/1e6;
        //printf("Elapsed:%f single_start_time_event:%p event:%p\n", elapsed, single_start_time_event_, event);
        //printf("Elapsed:%f single_start_time_event:%p start_time_event:%p event:%p\n", elapsed, single_start_time_event_, start_time_event_[stream], event);
    }
    return elapsed; 
}
void DeviceOpenCL::CreateEvent(void **event, int flags)
{
    *event = NULL;
}
void DeviceOpenCL::RecordEvent(void **event, int stream, int event_creation_flag)
{
    cl_int err;
    err = ld_->clEnqueueMarker(clcmdq_[stream], (cl_event*)event);
    _clerror(err);
}
void DeviceOpenCL::WaitForEvent(void *event, int stream, int flags)
{
    cl_event event_arr[1];
    event_arr[0] = (cl_event) event;
    cl_int err;
    err = ld_->clEnqueueWaitForEvents(clcmdq_[stream], 1, event_arr);
    _clerror(err);
}
void DeviceOpenCL::DestroyEvent(void *event)
{
    cl_int err = ld_->clReleaseEvent((cl_event)event);
    _clerror(err);
}
void DeviceOpenCL::EventSynchronize(void *event)
{
    cl_event event_arr[1];
    event_arr[0] = (cl_event) event;
    cl_int err;
    err = ld_->clWaitForEvents(1, event_arr);
    _clerror(err);
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


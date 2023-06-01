#include "DeviceCUDA.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "LoaderCUDA.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Worker.h"
#include "Utils.h"

namespace iris {
namespace rt {

void testMemcpy(LoaderCUDA *ld)
{
  int M = 60;
  int N = 70;
  int off_y = 0;
  int off_x = 0;
  int size_y = 60;
  int size_x = 70;
  int *xy = (int *)malloc(M * N * sizeof(int));
  int *y = (int *)malloc(M * N * sizeof(int));
  for(int i=0; i<M; i++) {
    for(int j=0; j<N; j++) {
        xy[i*N+j] = i*1000+j+100;
        y[i*N+j] = 0;
    }
  }
  CUresult err_;
  CUdeviceptr d_xy;

  //cudaMalloc(&d_xy, M*N*sizeof(int)); 
  err_ = ld->cuMemAlloc(&d_xy, M*N*sizeof(int)); 
  _cuerror(err_);

  //cudaMemcpy(d_xy, xy, M*N*sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(y, d_xy, M*N*sizeof(int), cudaMemcpyDeviceToHost);
#if 1
  int width  = size_x;
  int height = size_y;
  int elem_size = sizeof(int);
  err_ = ld->cudaMemcpy2D((void *)d_xy, width*elem_size, xy, width*elem_size, width, height, cudaMemcpyHostToDevice);
  _cuerror(err_);
#else
  err_ = ld->cuMemcpyHtoD(d_xy, xy, M*N*sizeof(int));
  _cuerror(err_);
#endif

  err_ = ld->cuMemcpyDtoH(y, d_xy, M*N*sizeof(int) );
  _cuerror(err_);

  int errors = 0;
  for(int i=off_y; i<off_y+size_y; i++) {
    for(int j=off_x; j<off_x+size_x; j++) {
        if (xy[i*N+j] != y[i*N+j]) errors++;
    }
  }
  #define MIN(X,Y)  ((X) < (Y) ? (X) : (Y))
  printf("Max error: %d\n", errors);
  for (int i=0; i<MIN(size_y*size_x,10); i++) {
    printf("%d:%d ", xy[i], y[i]);
  }
  //cudaFree(d_xy);
  ld->cuMemFree(d_xy);
  free(xy);
  free(y);
}
DeviceCUDA::DeviceCUDA(LoaderCUDA* ld, LoaderHost2CUDA *host2cuda_ld, CUdevice cudev, int devno, int platform) : Device(devno, platform) {
  ld_ = ld;
  host2cuda_ld_ = host2cuda_ld;
  peers_count_ = 0;
  max_arg_idx_ = 0;
  ngarbage_ = 0;
  shared_mem_bytes_ = 0;
  dev_ = cudev;
  strcpy(vendor_, "NVIDIA Corporation");
  enableD2D();
  err_ = ld_->cuDeviceGetName(name_, sizeof(name_), dev_);
  _cuerror(err_);
  type_ = iris_nvidia;
  model_ = iris_cuda;
  err_ = ld_->cuDriverGetVersion(&driver_version_);
  _cuerror(err_);
  //err_ = ld_->cudaSetDevice(dev_);
  _cuerror(err_);
  sprintf(version_, "NVIDIA CUDA %d", driver_version_);
  int tb, mc, bx, by, bz, dx, dy, dz, ck, ae;
  err_ = ld_->cuDeviceGetAttribute(&tb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev_);
  err_ = ld_->cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_);
  err_ = ld_->cuDeviceGetAttribute(&bx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev_);
  err_ = ld_->cuDeviceGetAttribute(&by, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev_);
  err_ = ld_->cuDeviceGetAttribute(&bz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev_);
  err_ = ld_->cuDeviceGetAttribute(&dz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev_);
  err_ = ld_->cuDeviceGetAttribute(&ck, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev_);
  err_ = ld_->cuDeviceGetAttribute(&ae, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev_);
  max_work_group_size_ = tb;
  max_compute_units_ = mc;
  max_block_dims_[0] = bx;
  max_block_dims_[1] = by;
  max_block_dims_[2] = bz;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%zu] max_work_group_size_[%zu] max_work_item_sizes[%zu,%zu,%zu] max_block_dims[%d,%d,%d] concurrent_kernels[%d] async_engines[%d]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], max_block_dims_[0], max_block_dims_[1], max_block_dims_[2], ck, ae);
}

void DeviceCUDA::RegisterPin(void *host, size_t size)
{
    //ld_->cudaHostRegister(host, size, cudaHostRegisterMapped);
    ld_->cudaHostRegister(host, size, cudaHostRegisterDefault);
}

DeviceCUDA::~DeviceCUDA() {
    if (host2cuda_ld_->iris_host2cuda_finalize){
        host2cuda_ld_->iris_host2cuda_finalize();
    }
    if (host2cuda_ld_->iris_host2cuda_finalize_handles){
        host2cuda_ld_->iris_host2cuda_finalize_handles(dev_);
    }
}

int DeviceCUDA::Compile(char* src) {
  char cmd[1024];
  memset(cmd, 0, 256);
  sprintf(cmd, "nvcc -ptx %s -o %s", src, kernel_path_);
  //printf("Cmd: %s\n", cmd);
  if (system(cmd) != EXIT_SUCCESS) {
    _error("cmd[%s]", cmd);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}
void DeviceCUDA::SetPeerDevices(int *peers, int count)
{
    std::copy(peers, peers+count, peers_);
    peers_count_ = count;
}
void DeviceCUDA::EnablePeerAccess()
{
#if 0 
    // It has some performance issues
    for(int i=0; i<peers_count_; i++) {
        CUdevice target_dev = peers_[i];
        if (target_dev == dev_) continue;
        int can_access=0;
        err_ = ld_->cudaDeviceCanAccessPeer(&can_access, dev_, target_dev);
        if (can_access) {
            //printf("Can access dev:%d -> %d = %d\n", dev_, target_dev, can_access);
            err_ = ld_->cudaDeviceEnablePeerAccess(target_dev, 0);
            _cuerror(err_);
        }
    }
#endif
}
int DeviceCUDA::Init() {
  err_ = ld_->cudaSetDevice(dev_);
  err_ = ld_->cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, dev_);
  EnablePeerAccess();
  _cuerror(err_);
#ifndef TRACE_DISABLE
  CUcontext ctx;
  ld_->cuCtxGetCurrent(&ctx);
  _trace("Init:: Context create dev:%d cctx:%p octx:%p self:%p thread:%p", devno_, ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
  if (ctx != ctx_) {
      _trace("Init:: Context wrong for CUDA resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      _trace("Init:: Context wrong for Kernel launch Context Switch: %p %p", ctx, ctx_);
  }
#endif
  //err_ = ld_->cuCtxEnablePeerAccess(ctx_, 0);
  _cuerror(err_);
  for (int i = 0; i < nqueues_; i++) {
    err_ = ld_->cuStreamCreate(streams_ + i, CU_STREAM_DEFAULT);
    _cuerror(err_);
  }
  if (host2cuda_ld_->iris_host2cuda_init != NULL) {
    //c_string_array data = host2cuda_ld_->iris_get_kernel_names();
    //printf("-------K %s\n", data[0]);
    host2cuda_ld_->iris_host2cuda_init();
  }

  if (host2cuda_ld_->iris_host2cuda_init_handles != NULL) {
    //c_string_array data = host2cuda_ld_->iris_get_kernel_names();
    //printf("-------K %s\n", data[0]);
    host2cuda_ld_->iris_host2cuda_init_handles(dev_);
  }

  char* path = kernel_path_;
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
    _trace("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    return IRIS_SUCCESS;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  err_ = ld_->cuModuleLoad(&module_, path);
  if (err_ != CUDA_SUCCESS) {
    _cuerror(err_);
    _error("srclen[%zu] src\n%s", srclen, src);
    if (src) free(src);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (src) free(src);
  return IRIS_SUCCESS;
}

int DeviceCUDA::ResetMemory(BaseMem *mem, uint8_t reset_value) {
    err_ = ld_->cudaMemset(mem->arch(this), reset_value, mem->size());
    _cuerror(err_);
    if (err_ != CUDA_SUCCESS){
       worker_->platform()->IncrementErrorCount();
       return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}
int DeviceCUDA::MemAlloc(void** mem, size_t size, bool reset) {
  if (IsContextChangeRequired()) {
      ld_->cuCtxSetCurrent(ctx_);
  }
  CUdeviceptr* cumem = (CUdeviceptr*) mem;
  //double mtime = timer_->Now();
  err_ = ld_->cuMemAlloc(cumem, size);
  //mtime = timer_->Now() - mtime;
  //printf("CUDA MemAlloc size:%zu ptr:%p time:%f devno:%d\n", size, *cumem, mtime, devno_);
  _cuerror(err_);
  if (reset) ld_->cudaMemset(*mem, 0, size);
  //printf("CUDA Malloc: %p size:%d reset:%d\n", *mem, size, reset);
  if (err_ != CUDA_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::MemFree(void* mem) {
  CUdeviceptr cumem = (CUdeviceptr) mem;
  if (ngarbage_ >= IRIS_MAX_GABAGES) _error("ngarbage[%d]", ngarbage_);
  else garbage_[ngarbage_++] = cumem;
  /*
  _trace("dptr[%p]", cumem);
  err_ = ld_->cuMemFree(cumem);
  _cuerror(err_);
  */
  return IRIS_SUCCESS;
}

void DeviceCUDA::ClearGarbage() {
  if (ngarbage_ == 0) return;
  for (int i = 0; i < ngarbage_; i++) {
    CUdeviceptr cumem = garbage_[i];
    err_ = ld_->cuMemFree(cumem);
    _cuerror(err_);
  }
  ngarbage_ = 0;
}

void DeviceCUDA::MemCpy3D(CUdeviceptr dev, uint8_t *host, size_t *off, 
        size_t *dev_sizes, size_t *host_sizes, 
        size_t elem_size, bool host_2_dev)
{
    size_t host_row_pitch = elem_size * host_sizes[0];
    size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
    size_t dev_row_pitch = elem_size * dev_sizes[0];
    size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
    uint8_t *host_start = host + off[0]*elem_size + off[1] * host_row_pitch + off[2] * host_slice_pitch;
    size_t dev_off[3] = {  0, 0, 0 };
    CUdeviceptr dev_start = dev + dev_off[0] * elem_size + dev_off[1] * dev_row_pitch + dev_off[2] * dev_slice_pitch;
    //printf("Host:%p Dest:%p\n", host_start, dev_start);
    for(size_t i=0; i<dev_sizes[2]; i++) {
        uint8_t *z_host = host_start + i * host_slice_pitch;
        CUdeviceptr z_dev = dev_start + i * dev_slice_pitch;
        for(size_t j=0; j<dev_sizes[1]; j++) {
            uint8_t *y_host = z_host + j * host_row_pitch;
            CUdeviceptr d_dev = z_dev + j * dev_row_pitch;
            if (host_2_dev) {
                //printf("(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                err_ = ld_->cuMemcpyHtoD(d_dev, y_host, dev_sizes[0]*elem_size);
                //_cuerror(err_);
            }
            else {
                //printf("(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                err_ = ld_->cuMemcpyDtoH(y_host, d_dev, dev_sizes[0]*elem_size);
                //_cuerror(err_);
            }
        }
    }
}
int DeviceCUDA::MemD2D(Task *task, BaseMem *mem, void *dst, void *src, size_t size) {
  CUdeviceptr src_cumem = (CUdeviceptr) src;
  CUdeviceptr dst_cumem = (CUdeviceptr) dst;
  if (IsContextChangeRequired()) {
      _trace("CUDA context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->cuCtxSetCurrent(ctx_);
  }
#ifndef IRIS_SYNC_EXECUTION
  q_ = task->uid() % nqueues_; 
  err_ = ld_->cuMemcpyDtoDAsync(dst_cumem, src_cumem, size, streams_[q_]);
#else
  err_ = ld_->cudaMemcpy((void *)dst_cumem, (void *)src_cumem, size, cudaMemcpyDeviceToDevice);
#endif
  _cuerror(err_);
  _trace("dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, size, q_);
  if (err_ != CUDA_SUCCESS) return IRIS_ERROR;
  return IRIS_SUCCESS;
}

int DeviceCUDA::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
#ifndef TRACE_DISABLE
  CUcontext ctx;
  ld_->cuCtxGetCurrent(&ctx);
  _trace("MemH2D:: Context create %sdev[%d][%s] task[%ld:%s] mem[%lu] cctx:%p octx:%p self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
#endif
  if (IsContextChangeRequired()) {
      _trace("CUDA context switch %sdev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->cuCtxSetCurrent(ctx_);
  }
  //testMemcpy(ld_);
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this);
#ifndef IRIS_SYNC_EXECUTION
  q_ = task->uid() % nqueues_; 
#endif
  if (dim == 3) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      MemCpy3D(cumem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
  }
  else if (dim == 2) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
       size_t host_row_pitch = elem_size * host_sizes[0];
       void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
       err_ = ld_->cudaMemcpy2D((void *)cumem, dev_sizes[0]*elem_size, host_start, 
               host_row_pitch, dev_sizes[0]*elem_size, dev_sizes[1], 
               cudaMemcpyHostToDevice);
#if 0
       printf("H2D: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), cumem, host, host_start, elem_size);
       float *A = (float *) host;
       for(int i=0; i<dev_sizes[1]; i++) {
           int ai = off[1] + i;
           for(int j=0; j<dev_sizes[0]; j++) {
               int aj = off[0] + j;
               printf("%10.1lf ", A[ai*host_sizes[1]+aj]);
           }
       }
       printf("\n");
#endif
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], size, host, q_);
#ifdef IRIS_SYNC_EXECUTION
  err_ = ld_->cuMemcpyHtoD(cumem + off[0], host, size);
#if 0
  printf("H2D: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), cumem+off[0], host, host, elem_size);
  float *A = (float *) host;
  for(int i=0; i<size/4; i++) {
      printf("%10.1lf ", A[i]);
  }
  printf("\n");
#endif
#else
  err_ = ld_->cuMemcpyHtoDAsync(cumem + off[0], host, size, streams_[q_]);
#endif
  }
  _trace("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host);
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS){
   worker_->platform()->IncrementErrorCount();
   return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}
bool DeviceCUDA::IsContextChangeRequired() {
    return (worker()->self() != worker()->thread());
}
void DeviceCUDA::SetContextToCurrentThread()
{
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
}
void DeviceCUDA::ResetContext()
{
    CUcontext ctx;
    ld_->cuCtxGetCurrent(&ctx);
    _trace("CUDA resetting context switch dev[%d][%s] self:%p thread:%p", devno_, name_, (void *)worker()->self(), (void *)worker()->thread());
    _trace("Resetting Context Switch: %p %p", ctx, ctx_);
    ld_->cuCtxSetCurrent(ctx_);
}

int DeviceCUDA::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
#ifndef TRACE_DISABLE
  CUcontext ctx;
  ld_->cuCtxGetCurrent(&ctx);
  _trace("MemD2H:: Context create %sdev[%d][%s] task[%ld:%s] mem[%lu] cctx:%p octx:%p self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
#endif
  if (IsContextChangeRequired()) {
      _trace("CUDA context switch %sdev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->cuCtxSetCurrent(ctx_);
  }
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this);
#ifndef IRIS_SYNC_EXECUTION
  q_ = task->uid() % nqueues_; 
#endif
  if (dim == 3) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      MemCpy3D(cumem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, false);
  }
  else if (dim == 2) {
    _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
    size_t host_row_pitch = elem_size * host_sizes[0];
    void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
    err_ = ld_->cudaMemcpy2D((void *)host_start, host_sizes[0]*elem_size, (void*)cumem, 
            dev_sizes[0]*elem_size, dev_sizes[0]*elem_size, dev_sizes[1], 
            cudaMemcpyDeviceToHost);
#if 0
    printf("D2H: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), cumem, host, host_start, elem_size);
    float *A = (float *) host;
    for(int i=0; i<dev_sizes[1]; i++) {
        int ai = off[1] + i;
        for(int j=0; j<dev_sizes[0]; j++) {
            int aj = off[0] + j;
            printf("%10.1lf ", A[ai*host_sizes[1]+aj]);
        }
    }
    printf("\n");
    if (task->uid() == 277) {
        printf("Situation\n");
    }
#endif
  }
  else {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], size, host, q_);
#ifdef IRIS_SYNC_EXECUTION
      err_ = ld_->cuMemcpyDtoH(host, cumem + off[0], size);
#if 0
      printf("D2H: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), cumem+off[0], host, host, elem_size);
      float *A = (float *) host;
      for(int i=0; i<size/4; i++) {
          printf("%10.1lf ", A[i]);
      }
      printf("\n");
#endif
#else
      err_ = ld_->cuMemcpyDtoHAsync(host, cumem + off[0], size, streams_[q_]);
#endif
  }
  if (err_ != CUDA_SUCCESS) {
      _error("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], size, host, q_);
  }
  _trace("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host);
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS){
   worker_->platform()->IncrementErrorCount();
   return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  int kernel_idx=-1;
  if (kernel->is_vendor_specific_kernel(devno_) && 
          host2cuda_ld_->iris_host2cuda_kernel_with_obj &&
      host2cuda_ld_->iris_host2cuda_kernel_with_obj(&kernel_idx, name) == IRIS_SUCCESS)  {
          *kernel_bin = host2cuda_ld_->GetFunctionPtr(name);
          return IRIS_SUCCESS;
  }
  if (kernel->is_vendor_specific_kernel(devno_) && host2cuda_ld_->iris_host2cuda_kernel) {
      *kernel_bin = host2cuda_ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (IsContextChangeRequired()) {
      _trace("Changed Context for CUDA resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->cuCtxSetCurrent(ctx_);
  }
  if (native_kernel_not_exists()) {
      if (report_error) {
          _error("CUDA kernel:%s not found !", name);
          worker_->platform()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  CUfunction* cukernel = (CUfunction*) kernel_bin;
  err_ = ld_->cuModuleGetFunction(cukernel, module_, name);
  if (report_error) _cuerror(err_);
  if (err_ != CUDA_SUCCESS) {
      if (report_error) {
          _error("CUDA kernel:%s not found !", name);
          worker_->platform()->IncrementErrorCount();
      }
      return IRIS_ERROR;
  }
  char name_off[256];
  memset(name_off, 0, sizeof(name_off));
  sprintf(name_off, "%s_with_offsets", name);
  CUfunction cukernel_off;
  err_ = ld_->cuModuleGetFunction(&cukernel_off, module_, name_off);
  if (err_ == CUDA_SUCCESS) {
    kernels_offs_.insert(std::pair<CUfunction, CUfunction>(*cukernel, cukernel_off));
  }

  return IRIS_SUCCESS;
}

int DeviceCUDA::KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value) {
  if (value) params_[idx] = value;
  else {
    shared_mem_offs_[idx] = shared_mem_bytes_;
    params_[idx] = shared_mem_offs_ + idx;
    shared_mem_bytes_ += size;
  }
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  if (kernel->is_vendor_specific_kernel(devno_)) {
     if (host2cuda_ld_->iris_host2cuda_setarg_with_obj)
         host2cuda_ld_->iris_host2cuda_setarg_with_obj(
                kernel->GetParamWrapperMemory(), kindex, size, value);
     else if (host2cuda_ld_->iris_host2cuda_setarg)
         host2cuda_ld_->iris_host2cuda_setarg(kindex, size, value);
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  void **dev_alloc_ptr = mem->arch_ptr(this);
  void *dev_ptr = NULL;
  if (off) {
      *(mem->archs_off() + devno_) = (void*) ((CUdeviceptr) *dev_alloc_ptr + off);
      params_[idx] = mem->archs_off() + devno_;
      dev_ptr = *(mem->archs_off() + devno_);
  } else {
      params_[idx] = dev_alloc_ptr;
      dev_ptr = *dev_alloc_ptr; 
  }
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  if (kernel->is_vendor_specific_kernel(devno_)) {
      if (host2cuda_ld_->iris_host2cuda_setmem_with_obj) 
          host2cuda_ld_->iris_host2cuda_setmem_with_obj(
                  kernel->GetParamWrapperMemory(), kindex, dev_ptr);
      else if (host2cuda_ld_->iris_host2cuda_setmem) 
          host2cuda_ld_->iris_host2cuda_setmem(kindex, dev_ptr);
  }
  return IRIS_SUCCESS;
}

void DeviceCUDA::CheckVendorSpecificKernel(Kernel* kernel) {
    kernel->set_vendor_specific_kernel(devno_, false);
    if (host2cuda_ld_->iris_host2cuda_kernel_with_obj) {
        int status = host2cuda_ld_->iris_host2cuda_kernel_with_obj(
                kernel->GetParamWrapperMemory(), kernel->name());
        if (status == IRIS_SUCCESS && 
                host2cuda_ld_->IsFunctionExists(kernel->name())) {
            kernel->set_vendor_specific_kernel(devno_, true);
        }
    }
    else if (host2cuda_ld_->iris_host2cuda_kernel) {
        int status = host2cuda_ld_->iris_host2cuda_kernel(
                kernel->name());
        if (status == IRIS_SUCCESS && 
                host2cuda_ld_->IsFunctionExists(kernel->name())) {
            kernel->set_vendor_specific_kernel(devno_, true);
        }
    }
    kernel->set_vendor_specific_kernel_check(devno_, true);
}
int DeviceCUDA::KernelLaunchInit(Kernel* kernel) {
    return IRIS_SUCCESS;
}

int DeviceCUDA::KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
#ifndef TRACE_DISABLE
    CUcontext ctx;
    ld_->cuCtxGetCurrent(&ctx);
    _trace("Getting Context for Kernel launch Context Switch: dev:%d cctx:%p octx:%p self:%p thread:%p", devno_, ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
    if (ctx != ctx_) {
        _trace("Context wrong for CUDA resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
        _trace("Context wrong for Kernel launch Context Switch: %p %p", ctx, ctx_);
    }
#endif
#ifndef IRIS_SYNC_EXECUTION
  q_ = cmd->task()->uid() % nqueues_; 
#endif
  if (kernel->is_vendor_specific_kernel(devno_)) {
     if(host2cuda_ld_->iris_host2cuda_launch_with_obj) {
         _trace("dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, q_);
         host2cuda_ld_->SetKernelPtr(kernel->GetParamWrapperMemory(), kernel->name());
         int status = host2cuda_ld_->iris_host2cuda_launch_with_obj(
#ifndef IRIS_SYNC_EXECUTION
                 streams_[q_], 
#endif
                 kernel->GetParamWrapperMemory(), dev_,  
                 dim, off[0], gws[0]);
         err_ = ld_->cuStreamSynchronize(0);
         _cuerror(err_);
         return status;
     }
     else if(host2cuda_ld_->iris_host2cuda_launch) {
         _trace("dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, q_);
         int status = host2cuda_ld_->iris_host2cuda_launch(
#ifndef IRIS_SYNC_EXECUTION
                 streams_[q_], 
#endif
                 dim, off[0], gws[0]);
         err_ = ld_->cuStreamSynchronize(0);
         _cuerror(err_);
         return status;
     }
  }
  _trace("native kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, q_);
  CUfunction cukernel = (CUfunction) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0]) block[0] >>= 1;
  }
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };
  size_t blockOff_x = off[0] / block[0];
  size_t blockOff_y = off[1] / block[1];
  size_t blockOff_z = off[2] / block[2];

  if (off[0] != 0 || off[1] != 0 || off[2] != 0) {
    params_[max_arg_idx_ + 1] = &blockOff_x;
    params_[max_arg_idx_ + 2] = &blockOff_y;
    params_[max_arg_idx_ + 3] = &blockOff_z;
    if (kernels_offs_.find(cukernel) == kernels_offs_.end()) {
      _trace("off0[%lu] cannot find %s_with_offsets kernel. ignore offsets", off[0], kernel->name());
    } else {
      cukernel = kernels_offs_[cukernel];
      _trace("off0[%lu] running %s_with_offsets kernel.", off[0], kernel->name());
    }
  }
  _trace("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, q_);
#ifdef IRIS_SYNC_EXECUTION
  err_ = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
  err_ = ld_->cuStreamSynchronize(0);
#else
  err_ = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, streams_[q_], params_, NULL);
#endif
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS){
   worker_->platform()->IncrementErrorCount();
   return IRIS_ERROR;
  }
  for (int i = 0; i < IRIS_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  return IRIS_SUCCESS;
}

int DeviceCUDA::Synchronize() {
  err_ = ld_->cuCtxSynchronize();
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS){
   worker_->platform()->IncrementErrorCount();
   return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::AddCallback(Task* task) {
  q_ = task->uid() % nqueues_; 
  err_ = ld_->cuStreamAddCallback(streams_[q_], DeviceCUDA::Callback, task, 0);
  _cuerror(err_);
  if (err_ != CUDA_SUCCESS){
   worker_->platform()->IncrementErrorCount();
   return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::Custom(int tag, char* params) {
  if (!cmd_handlers_.count(tag)) {
    _error("unknown tag[0x%x]", tag);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  command_handler handler = cmd_handlers_[tag];
  handler(params, this);
  return IRIS_SUCCESS;
}

void DeviceCUDA::Callback(CUstream stream, CUresult status, void* data) {
  Task* task = (Task*) data;
  task->Complete();
}

void DeviceCUDA::TaskPre(Task* task) {
  ClearGarbage();
}

} /* namespace rt */
} /* namespace iris */


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
#include <array>
#include <string>
#include <vector>
#include <stdexcept>
#include <regex>
#include <sstream>
#include <unordered_map>

using namespace std;

namespace iris {
namespace rt {

class NvidiaTopology {
    public:
        struct NVLinkConnection {
            int gpu1;
            int gpu2;
            std::string linkType;
        };
    private:
        std::vector<NVLinkConnection> connections_;
    public:
        std::vector<NVLinkConnection> connections() { return connections_; }
        NvidiaTopology() {
            std::string topoOutput;
            try {
                topoOutput = exec("nvidia-smi topo -m");
            } catch (const std::exception& e) {
                std::cerr << "Error executing nvidia-smi command: " << e.what() << std::endl;
            }
            parseNVLinkConnections(topoOutput);
        }
        std::string exec(const char* cmd) {
            std::array<char, 128> buffer;
            std::string result;
            std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
            if (!pipe) {
                throw std::runtime_error("popen() failed!");
            }
            while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                result += buffer.data();
            }
            return result;
        }
        void parseNVLinkConnections(std::string topoOutput) {
            std::istringstream stream(topoOutput);
            std::string line;
            std::vector<std::string> headers;
            std::unordered_map<int, std::unordered_map<int, std::string>> topology;
            bool first  = true;
            while (std::getline(stream, line)) {
                line = std::regex_replace(line, std::regex("^.*GPU0"), "GPU0");
                std::istringstream linestream(line);
                std::string token;
                std::vector<std::string> tokens;
                while (linestream >> token) {
                    tokens.push_back(token);
                }

                if (tokens.empty()) {
                    continue;
                }
                std::string first_token = tokens[0].c_str();
                //std::cout << " Line: "<<line<<std::endl;
                if (first && (first_token.compare("GPU0")==0)) {
                    for(std::string tok : tokens) 
                        headers.push_back(tok);
                    first = false;
                } else if (tokens[0].rfind("GPU", 0) == 0) {
                    int gpu1 = std::stoi(tokens[0].substr(3));
                    for (size_t i = 1; i < tokens.size(); ++i) {
                        if (tokens[i].rfind("NV", 0) == 0) {
                            //std::cout <<"GPU"<< i-1 <<" " << tokens[i] << std::endl;
                            int gpu2 = std::stoi(headers[i-1].substr(3));
                            topology[gpu1][gpu2] = tokens[i];
                        }
                    }
                }
            }

            for (const auto& entry1 : topology) {
                for (const auto& entry2 : entry1.second) {
                    connections_.push_back({entry1.first, entry2.first, entry2.second});
                }
            }
#if 0
            std::cout << "NVLink Connections:" << std::endl;
            for (const auto& conn : connections_) {
                std::cout << "GPU " << conn.gpu1 << " <--> GPU " << conn.gpu2 << ": " << conn.linkType << std::endl;
            }
#endif
        }
};
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
  CUresult err;
  CUdeviceptr d_xy;

  //cudaMalloc(&d_xy, M*N*sizeof(int)); 
  err = ld->cuMemAlloc(&d_xy, M*N*sizeof(int)); 
  _cuerror(err);

  //cudaMemcpy(d_xy, xy, M*N*sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(y, d_xy, M*N*sizeof(int), cudaMemcpyDeviceToHost);
#if 1
  int width  = size_x;
  int height = size_y;
  int elem_size = sizeof(int);
  err = ld->cudaMemcpy2D((void *)d_xy, width*elem_size, xy, width*elem_size, width, height, cudaMemcpyHostToDevice);
  _cuerror(err);
#else
  err = ld->cuMemcpyHtoD(d_xy, xy, M*N*sizeof(int));
  _cuerror(err);
#endif

  err = ld->cuMemcpyDtoH(y, d_xy, M*N*sizeof(int) );
  _cuerror(err);

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
DeviceCUDA::DeviceCUDA(LoaderCUDA* ld, LoaderHost2CUDA *host2cuda_ld, CUdevice cudev, int ordinal, int devno, int platform, int local_devno) : Device(devno, platform) {
  ld_ = ld;
  local_devno = local_devno;
  ordinal_ = ordinal;
  set_async(true && Platform::GetPlatform()->is_async()); 
  host2cuda_ld_ = host2cuda_ld;
  peers_count_ = 0;
  max_arg_idx_ = 0;
  ngarbage_ = 0;
  shared_mem_bytes_ = 0;
  dev_ = cudev;
  module_ = (CUmodule)NULL;
  strcpy(vendor_, "NVIDIA Corporation");
#ifndef DISABLE_D2D
  enableD2D();
#endif
  CUresult err = ld_->cuDeviceGetName(name_, sizeof(name_), dev_);
  _cuerror(err);
  bool usm_flag = iris_read_bool_env("CUDA_USM");
  can_share_host_memory_ = usm_flag;
  shared_memory_buffers_ = usm_flag;
  type_ = iris_nvidia;
  model_ = iris_cuda;
  err = ld_->cuDriverGetVersion(&driver_version_);
  _cuerror(err);
  //err = ld_->cudaSetDevice(dev_);
  _cuerror(err);
  sprintf(version_, "NVIDIA CUDA %d", driver_version_);
  int tb, mc, bx, by, bz, dx, dy, dz, ck, ae, nce;
  err = ld_->cuDeviceGetAttribute(&tb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev_);
  err = ld_->cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_);
  err = ld_->cuDeviceGetAttribute(&bx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev_);
  err = ld_->cuDeviceGetAttribute(&by, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev_);
  err = ld_->cuDeviceGetAttribute(&bz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev_);
  err = ld_->cuDeviceGetAttribute(&dx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev_);
  err = ld_->cuDeviceGetAttribute(&dy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev_);
  err = ld_->cuDeviceGetAttribute(&dz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev_);
  err = ld_->cuDeviceGetAttribute(&ck, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev_);
  err = ld_->cuDeviceGetAttribute(&ae, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev_);
  err = ld_->cuDeviceGetAttribute(&nce, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev_);
  //nqueues_ = ae*2;
  //n_copy_engines_ = nce & 0xFF;
  max_work_group_size_ = tb;
  max_compute_units_ = mc;
  max_block_dims_[0] = bx;
  max_block_dims_[1] = by;
  max_block_dims_[2] = bz;
  max_work_item_sizes_[0] = (size_t) bx * (size_t) dx;
  max_work_item_sizes_[1] = (size_t) by * (size_t) dy;
  max_work_item_sizes_[2] = (size_t) bz * (size_t) dz;
  streams_ = new CUstream[nqueues_*2];
  memset(streams_, 0, sizeof(CUstream)*nqueues_*2);
  //memset(start_time_event_, 0, sizeof(CUevent)*IRIS_MAX_DEVICE_NQUEUES);
  single_start_time_event_ = NULL;
  _info("device[%d] platform[%d] vendor[%s] device[%s] type[%d] version[%s] max_compute_units[%zu] max_work_group_size_[%zu] max_work_item_sizes[%zu,%zu,%zu] max_block_dims[%d,%d,%d] concurrent_kernels[%d] async_engines[%d] ncopy_engines[%d]", devno_, platform_, vendor_, name_, type_, version_, max_compute_units_, max_work_group_size_, max_work_item_sizes_[0], max_work_item_sizes_[1], max_work_item_sizes_[2], max_block_dims_[0], max_block_dims_[1], max_block_dims_[2], ck, ae, n_copy_engines_);
}
int DeviceCUDA::CheckPinnedMemory(void* ptr) {
    unsigned int memoryType;
    CUresult result = ld_->cuPointerGetAttribute(&memoryType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)ptr);

    if (result == CUDA_SUCCESS) {
        if (memoryType == CU_MEMORYTYPE_HOST) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return -1;
    }
}
void DeviceCUDA::RegisterPin(void *host, size_t size)
{
    //ld_->cudaHostRegister(host, size, cudaHostRegisterMapped);
    //CUresult err = ld_->cudaHostRegister(host, size, cudaHostRegisterDefault);
    ld_->cuCtxSetCurrent(ctx_);
    //printf("Host:%p size:%lu, flags:%d\n", host, size, 0);
    //int mem_type = CheckPinnedMemory(host);
    //printf("Host:%p size:%lu, flags:%d mem:%d\n", host, size, 0, mem_type);
    if (CheckPinnedMemory(host) != 1) {
        CUresult err = ld_->cuMemHostRegister_v2(host, size, 0);
        _cuwarning(err);
    }
}

void DeviceCUDA::UnRegisterPin(void *host)
{
    //ld_->cudaHostRegister(host, size, cudaHostRegisterMapped);
    //CUresult err = ld_->cudaHostUnregister(host);
    ld_->cuCtxSetCurrent(ctx_);
    if (CheckPinnedMemory(host) == 1) {
        CUresult err = ld_->cuMemHostUnregister(host); 
        _cuwarning(err);
    }
}

DeviceCUDA::~DeviceCUDA() {
    _trace("CUDA device:%d is getting destroyed", devno());
    ld_->cuCtxSetCurrent(ctx_);
    host2cuda_ld_->finalize(devno());
    if (julia_if_ != NULL) julia_if_->finalize(devno());
    for (int i = 0; i < nqueues_; i++) {
      if (streams_[i] != NULL) {
        CUresult err = ld_->cuStreamDestroy(streams_[i]);
        _cuerror(err);
      }
      //DestroyEvent(start_time_event_[i]);
    }
    delete [] streams_;
    if (is_async(false) && platform_obj_->is_event_profile_enabled()) 
        DestroyEvent(single_start_time_event_);
    CUresult err;
    err = ld_->cudaDeviceReset();
    _cuerror(err);
    err = ld_->cuCtxDestroy(ctx_);
    _cuerror(err);
    _trace("CUDA device:%d is destroyed", devno());
}

typedef void *(*SymbolFn)();
typedef void *(*PrintSymbolFn)(void *host_ptr, void *dev_ptr);
void *DeviceCUDA::GetSymbol(const char *name)  { 
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    ASSERT(ld_ != NULL); 
    void *ptr = ld_->GetSymbol(name); 
    if (ptr == NULL) {
        string fn_symbol = name;
        fn_symbol = "get_symbol_"+fn_symbol;
        void *sptr = (void *) host2cuda_ld_->GetSymbol("cData");
        SymbolFn fptr = (SymbolFn) host2cuda_ld_->GetSymbol(fn_symbol.c_str());
        void *dev_var_ptr = fptr();
        //ld_->cudaGetSymbolAddress((void **)&ptr, dev_var_ptr);
        //printf("Symbol read name:%s sptr:%p fptr:%p dev_var_ptr:%p\n", name, sptr, fptr, dev_var_ptr);
        ptr = dev_var_ptr;
    }
    return ptr;
}

int DeviceCUDA::Compile(char* src, const char *out, const char *flags) {
  char default_comp_flags[] = "-ptx";
  char cmd[1024];
  memset(cmd, 0, 256);
  if (flags == NULL) 
      flags = default_comp_flags;
  if (out == NULL) 
      out = kernel_path();
  sprintf(cmd, "nvcc %s -o %s %s > /dev/null 2>&1", src, out, flags);
  //printf("Cmd: %s\n", cmd);
  if (system(cmd) != EXIT_SUCCESS) {
    int result = system("nvcc --version > /dev/null 2>&1");
    if (result == 0) {
        _error("cmd[%s]", cmd);
        worker_->platform()->IncrementErrorCount();
        return IRIS_ERROR;
    }
    else {
        _warning("nvcc is not available for JIT compilation of cmd [%s]", cmd);
        return IRIS_WARNING;
    }
  }
  return IRIS_SUCCESS;
}
bool DeviceCUDA::IsAddrValidForD2D(BaseMem *mem, void *ptr)
{
    int data;
    if (ptr == NULL) return true;
    CUresult err;
    err = ld_->cuCtxSetCurrent(ctx_);
    _cuerror(err);
    err = ld_->cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr) ptr);
    if (err == CUDA_SUCCESS) return true;
    return false;
}
void DeviceCUDA::SetPeerDevices(int *peers, int count)
{
    std::copy(peers, peers+count, peers_);
    peers_count_ = count;
    peer_access_ = new int[peers_count_];
    memset(peer_access_, 0, sizeof(int)*peers_count_);
}
void DeviceCUDA::EnablePeerAccess()
{
#if 1
    CUresult err;
    int offset_dev = devno_ - dev_;
    // It has some performance issues
    static NvidiaTopology topo;
    for (const auto& conn : topo.connections()) {
        if (conn.gpu1 == dev_) 
            peer_access_[conn.gpu2] = 1;
        if (conn.gpu2 == dev_) 
            peer_access_[conn.gpu1] = 1;
    }
    for(int i=0; i<peers_count_; i++) {
        CUdevice target_dev = peers_[i];
        if (target_dev == dev_) {
            peer_access_[i] = 1;
            continue;
        }
        // TODO: NVLink based D2D check is not enabled
        //if (peer_access_[i] != 1) continue;
        err = ld_->cuDeviceCanAccessPeer(&peer_access_[i], dev_, target_dev);
        _cuerror(err);
        int can_access = peer_access_[i];
        if (can_access) {
            DeviceCUDA *target = (DeviceCUDA *)platform_obj_->device(offset_dev + target_dev);
            CUcontext target_ctx = target->ctx_;
            //printf("Can access dev:%d(%d) -> %d(%d) = %d api:%p api1:%p\n", dev_, dev_+offset_dev, target_dev, target_dev+offset_dev, can_access, ld_->cudaDeviceEnablePeerAccess, ld_->cuCtxEnablePeerAccess);
            err = ld_->cuCtxSetCurrent(ctx_);
            _cuerror(err);
            err = ld_->cuCtxEnablePeerAccess(target_ctx, 0);
            //err = ld_->cudaDeviceEnablePeerAccess(target_dev, 0);
            _cuerror(err);
        }
        else {
            //printf("Can not access dev:%d -> %d = %d\n", dev_, target_dev, can_access);
        }
    }
#endif
}
bool DeviceCUDA::IsD2DPossible(Device *target)
{
  if (peer_access_ == NULL) return true;
  if (peer_access_[((DeviceCUDA *)target)->dev_]) return true;
  return false;
}
int DeviceCUDA::Init() {
  CUresult err;
  err = ld_->cudaSetDevice(dev_);
  _cuerror(err);
  err = ld_->cuInit(0);
  _cuerror(err);
  err = ld_->cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, dev_);
  _cuerror(err);
  //err = ld_->cuCtxEnablePeerAccess(ctx_, 0);
  //_cuerror(err);
  //EnablePeerAccess();
  //_printf("Init:: Context create dev:%d ctx:%p self:%p thread:%p", devno_, ctx_, (void *)worker()->self(), (void *)worker()->thread());
#ifndef TRACE_DISABLE
  CUcontext ctx;
  err = ld_->cuCtxGetCurrent(&ctx);
  _cuerror(err);
  _trace("Init:: Context create dev:%d cctx:%p octx:%p self:%p thread:%p", devno_, ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
  if (ctx != ctx_) {
      _trace("Init:: Context wrong for CUDA resetting context switch dev[%d][%s] worker:%d self:%p thread:%p", devno(), name_, worker()->device()->devno(), (void *)worker()->self(), (void *)worker()->thread());
      _trace("Init:: Context wrong for Kernel launch Context Switch: %p %p", ctx, ctx_);
  }
#endif
  //err = ld_->cuCtxEnablePeerAccess(ctx_, 0);
  //_cuerror(err);
  if (is_async(false)) {
      for (int i = 0; i < nqueues_; i++) {
          err = ld_->cuStreamCreate(streams_ + i, CU_STREAM_NON_BLOCKING);
          _cuerror(err);
          if (i < n_copy_engines_) continue;
          streams_[i+nqueues_-n_copy_engines_] = streams_[i];
          //RecordEvent((void **)(start_time_event_+i), i, iris_event_default);
      }
      _event_debug("Number of total streams: %d", nqueues_);
      _event_debug("Number of copy streams: %d", n_copy_engines_);
      _event_debug("Number of kernel streams: %d", nqueues_-n_copy_engines_);
      if (platform_obj_->is_event_profile_enabled()) {
          double start_time = timer_->Now();
          RecordEvent((void **)(&single_start_time_event_), -1, iris_event_default);
          double end_time = timer_->Now();
          set_first_event_cpu_begin_time(start_time);
          set_first_event_cpu_end_time(end_time);
          _event_prof_debug("Event start time of device:%f end time of record:%f", first_event_cpu_begin_time(), first_event_cpu_end_time());
      }
  }
  char flags[128];
  sprintf(flags, "-shared -x cu -g -Xcompiler -fPIC");
  LoadDefaultKernelLibrary("DEFAULT_CUDA_KERNELS", flags);

  host2cuda_ld_->init(devno());
  if (julia_if_ != NULL) julia_if_->init(devno());

  char* path = (char *)kernel_path();
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR) {
    _trace("dev[%d][%s] has no kernel file [%s]", devno_, name_, path);
    //err = ld_->cuMemFree(0);
    //_cuerror(err);
    return IRIS_SUCCESS;
  }
  _trace("dev[%d][%s] kernels[%s]", devno_, name_, path);
  err = ld_->cuModuleLoad(&module_, path);
  _cuerror(err);
  if (err != CUDA_SUCCESS) {
    _cuerror(err);
    _error("srclen[%zu] src\n%s", srclen, src);
    if (src) free(src);
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  if (src) free(src);
  //err = ld_->cuMemFree(0);
  //_cuerror(err);
  return IRIS_SUCCESS;
}

int DeviceCUDA::ResetMemory(Task *task, Command *cmd, BaseMem *mem) {
    CUresult err = CUDA_SUCCESS;
    int stream_index = 0;
    bool async = false;
    if (is_async(task)) {
        stream_index = GetStream(task); //task->uid() % nqueues_; 
        async = true;
        if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
    }
    if (IsContextChangeRequired()) {
        err=ld_->cuCtxSetCurrent(ctx_);
        _cuerror(err);
    }
    ResetData & reset_data = cmd->reset_data();
    if (cmd->reset_data().reset_type_ == iris_reset_memset) {
        uint8_t reset_value = reset_data.value_.u8;
        if (async) 
            err = ld_->cudaMemsetAsync(mem->arch(this), reset_value, mem->size(), streams_[stream_index]);
        else 
            err = ld_->cudaMemset(mem->arch(this), reset_value, mem->size());
        _cuerror(err);
        if (err != CUDA_SUCCESS){
            worker_->platform()->IncrementErrorCount();
            return IRIS_ERROR;
        }
    }
    else if (ld_default() != NULL) {
        pair<bool, int8_t> out = mem->IsResetPossibleWithMemset(reset_data);
        if (out.first) {
            if (async) 
                err = ld_->cudaMemsetAsync(mem->arch(this), out.second, mem->size(), streams_[stream_index]);
            else 
                err = ld_->cudaMemset(mem->arch(this), out.second, mem->size());
            _cuerror(err);
            if (err != CUDA_SUCCESS){
                worker_->platform()->IncrementErrorCount();
                return IRIS_ERROR;
            }
        }
        else if (mem->GetMemHandlerType() == IRIS_DMEM || 
                mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
            size_t elem_size = ((DataMem*)mem)->elem_size();
            if (async)
                CallMemReset(mem, mem->size(), cmd->reset_data(), streams_[stream_index]);
            else
                CallMemReset(mem, mem->size(), cmd->reset_data(), NULL);
        }
        else {
            _error("Unknow reset type for memory:%lu\n", mem->uid());
        }
    }
    else {
        _error("Couldn't find shared library of CUDA dev:%d default kernels with reset APIs", devno()); 
        return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}
void DeviceCUDA::set_can_share_host_memory_flag(bool flag)
{
    CUresult err;
    can_share_host_memory_ = flag;
    err = ld_->cudaSetDeviceFlags(cudaDeviceMapHost);
    _cuerror(err);
}
void *DeviceCUDA::GetSharedMemPtr(void* mem, size_t size) 
{ 
    CUresult err;
    CUdeviceptr* cumem = NULL; // = (CUdeviceptr *)mem;
    err = ld_->cudaHostRegister(mem, size, cudaHostRegisterDefault);
    err = ld_->cudaHostGetDevicePointer((void **)&cumem, mem, 0); 
    _cuerror(err);
    ASSERT(cumem != NULL);
    return cumem; 
}
int DeviceCUDA::MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset) {
  if (IsContextChangeRequired()) {
      CUresult err=ld_->cuCtxSetCurrent(ctx_);
      _cuerror(err);
  }
  CUdeviceptr* cumem = (CUdeviceptr*) mem_addr;
  int stream = mem->recommended_stream(devno());
  bool async = (is_async(false) && stream != DEFAULT_STREAM_INDEX && stream >=0);
  bool l_async = platform_obj_->is_malloc_async() && async && stream >= 0;
  //double mtime = timer_->Now();
  CUresult err;
  if (l_async)
      err = ld_->cuMemAllocAsync(cumem, size, streams_[stream]);
  else
      err = ld_->cuMemAlloc(cumem, size);
  //mtime = timer_->Now() - mtime;
  _cuerror(err);
  if (err != CUDA_SUCCESS){
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
  }
  _event_debug("CUDA MemAlloc dev:%d:%s mem:%lu size:%zu ptr:%p async:%d stream:%d\n", devno_, name_, mem->uid(), size, (void *)*cumem, async, stream);
  if (reset)  {
      if (mem->reset_data().reset_type_ == iris_reset_memset) {
          //printf("Resetting memory\n");
          if (l_async) 
              err = ld_->cudaMemsetAsync((void *)(*cumem), 0, size, streams_[stream]);
          else
              err = ld_->cudaMemset((void *)(*cumem), 0, size);
          _cuerror(err);
          if (err != CUDA_SUCCESS){
              worker_->platform()->IncrementErrorCount();
              return IRIS_ERROR;
          }
      }
      else if (ld_default() != NULL) {
          pair<bool, int8_t> out = mem->IsResetPossibleWithMemset();
          if (out.first) {
              if (l_async) 
                  err = ld_->cudaMemsetAsync((void *)(*cumem), out.second, size, streams_[stream]);
              else
                  err = ld_->cudaMemset((void *)(*cumem), out.second, size);
          }
          else if (mem->GetMemHandlerType() == IRIS_DMEM || 
                  mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
              size_t elem_size = ((DataMem*)mem)->elem_size();
              if (l_async)
                  CallMemReset(mem, size/elem_size, mem->reset_data(), streams_[stream]);
              else
                  CallMemReset(mem, size/elem_size, mem->reset_data(), NULL);
          }
          else {
              _error("Unknow reset type for memory:%lu\n", mem->uid());
          }
      }
      else {
          _error("Couldn't find shared library of CUDA dev:%d default kernels with reset APIs", devno()); 
          return IRIS_ERROR;
      }
  }
  //printf("CUDA Malloc: %p size:%d reset:%d\n", *mem, size, reset);
  return IRIS_SUCCESS;
}

int DeviceCUDA::MemFree(BaseMem *mem, void* mem_addr) {
  CUdeviceptr cumem = (CUdeviceptr) mem_addr;
#ifdef ENABLE_GC_CUDA
  if (ngarbage_ >= IRIS_MAX_GABAGES) _error("ngarbage[%d]", ngarbage_);
  else garbage_[ngarbage_++] = cumem;
#else
  _trace("CUDA Freeing dptr[%p]", (void *)cumem);
  int stream = mem->recommended_stream(devno());
  bool async = (is_async(false) && stream != DEFAULT_STREAM_INDEX && stream >=0);
  CUresult err;
  //if (async)
  //    err = ld_->cuMemFreeAsync(cumem, streams_[stream]);
  //else
      err = ld_->cuMemFree(cumem);
  _cuerror(err);
#endif
  return IRIS_SUCCESS;
}

void DeviceCUDA::ClearGarbage() {
  if (ngarbage_ == 0) return;
  for (int i = 0; i < ngarbage_; i++) {
    CUdeviceptr cumem = garbage_[i];
    //printf("Freeing garbage cumem:%p\n", cumem);
    CUresult err = ld_->cuMemFree(cumem);
    _cuerror(err);
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
                CUresult err = ld_->cuMemcpyHtoD(d_dev, y_host, dev_sizes[0]*elem_size);
                _cuerror(err);
            }
            else {
                //printf("(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                CUresult err = ld_->cuMemcpyDtoH(y_host, d_dev, dev_sizes[0]*elem_size);
                _cuerror(err);
            }
        }
    }
}
int DeviceCUDA::MemD2D(Task *task, Device *src_dev, BaseMem *mem, void *dst, void *src, size_t size) {
  if (mem->is_usm(devno()) || (dst == src) ) return IRIS_SUCCESS;
  CUdeviceptr src_cumem = (CUdeviceptr) src;
  CUdeviceptr dst_cumem = (CUdeviceptr) dst;
  if (IsContextChangeRequired()) {
      _trace("CUDA context switch dev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      CUresult err = ld_->cuCtxSetCurrent(ctx_);
      _cuerror(err);
  }
  bool error_occured = false;
  CUresult err = CUDA_SUCCESS;
  _cuerror(err);
  err = ld_->cuCtxSetCurrent(ctx_);
  int stream_index = 0;
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  _event_debug("D2D dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d] async:%d", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, size, stream_index, async);
  if (async) {
      ASSERT(stream_index >= 0);
      //err = ld_->cuMemcpyAsync((void *)dst_cumem, (void *)src_cumem, size, cudaMemcpyDeviceToDevice, streams_[stream_index]);
      //err = ld_->cuMemcpyDtoDAsync(dst_cumem, src_cumem, size, streams_[stream_index]);
#if 1
      err = ld_->cudaMemcpyAsync((void *)dst_cumem, (void *)src_cumem, size, cudaMemcpyDeviceToDevice, streams_[stream_index]);
#else
      err = ld_->cuMemcpyPeerAsync(dst_cumem, ctx_, src_cumem, ((DeviceCUDA *)src_dev)->ctx_, size, streams_[stream_index]);
#endif
      //printf("cuMemcpyAsync:%p\n", ld_->cuMemcpyAsync);
      //printf("cuMemcpyDtoDAsync:%p\n", ld_->cuMemcpyDtoDAsync);
      _cuerror(err);
      if (err != CUDA_SUCCESS) error_occured = true;
  }
  else {
      //printf("cuMemcpyAsync:%p\n", ld_->cuMemcpyAsync);
      //printf("cuMemcpyDtoDAsync:%p\n", ld_->cuMemcpyDtoDAsync);
      //printf("cuMemcpy:%p\n", ld_->cuMemcpy);
      //err = ld_->cuCtxSetCurrent(ctx_);
      //_cuerror(err);
      //err = ld_->cuMemcpyDtoD(dst_cumem, src_cumem, size);
      err = ld_->cudaMemcpy((void *)dst_cumem, (void *)src_cumem, size, cudaMemcpyDeviceToDevice);
      //err = ld_->cuMemcpyPeer(dst_cumem, ctx_, src_cumem, ((DeviceCUDA *)src_dev)->ctx_, size);
      _cuerror(err);
      if (err != CUDA_SUCCESS) error_occured = true;
  }
  _trace("dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, size, stream_index);
  if (error_occured) {
  _event_debug("Error dev[%d][%s] task[%ld:%s] mem[%lu] dst_dev_ptr[%p] src_dev_ptr[%p] size[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), dst, src, size, stream_index);
  }
  //ASSERT(!error_occured && "CUDA Error occured");
  if (error_occured) {
      worker_->platform()->IncrementErrorCount();
      return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
    bool error_occured = false;
    CUresult err = CUDA_SUCCESS;
#ifndef TRACE_DISABLE
  CUcontext ctx;
  err = ld_->cuCtxGetCurrent(&ctx);
  _cuerror(err);
  _trace("MemH2D:: Context create %sdev[%d][%s] task[%ld:%s] mem[%lu] cctx:%p octx:%p self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
#endif
  if (IsContextChangeRequired()) {
      _trace("CUDA context switch %sdev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->cuCtxSetCurrent(ctx_);
  }
  //testMemcpy(ld_);
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this, host);
  _trace("CUDA %sdev[%d][%s] task[%ld:%s] host_mem:%p dev_mem:%p", tag, devno_, name_, task->uid(), task->name(), host, (void *)cumem);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
  int stream_index = 0;
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  if (dim == 3) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host, stream_index);
      MemCpy3D(cumem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, true);
  }
  else if (dim == 2) {
      _debug2("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host, stream_index);
       size_t host_row_pitch = elem_size * host_sizes[0];
       void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
       if (!async) {
           err = ld_->cudaMemcpy2D((void *)cumem, dev_sizes[0]*elem_size, host_start, 
                   host_row_pitch, dev_sizes[0]*elem_size, dev_sizes[1], 
                   cudaMemcpyHostToDevice);
           _cuerror(err);
           if (err != CUDA_SUCCESS) error_occured = true;
       }
       else {
#if 0
           // Set up the 2D copy parameters
           CUDA_MEMCPY2D copyParams = {0};
           copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
           copyParams.srcHost = host_start;
           copyParams.srcPitch = host_row_pitch; 
           copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
           copyParams.dstDevice = cumem;
           copyParams.dstPitch = dev_sizes[0]*elem_size;
           copyParams.WidthInBytes = dev_sizes[0]*elem_size; 
           copyParams.Height = dev_sizes[1];
           err = ld_->cuCtxSetCurrent(ctx_);
           //err = ld_->cuMemFreeAsync(0, streams_[stream_index]);
           //_cuerror(err);
           err = cuMemcpy2DAsync(&copyParams, streams_[stream_index]);
#else
           err = ld_->cudaMemcpy2DAsync((void *)cumem, dev_sizes[0]*elem_size, host_start, 
                   host_row_pitch, dev_sizes[0]*elem_size, dev_sizes[1], 
                   cudaMemcpyHostToDevice, streams_[stream_index]);
#endif
           _cuerror(err);
           if (err != CUDA_SUCCESS) error_occured = true;
       }
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
      _debug2("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], size, host, stream_index);
      if (!async) {
          err = ld_->cuMemcpyHtoD(cumem, (uint8_t *)host + off[0] * elem_size, size);
          _cuerror(err);
          if (err != CUDA_SUCCESS) error_occured = true;
#if 0
          printf("H2D: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), cumem+off[0], host, host, elem_size);
          float *A = (float *) host;
          for(int i=0; i<size/4; i++) {
              printf("%10.1lf ", A[i]);
          }
          printf("\n");
#endif
      }
      else {
          err = ld_->cuMemcpyHtoDAsync(cumem, (uint8_t *)host + off[0]*elem_size, size, streams_[stream_index]);
          _cuerror(err);
          if (err != CUDA_SUCCESS) error_occured = true;
      }
  }
  _event_debug("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host, stream_index);
  _event_prof_debug("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]\n", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host, stream_index);
  _debug2("Completed H2D DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host, stream_index);
  ASSERT(!error_occured && "CUDA Error occured");
  if (error_occured){
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
    //CUcontext ctx;
    //ld_->cuCtxGetCurrent(&ctx);
    //_trace("CUDA resetting context switch dev[%d][%s] self:%p thread:%p", devno_, name_, (void *)worker()->self(), (void *)worker()->thread());
    //_trace("Resetting Context Switch: %p %p", ctx, ctx_);
    ld_->cuCtxSetCurrent(ctx_);
}

int DeviceCUDA::MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag) {
  bool error_occured = false;
  CUresult err;
#ifndef TRACE_DISABLE
  CUcontext ctx;
  ld_->cuCtxGetCurrent(&ctx);
  _trace("MemD2H:: Context create %sdev[%d][%s] task[%ld:%s] mem[%lu] cctx:%p octx:%p self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), ctx, ctx_, (void *)worker()->self(), (void *)worker()->thread());
#endif
  if (IsContextChangeRequired()) {
      _trace("CUDA context switch %sdev[%d][%s] task[%ld:%s] mem[%lu] self:%p thread:%p", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)worker()->self(), (void *)worker()->thread());
      ld_->cuCtxSetCurrent(ctx_);
  }
  CUdeviceptr cumem = (CUdeviceptr) mem->arch(this, host);
  if (mem->is_usm(devno())) return IRIS_SUCCESS;
  int stream_index = 0;
  bool async = false;
  if (is_async(task)) {
      stream_index = GetStream(task, mem); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
  }
  //printf("D2HRegister callback dev:%d stream:%d\n", devno(), stream_index);
  //CUresult status = ld_->cuStreamQuery(streams_[stream_index]);
  //if (status == CUDA_SUCCESS) {
  //    printf("D2HALL ops are completed\n");
  //}
  //else printf("D2HcuStreamQuery: %d\n", status);
  //printf("mem:[%lu] stream_index:%d devno:%d\n", mem->uid(), stream_index, devno());
  if (dim == 3) {
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
      MemCpy3D(cumem, (uint8_t *)host, off, dev_sizes, host_sizes, elem_size, false);
  }
  else if (dim == 2) {
    _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu,%lu,%lu] host_sizes[%lu,%lu,%lu] dev_sizes[%lu,%lu,%lu] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], off[1], off[2], host_sizes[0], host_sizes[1], host_sizes[2], dev_sizes[0], dev_sizes[1], dev_sizes[2], size, host);
    size_t host_row_pitch = elem_size * host_sizes[0];
    void *host_start = (uint8_t *)host + off[0]*elem_size + off[1] * host_row_pitch;
    if (!async) {
#if 0
        // Set up the 2D copy parameters
        CUDA_MEMCPY2D copyParams = {0};
        copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParams.srcDevice = cumem;
        copyParams.srcPitch = dev_sizes[0]*elem_size;
        copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
        copyParams.dstHost = host_start;
        copyParams.dstPitch = host_row_pitch;
        copyParams.WidthInBytes = dev_sizes[0]*elem_size; 
        copyParams.Height = dev_sizes[1];
        err = ld_->cuCtxSetCurrent(ctx_);
        //err = ld_->cuMemFreeAsync(0, streams_[stream_index]);
        //_cuerror(err);
        err = cuMemcpy2DAsync(&copyParams, streams_[stream_index]);
#else
        err = ld_->cudaMemcpy2D((void *)host_start, host_sizes[0]*elem_size, (void*)cumem, 
                dev_sizes[0]*elem_size, dev_sizes[0]*elem_size, dev_sizes[1], 
                cudaMemcpyDeviceToHost);
#endif
        _cuerror(err);
        if (err != CUDA_SUCCESS) error_occured = true;
    }
    else {
        //printf("-----D2H here---\n");
        err = ld_->cudaMemcpy2DAsync((void *)host_start, host_sizes[0]*elem_size, (void*)cumem, 
                dev_sizes[0]*elem_size, dev_sizes[0]*elem_size, dev_sizes[1], 
                cudaMemcpyDeviceToHost, streams_[stream_index]);
        _cuerror(err);
        //err = ld_->cuStreamSynchronize(streams_[stream_index]);
        if (err != CUDA_SUCCESS) error_occured = true;
    }
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
      _trace("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], size, host, stream_index);
      if (!async) {
          err = ld_->cuMemcpyDtoH((uint8_t *)host + off[0]*elem_size, cumem, size);
          _cuerror(err);
          if (err != CUDA_SUCCESS) error_occured = true;
#if 0
          printf("D2H: %ld:%s mem:%ld dev:%p host:%p host_start:%p elem_size:%lu ", task->uid(), task->name(), mem->uid(), cumem+off[0], host, host, elem_size);
          float *A = (float *) host;
          for(int i=0; i<size/4; i++) {
              printf("%10.1lf ", A[i]);
          }
          printf("\n");
#endif
      } 
      else {
          err = ld_->cuMemcpyDtoHAsync((uint8_t *)host + off[0]*elem_size, cumem, size, streams_[stream_index]);
          _cuerror(err);
          if (err != CUDA_SUCCESS) error_occured = true;
      }
  }
  _event_prof_debug("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host, stream_index);
  _event_debug("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host, stream_index);
  _debug2("Completed D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host, stream_index);
  if (error_occured){
   _debug2("Error D2H DT of %sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] size[%lu] host[%p]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, size, host);
   _error("%sdev[%d][%s] task[%ld:%s] mem[%lu] dptr[%p] off[%lu] size[%lu] host[%p] q[%d]", tag, devno_, name_, task->uid(), task->name(), mem->uid(), (void *)cumem, off[0], size, host, stream_index);
   worker_->platform()->IncrementErrorCount();
   assert(!error_occured && "CUDA Error occured");
   return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error) {
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  int kernel_idx=-1;
  if (kernel->is_vendor_specific_kernel(devno_) && host2cuda_ld_->host_kernel(&kernel_idx, name) == IRIS_SUCCESS) {
      *kernel_bin = host2cuda_ld_->GetFunctionPtr(name);
      return IRIS_SUCCESS;
  }
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
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
  CUresult err = ld_->cuModuleGetFunction(cukernel, module_, name);
  if (report_error) _cuerror(err);
  if (err != CUDA_SUCCESS) {
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
  err = ld_->cuModuleGetFunction(&cukernel_off, module_, name_off);
  if (err == CUDA_SUCCESS) {
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
     host2cuda_ld_->setarg(
            kernel->GetParamWrapperMemory(), kindex, size, value);
  }
  else if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
     julia_if_->setarg(
            kernel->GetParamWrapperMemory(), kindex, size, value);
  }
  return IRIS_SUCCESS;
}

int DeviceCUDA::KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off) {
  void **dev_alloc_ptr = mem->arch_ptr(this);
  void *dev_ptr = NULL;
  size_t size = mem->size() - off;
  if (off) {
      *(mem->archs_off() + devno_) = (void*) ((CUdeviceptr) *dev_alloc_ptr + off);
      params_[idx] = mem->archs_off() + devno_;
      dev_ptr = *(mem->archs_off() + devno_);
  } else {
      params_[idx] = dev_alloc_ptr;
      dev_ptr = *dev_alloc_ptr; 
  }
  _debug2("task:%lu:%s idx:%d::%d off:%lu dev_ptr:%p dev_alloc_ptr:%p", 
          kernel->task()->uid(), kernel->task()->name(),
          idx, kindex, off, dev_ptr, dev_alloc_ptr);
  if (max_arg_idx_ < idx) max_arg_idx_ = idx;
  if (kernel->is_vendor_specific_kernel(devno_)) {
      host2cuda_ld_->setmem(
              kernel->GetParamWrapperMemory(), kindex, dev_ptr, size);
  }
  else if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      julia_if_->setmem(
              kernel->GetParamWrapperMemory(), mem, kindex, dev_ptr, size);
  }
  return IRIS_SUCCESS;
}

void DeviceCUDA::CheckVendorSpecificKernel(Kernel* kernel) {
    kernel->set_vendor_specific_kernel(devno_, false);
    if (host2cuda_ld_->host_kernel(kernel->GetParamWrapperMemory(), kernel->name())==IRIS_SUCCESS) {
            kernel->set_vendor_specific_kernel(devno_, true);
    }
    kernel->set_vendor_specific_kernel_check(devno_, true);
}
int DeviceCUDA::KernelLaunchInit(Command *cmd, Kernel* kernel) {
    int stream_index = 0;
    CUstream *kstream = NULL;
    int nstreams = 0;
    if (is_async(kernel->task(), false)) {
        stream_index = GetStream(kernel->task()); //task->uid() % nqueues_; 
        if (stream_index == DEFAULT_STREAM_INDEX) { stream_index = 0; }
        kstream = &streams_[stream_index];
        //nstreams = nqueues_ - stream_index;
        nstreams = nqueues_-n_copy_engines_;
    }
    host2cuda_ld_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
    if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
        julia_if_->launch_init(model(), devno_, stream_index, nstreams, (void **)kstream, kernel->GetParamWrapperMemory(), cmd);
        julia_if_->set_julia_kernel_type(kernel->GetParamWrapperMemory(), kernel->task()->julia_kernel_type());
    }
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
  if (IsContextChangeRequired()) {
      ld_->cuCtxSetCurrent(ctx_);
  }
  CUresult err;
  int stream_index = 0;
  CUstream *kstream = NULL;
  bool async = false;
  int nstreams = 0;
  if (is_async(kernel->task(), false)) { //Disable stream policy check
      stream_index = GetStream(kernel->task()); //task->uid() % nqueues_; 
      async = true;
      if (stream_index == DEFAULT_STREAM_INDEX) { async = false; stream_index = 0; }
      // Though async is set to false, we still pass all streams to kernel to use it
      kstream = &streams_[stream_index];
      //nstreams = nqueues_ - stream_index;
      nstreams = nqueues_-n_copy_engines_;
  }
  //_debug2("dev[%d][%s] task[%ld:%s] kernel launch::%ld:%s q[%d]", devno_, name_, kernel->task()->uid(), kernel->task()->name(), kernel->uid(), kernel->name(), stream_index);
  _event_prof_debug("kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]\n", devno_, name_, kernel->name(), kernel->get_task_name(), dim, stream_index);
  if (kernel->is_vendor_specific_kernel(devno_)) {
     if (host2cuda_ld_->host_launch((void **)kstream, stream_index, 
                 nstreams, kernel->name(), 
                 kernel->GetParamWrapperMemory(), devno(),
                 dim, off, gws) == IRIS_SUCCESS) {
         if (!async) {
             //err = ld_->cuStreamSynchronize(0);
             err = ld_->cuCtxSynchronize();
             _cuerror(err);
             if (err != CUDA_SUCCESS){
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
  _trace("native kernel start dev[%d][%s] kernel[%s:%s] dim[%d] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, stream_index);
  CUfunction cukernel = (CUfunction) kernel->arch(this);
  int block[3] = { lws ? (int) lws[0] : 1, lws ? (int) lws[1] : 1, lws ? (int) lws[2] : 1 };
  if (!lws) {
    while (max_compute_units_ * block[0] < gws[0]) block[0] <<= 1;
    while (block[0] > max_block_dims_[0]) block[0] >>= 1;
  }
  int grid[3] = { (int) (gws[0] / block[0]), (int) (gws[1] / block[1]), (int) (gws[2] / block[2]) };
  //int grid[3] = { (int) ((gws[0]-off[0]) / block[0]), (int) ((gws[1]-off[1]) / block[1]), (int) ((gws[2]-off[2]) / block[2]) };
  size_t blockOff_x = off[0] / block[0];
  size_t blockOff_y = off[1] / block[1];
  size_t blockOff_z = off[2] / block[2];

  if (off[0] != 0 || off[1] != 0 || off[2] != 0) {
    params_[max_arg_idx_ + 1] = &blockOff_x;
    params_[max_arg_idx_ + 2] = &blockOff_y;
    params_[max_arg_idx_ + 3] = &blockOff_z;
    if (kernels_offs_.find(cukernel) == kernels_offs_.end()) {
      _trace("off0[%lu] cannot find %s_with_offsets kernel. ignore offsets", off[0], kernel->name());
      _error("CUDA kernel name:%s with offset kernel:%s_with_offsets function is not found", kernel->name(), kernel->name());
      worker_->platform()->IncrementErrorCount();
    } else {
      cukernel = kernels_offs_[cukernel];
      _trace("off0[%lu] running %s_with_offsets kernel.", off[0], kernel->name());
    }
  }
  _debug2("off[%lu, %lu, %lu]", off[0], (dim > 1) ? off[1] : 0, (dim > 2) ? off[2] : 0);
  _debug2("lws[%lu, %lu, %lu]", (lws != NULL) ? lws[0] : 1, (lws != NULL && dim > 1) ? lws[1] : 1, (lws != NULL && dim > 2) ? lws[2] : 1);
  _debug2("gws[%lu, %lu, %lu]", gws[0], (dim > 1) ? gws[1] : 0, (dim > 2) ? gws[2] : 0);
  _debug2("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index);
  _event_debug("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] off[%ld,%ld,%ld] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], off[0], off[1], off[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index);
  _trace("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] off[%ld,%ld,%ld] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], off[0], off[1], off[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index);
  /*if (async) {
      _printf("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] off[%ld,%ld,%ld] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d] stream[%p] ctx[%p]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], off[0], off[1], off[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index, *kstream, ctx_);
  }
  else {
      _printf("dev[%d][%s] kernel[%s:%s] dim[%d] grid[%d,%d,%d] off[%ld,%ld,%ld] block[%d,%d,%d] blockoff[%lu,%lu,%lu] max_arg_idx[%d] shared_mem_bytes[%u] q[%d]", devno_, name_, kernel->name(), kernel->get_task_name(), dim, grid[0], grid[1], grid[2], off[0], off[1], off[2], block[0], block[1], block[2], blockOff_x, blockOff_y, blockOff_z, max_arg_idx_, shared_mem_bytes_, stream_index);
  }*/
  if (julia_if_ != NULL && kernel->task()->enable_julia_if()) {
      size_t grid_s[3] =  { (size_t)grid[0],  (size_t)grid[1],  (size_t)grid[2] };
      size_t block_s[3] = { (size_t)block[0], (size_t)block[1], (size_t)block[2] };
      julia_if_->host_launch(kernel->task()->uid(), (void **)kstream, stream_index, (void *)&ctx_, async,
                  nstreams, kernel->name(), 
                  kernel->GetParamWrapperMemory(), ordinal_,
                  dim, grid_s, block_s);
      return IRIS_SUCCESS;
  }
  //printf("Shared mem bytes: %d\n", shared_mem_bytes_);
  if (!async) {
      err = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, 0, params_, NULL);
      _cuerror(err);
      if (err != CUDA_SUCCESS){
          worker_->platform()->IncrementErrorCount();
          return IRIS_ERROR;
      }
      err = ld_->cuStreamSynchronize(0);
      _cuerror(err);
      if (err != CUDA_SUCCESS){
          worker_->platform()->IncrementErrorCount();
          return IRIS_ERROR;
      }
  }
  else {
      err = ld_->cuLaunchKernel(cukernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem_bytes_, streams_[stream_index], params_, NULL);
      _cuerror(err);
      if (err != CUDA_SUCCESS){
          worker_->platform()->IncrementErrorCount();
          return IRIS_ERROR;
      }
  }
  for (int i = 0; i < IRIS_MAX_KERNEL_NARGS; i++) params_[i] = NULL;
  max_arg_idx_ = 0;
  shared_mem_bytes_ = 0;
  return IRIS_SUCCESS;
}
void DeviceCUDA::VendorKernelLaunch(void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params) 
{ 
  printf("IRIS Received kernel:%p stream:%p\n", kernel, stream);
  if (IsContextChangeRequired()) {
      ld_->cuCtxSetCurrent(ctx_);
  }
  //void *par[4] = { (void *)&params[0], (void *)params[1], (void *)&params[2], (void *)&params[3] };
  //printf("Params0: %p %p\n", params[0], par[0]);
  //printf("Params1: %p fl:%f\n", params[1], *((float*)params[1]));
  //printf("Params2: %p %p\n", params[2], par[2]);
  //printf("Params3: %p %p\n", params[3], par[3]);
  CUresult err = ld_->cuLaunchKernel((CUfunction)kernel, gridx, gridy, gridz, blockx, blocky, blockz, shared_mem_bytes, (CUstream)stream, params, NULL);
  _cuerror(err);
  //ld_->cuStreamSynchronize((CUstream)stream);
}

int DeviceCUDA::Synchronize() {
  CUresult err = ld_->cuCtxSynchronize();
  _cuerror(err);
  if (err != CUDA_SUCCESS){
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

int DeviceCUDA::RegisterCallback(int stream, CallBackType callback_fn, void *data, int flags) 
{
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    //TODO: cuStreamAddCallback supports only flags = 0, it is reserved in future for nonblocking
    //printf("Register callback dev:%d stream:%d\n", devno(), stream);
    //CUresult status = ld_->cuStreamQuery(streams_[stream]);
    //if (status == CUDA_SUCCESS) {
    //    printf("ALL ops are completed\n");
    //}
    //else printf("cuStreamQuery: %d\n", status);
    CUresult err = ld_->cuStreamAddCallback(streams_[stream], (CUstreamCallback)callback_fn, data, iris_stream_default);
    _cuerror(err);
    if (err != CUDA_SUCCESS){
        worker_->platform()->IncrementErrorCount();
        return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}

void DeviceCUDA::TaskPre(Task* task) {
  ClearGarbage();
}
void DeviceCUDA::CreateEvent(void **event, int flags)
{
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    CUresult err;
    err = ld_->cuEventCreate((CUevent *)event, flags);   
    _cuerror(err);
    if (err != CUDA_SUCCESS)
        worker_->platform()->IncrementErrorCount();
    _event_debug("Create dev:%d event_ptr:%p event:%p err_id:%d", devno(), event, *event, err);
}
float DeviceCUDA::GetEventTime(void *event, int stream) 
{ 
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    float elapsed=0.0f;
    if (event != NULL) {
        //CUresult err = ld_->cuEventElapsedTime(&elapsed, ((DeviceCUDA *)root_device())->single_start_time_event_, (CUevent)event);
        CUresult err = ld_->cuEventElapsedTime(&elapsed, single_start_time_event_, (CUevent)event);
        _cuerror(err);
        if (err != 0) {
        //_event_prof_debug("Elapsed:%f single_start_time_event:%p start_time_event:%p event:%p\n", elapsed, single_start_time_event_, start_time_event_[stream], event);
        _event_prof_debug("Dev:%d:%s Elapsed:%f single_start_time_event:%p event:%p stream:%d\n", devno(), name(), elapsed, single_start_time_event_, event, stream);
        }
    }
    return elapsed; 
}
void DeviceCUDA::RecordEvent(void **event, int stream, int event_creation_flag)
{
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    if (*event == NULL) 
        CreateEvent(event, event_creation_flag);
    CUresult err;
    CUresult status = ld_->cuEventQuery(*((CUevent *)event));
    if (stream == -1)
        err = ld_->cuEventRecord(*((CUevent *)event), 0);
    else
        err = ld_->cuEventRecord(*((CUevent *)event), streams_[stream]);
    _cuerror(err);
    if (err != CUDA_SUCCESS) {
        worker_->platform()->IncrementErrorCount();
        Utils::PrintStackTrace();
    }
    _event_debug("Recorded dev:[%d]:[%s] event:%p stream:%d err_id:%d", devno(), name(), *event, stream, err);
}
void DeviceCUDA::WaitForEvent(void *event, int stream, int flags)
{
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    CUresult err = ld_->cuStreamWaitEvent(streams_[stream], (CUevent)event, flags);
    _cuerror(err);
    if (err != CUDA_SUCCESS) {
        worker_->platform()->IncrementErrorCount();
        Utils::PrintStackTrace();
    }
}
void DeviceCUDA::DestroyEvent(void *event)
{
    ASSERT(event != NULL && "Event shouldn't be null");
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    //printf("Trying to Destroy dev:%d event:%p\n", devno(), event);
    _event_debug("Destroying dev:%d event:%p ld_:%p cuEventQuery:%p", devno(), event, ld_, ld_->cuEventQuery);
    CUresult err1 = ld_->cuEventQuery((CUevent) event);
    //printf("Query result: %d\n", err1);
    _event_debug("Destroyed dev:%d event:%p", devno(), event);
    CUresult err = ld_->cuEventDestroy((CUevent) event);
    _cuerror(err);
    if (err != CUDA_SUCCESS)
        worker_->platform()->IncrementErrorCount();
}
void DeviceCUDA::EventSynchronize(void *event)
{
    if (IsContextChangeRequired()) {
        ld_->cuCtxSetCurrent(ctx_);
    }
    CUresult err = ld_->cuEventSynchronize((CUevent) event);
    _cuerror(err);
    if (err != CUDA_SUCCESS) {
        worker_->platform()->IncrementErrorCount();
        Utils::PrintStackTrace();
    }
}
} /* namespace rt */
} /* namespace iris */


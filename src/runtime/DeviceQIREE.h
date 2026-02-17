#ifndef IRIS_SRC_RT_DEVICE_QIREE_H
#define IRIS_SRC_RT_DEVICE_QIREE_H

#include "Device.h"
#include "LoaderQIREE.h"
#include "iris/qiree.h"

#define RUN_QIREE_OUT(OUT, DEFAULT, X, ...) \
    if (ld_->X == NULL) { \
        _error(#X " not found in QIREE library"); \
        OUT = DEFAULT; \
    } else { \
        OUT = ld_->X(__VA_ARGS__); \
    }
#define RUN_QIREE(X, ...) \
    if (ld_->X == NULL) { \
        _error(#X " not found in QIREE library"); \
    } else { \
        ASSERT(ld_->X(__VA_ARGS__) == IRIS_SUCCESS); \
    }
#define RUN_QIREE_NO_RETURN(X, ...) \
    if (ld_->X == NULL) { \
        _error(#X " not found in QIREE library"); \
    } else { \
        ld_->X(__VA_ARGS__); \
    }

namespace iris {
namespace rt {

class ProcessQiree;
class DeviceQIREE : public Device {
public:
  DeviceQIREE(LoaderQIREE* ld, int ordinal, int devno, int platform);
  ~DeviceQIREE();
  int Init();
  int ResetMemory(Task *task, Command *cmd, BaseMem *mem);
  int MemAlloc(BaseMem *mem, void** mem_addr, size_t size, bool reset=false);
  int MemFree(BaseMem *mem, void* mem_addr);
  int MemAlloc(void** mem_addr, size_t size, void *stream_v, bool reset=false);
  int MemFree(void* mem_addr, void *stream_v);
  int MemD2D(Task *task, Device *src_dev, BaseMem *mem, void *dst, void *src, size_t size) { _error("Device:%d:%s doesn't support MemD2D", devno_, name()); return IRIS_ERROR; }
  int MemD2D(Task *task, BaseMem *mem, int dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size) { _error("Device:%d:%s doesn't support MemD2D", devno_, name()); return IRIS_ERROR; }
  int MemH2D(Task *task, BaseMem *mem, int dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size) { _error("Device:%d:%s doesn't support MemH2D", devno_, name()); return IRIS_ERROR; }
  int MemD2H(Task *task, BaseMem *mem, int dim, void *dst, size_t *dst_size, size_t *dst_off, size_t *dst_strides, int src_dim, void *src, size_t *src_size, size_t *src_off, size_t *src_strides, size_t *tile_size, size_t elem_size) { _error("Device:%d:%s doesn't support MemD2H", devno_, name()); return IRIS_ERROR; }
  int MemH2D(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int MemD2H(Task *task, BaseMem* mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, size_t size, void* host, const char *tag="");
  int KernelGet(Kernel *kernel, void** kernel_bin, const char* name, bool report_error=true);
  int KernelLaunchInit(Command *cmd, Kernel* kernel);
  int KernelSetArg(Kernel* kernel, int idx, int kindex, size_t size, void* value);
  int KernelSetMem(Kernel* kernel, int idx, int kindex, BaseMem* mem, size_t off);
  int KernelLaunch(Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws);
  int Synchronize();
  int AddCallback(Task* task);
  bool SupportJIT() { return false; }
  void CreateCQiree();

private:
  bool is_qiree_process_;
  bool is_xacc_;
  bool is_qsim_;
  CQiree *manager_;
  LoaderQIREE* ld_;
  unsigned int shared_mem_bytes_;
  unsigned int shared_mem_offs_[IRIS_MAX_KERNEL_NARGS];
  void* params_[IRIS_MAX_KERNEL_NARGS];
  int max_arg_idx_;
  int nquant_results_;
  int nclass_results_;
  uint32_t num_shots_;
  const char *backend_;
  const char *config_json_;
  size_t result_items_;
  CQireeResultRecord *encoded_;
  ProcessQiree *qiree_process_;

  //float* Z;
  //float* X;
  //float* Y;
  //float A;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEVICE_QIREE_H */


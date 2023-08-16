#ifndef IRIS_SRC_RT_DATAMEM_H
#define IRIS_SRC_RT_DATAMEM_H

#include "Config.h"
#include "Retainable.h"
#include "BaseMem.h"
#include <pthread.h>
#include <set>
#include <vector>
#include <assert.h>
#include "AsyncData.h"

namespace iris {
namespace rt {

class Platform;
class Command;
class Device;
class DataMemRegion;
class DataMem;

using DataMemDevice = AsyncData<DataMem>;

class DataMem: public BaseMem {
public:
  DataMem(Platform* platform, void *host, size_t size);
  DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
  static void CompleteCallback(void *stream, int status, DataMemDevice *data);
  void Init(Platform *platform, void *host_ptr, size_t size);
  virtual ~DataMem();
  void UpdateHost(void *host);
  void EnableOuterDimensionRegions();
  void AddWaitEvent(int devno, void *event) { device_map_[devno].AddWaitEvent(event); }
  void ClearWaitEvents(int devno) { device_map_[devno].ClearWaitEvents(); }
  vector<void *> & GetWaitEvents(int devno) { device_map_[devno].GetWaitEvents(); }
  int GetWriteStream(int devno) { return device_map_[devno].GetWriteStream(); }
  void SetWriteStream(int devno, int stream) { device_map_[devno].SetWriteStream(stream); }
  bool IsProactive(int devno) { return device_map_[devno].IsProactive(); }
  void EnableProactive(int devno) { device_map_[devno].EnableProactive(); }
  void DisableProactive(int devno) { device_map_[devno].DisableProactive(); }
  void *GetCompletionEvent(int devno) { return device_map_[devno].GetCompletionEvent(); }
  void **GetCompletionEventPtr(int devno) { return device_map_[devno].GetCompletionEventPtr(); }
  void RecordEvent(int devno, int stream);
  void WaitForEvent(int devno, int stream, int dep_devno);
  void init_reset(bool reset=true);
  bool is_host_dirty() { return host_dirty_flag_; }
  void clear_host_dirty() { host_dirty_flag_ = false; }
  void set_host_dirty(bool flag=true) { host_dirty_flag_ = flag; }
  bool is_dev_dirty(int devno) { return dirty_flag_[devno]; }
  void set_dev_dirty(int devno, bool flag=true) { dirty_flag_[devno] = flag; }
  void clear_dev_dirty(int devno) { dirty_flag_[devno] = false; }
  void set_dirty_except(int devno) {
    for(int i=0; i<ndevs_; i++) {
        if (i != devno) dirty_flag_[i] = true;
    }
    dirty_flag_[devno] = false;
  }
  void set_dirty_all(int devno) {
    for(int i=0; i<ndevs_; i++) {
        dirty_flag_[i] = true;
    }
  }
  int *get_non_dirty_devices(int *dev) {
    int i=0,j=0;
    for(i=0, j=0; i<ndevs_; i++) {
        if (!dirty_flag_[i]) dev[j++] = i;
    }
    dev[j++] = -1;
    return dev;
  }
  bool is_dirty_all() {
    bool all=true;
    for(int i=0; i<ndevs_; i++) {
        all = all & dirty_flag_[i];
    }
    return all;
  }
  void dev_unlock(int devno) {
    device_map_[devno].Unlock();
  }
  void dev_lock(int devno) {
    device_map_[devno].Lock();
  }
  void clear();
  size_t *off() { return off_; }
  virtual size_t *local_off() { return off_; }
  size_t *host_size() { return host_size_; }
  size_t *dev_size() { return dev_size_; }
  size_t elem_size() { return elem_size_; }
  int dim() { return dim_; }
  void *host_ptr() { return host_ptr_; }
  virtual void *host_root_memory() { return host_memory(); }
  virtual void *host_memory();
  void lock_host_region(int region);
  void unlock_host_region(int region);
  int get_n_regions() { return n_regions_; }
  bool is_regions_enabled() { return n_regions_ != -1; }
  DataMemRegion *get_region(int i) { return regions_[i]; }
  virtual void* arch(int devno, void *host=NULL);
  virtual void* arch(Device* dev, void *host=NULL);
  virtual void** arch_ptr(int devno, void *host=NULL);
  virtual void** arch_ptr(Device *dev, void *host=NULL);
  inline void create_dev_mem(Device *dev, int devno, void *host);
  Platform *platform() { return platform_; }
protected:
  bool host_dirty_flag_;
  bool  *dirty_flag_;
  pthread_mutex_t host_mutex_;
  int n_regions_;
  void *host_ptr_;
  size_t off_[3];
  size_t host_size_[3];
  size_t dev_size_[3];
  size_t elem_size_;
  int dim_;
  bool host_ptr_owner_;
  Platform *platform_;
  DataMemRegion **regions_;
  DataMemDevice *device_map_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEM_H */

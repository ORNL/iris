#ifndef IRIS_SRC_RT_DATAMEMREGION_H
#define IRIS_SRC_RT_DATAMEMREGION_H

#include "Config.h"
#include "Retainable.h"
#include "BaseMem.h"
#include <pthread.h>
#include <set>
#include <vector>
#include <assert.h>

namespace iris {
namespace rt {

class Platform;
class Command;
class Device;
class DataMem;

class DataMemRegion : public BaseMem
{
  public:
      DataMemRegion(DataMem *mem, int region, size_t *off, size_t *loff, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, size_t dev_offset_from_root);
      ~DataMemRegion() {
        for(int i=0; i<ndevs_; i++) {
          pthread_mutex_destroy(&dev_mutex_[i]);
        }
        pthread_mutex_destroy(&host_mutex_);
        delete [] dirty_flag_;
        delete [] dev_mutex_;
      }
      bool is_host_dirty() { return host_dirty_flag_; }
      inline DataMem *get_dmem() {
        return mem_;
      }
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
        pthread_mutex_unlock(&dev_mutex_[devno]);
      }
      void dev_lock(int devno) {
        pthread_mutex_lock(&dev_mutex_[devno]);
      }
      void init_reset(bool reset=true);
      void* arch(int devno, void *host=NULL);
      void* arch(Device* dev, void *host=NULL);
      void** arch_ptr(int devno, void *host=NULL);
      void** arch_ptr(Device *dev, void *host=NULL);
      size_t dev_offset_from_root() { return dev_offset_from_root_; }
      size_t *off() { return off_; }
      size_t *local_off() { return loff_; }
      size_t *host_size() { return host_size_; }
      size_t *dev_size() { return dev_size_; }
      size_t elem_size() { return elem_size_; }
      int dim() { return dim_; }
      //void *host_ptr() { return host_ptr_; }
      void *host_memory();
      void *host_root_memory();

  private:
      DataMem *mem_;
      //void *host_ptr_;
      size_t loff_[3];
      size_t off_[3];
      size_t host_size_[3];
      size_t dev_size_[3];
      size_t elem_size_;
      int dim_;
      int region_;
      size_t dev_offset_from_root_;
      bool host_dirty_flag_;
      bool *dirty_flag_;
      pthread_mutex_t *dev_mutex_;
      pthread_mutex_t host_mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEMREGION_H */

#ifndef IRIS_SRC_RT_DATAMEM_H
#define IRIS_SRC_RT_DATAMEM_H

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
class DataMemRegion;
class DataMem;

class DataMem: public BaseMem {
public:
  DataMem(Platform* platform, void *host, size_t size);
  DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
  void Init(Platform *platform, void *host_ptr, size_t size);
  virtual ~DataMem();
  void UpdateHost(void *host);
  void EnableOuterDimensionRegions();
  void init_reset(bool reset=true);
  bool is_host_dirty() { return host_dirty_flag_; }
  void clear_host_dirty() { host_dirty_flag_ = false; }
  void set_host_dirty(bool flag=true) { host_dirty_flag_ = flag; }
  bool is_dev_dirty(int devno) { return dirty_flag_[devno]; }
  void set_dev_dirty(int devno, bool flag=true) { dirty_flag_[devno] = flag; }
  int  get_dev_affinity() { 
      for(int i=0; i<ndevs_; i++) 
          if (!dirty_flag_[i]) 
              return i; 
      return -1;
  }
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
  void clear();
  size_t *off() { return off_; }
  virtual size_t *local_off() { return off_; }
  size_t *host_size() { return host_size_; }
  size_t *dev_size() { return dev_size_; }
  size_t elem_size() { return elem_size_; }
  int dim() { return dim_; }
  void *host_ptr() { return host_ptr_; }
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
  void* get_host_ptr_shadow(){return host_ptr_shadow_;}
  void set_host_ptr_shadow(void* host_ptr_shadow){
	  host_ptr_shadow_ = host_ptr_shadow;}
  bool is_host_shadow_dirty() { return host_dirty_flag_; }
  void clear_host_shadow_dirty() { host_shadow_dirty_flag_ = false; }
  void set_host_shadow_dirty(bool flag=true) { host_shadow_dirty_flag_ = flag; }
  DataMem* get_current_dmem_shadow(){return current_dmem_shadow_;}
  void set_current_dmem_shadow(DataMem* current_dmem_shadow){current_dmem_shadow_  = current_dmem_shadow; has_shadow_ = true;}
  DataMem* get_main_dmem(){return main_dmem_;}
  void set_main_dmem(DataMem* main_dmem){ main_dmem_ = main_dmem;}
  bool get_is_shadow(){ return is_shadow_;}
  void set_is_shadow(bool is_shadow){ is_shadow_ = is_shadow;}
  bool get_has_shadow(){ return has_shadow_;}
  void set_has_shadow(bool has_shadow){ has_shadow_ = has_shadow;}
#endif
#endif
  void update_bc_row_col(bool bc, int row, int col){  bc_ = bc; row_ = row; col_ = col;}
  bool get_bc(){ return bc_;}
  int get_row(){ return row_;}
  int get_col(){ return col_;}
  int get_rr_bc_dev(){ return rr_bc_dev_;}
  int set_rr_bc_dev(int rr_bc_dev){ rr_bc_dev_ = rr_bc_dev;}
  void *host_root_memory() { return host_memory(); }
  void *host_memory();
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
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
  void *host_ptr_shadow_;
  bool host_shadow_dirty_flag_;
  DataMem* current_dmem_shadow_;  // shadow object
  DataMem* main_dmem_; // shadow of this main dmem 
  bool is_shadow_;
  bool has_shadow_;
#endif
#endif
  int row_, col_, rr_bc_dev_; // index for BC distribution
  bool bc_; // for BC distribution
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEM_H */

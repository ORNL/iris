#ifndef IRIS_SRC_RT_DATAMEM_H
#define IRIS_SRC_RT_DATAMEM_H

#include "Config.h"
#include "Retainable.h"
#include "BaseMem.h"
#include <pthread.h>
#include <set>
#include <vector>
#include <utility>
#include <assert.h>

using namespace std;

namespace iris {
namespace rt {

class Platform;
class Command;
class Device;
class DataMemRegion;
class DataMem;

class DataMem: public BaseMem {
public:
  DataMem(Platform* platform, void *host, size_t size, int element_type=iris_unknown);
  DataMem(Platform* platform, void *host, size_t size, const char *symbol, int element_type=iris_unknown);
  DataMem(Platform* platform, void *host, size_t *host_size, int dim, size_t elem_size, int element_type=iris_unknown);
  DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type=iris_unknown);
  void Init(Platform *platform, void *host_ptr, size_t size);
  virtual ~DataMem();
  void AddChild(DataMem *child, size_t offset);
  void FetchDataFromDevice(void *dst_host_ptr);
  void FetchDataFromDevice(void *dst_host_ptr, size_t size);
  void UpdateHost(void *host);
  void RefreshHost();
  void EnableOuterDimensionRegions();
  vector<pair<DataMem *, size_t>> & child() { return child_; }
  void init_reset(bool reset=true);
  bool is_host_dirty() { 
      if (source_mem_ == NULL) return host_dirty_flag_; 
      else return source_mem_->host_dirty_flag_;
  }
  void clear_host_dirty() { 
      if (source_mem_ == NULL) host_dirty_flag_ = false; 
      else source_mem_->host_dirty_flag_ = false;
  }
  void set_host_dirty(bool flag=true) { 
      if (source_mem_ == NULL) host_dirty_flag_ = flag; 
      else source_mem_->host_reset_ = false; 
  }
  bool is_dev_dirty(int devno) { 
      if (source_mem_ == NULL) return dirty_flag_[devno]; 
      else return source_mem_->dirty_flag_[devno];
  }
  void set_dev_dirty(int devno, bool flag=true) { 
      if (source_mem_ == NULL) dirty_flag_[devno] = flag; 
      else source_mem_->dirty_flag_[devno] = flag;
  }
  int  get_dev_affinity() { 
      if (source_mem_ != NULL) return source_mem_->get_dev_affinity();
      for(int i=0; i<ndevs_; i++) 
          if (!dirty_flag_[i]) 
              return i; 
      return -1;
  }
  void clear_dev_dirty(int devno) { 
      if (source_mem_ == NULL) dirty_flag_[devno] = false; 
      else source_mem_->dirty_flag_[devno] = false; 
  }
  void set_dirty_except(int devno) {
    if (source_mem_ != NULL) {
        source_mem_->set_dirty_except(devno);
        return;
    }
    for(int i=0; i<ndevs_; i++) {
        if (i != devno) dirty_flag_[i] = true;
    }
    dirty_flag_[devno] = false;
  }
  void set_dirty_all(bool flag=true) {
    if (source_mem_ != NULL) {
        source_mem_->set_dirty_all(flag);
        return;
    }
    for(int i=0; i<ndevs_; i++) {
        dirty_flag_[i] = flag;
    }
  }
  int *get_non_dirty_devices(int *dev) {
    if (source_mem_ != NULL) 
        return source_mem_->get_non_dirty_devices(dev);
    int i=0,j=0;
    for(i=0, j=0; i<ndevs_; i++) {
        if (!dirty_flag_[i]) dev[j++] = i;
    }
    dev[j++] = -1;
    return dev;
  }
  bool is_dirty_all() {
    if (source_mem_ != NULL) return source_mem_->is_dirty_all();
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
  void *host_ptr() { 
      if (source_mem_ == NULL) return host_ptr_; 
      else return source_mem_->host_ptr_; 
  }
  void *tmp_host_ptr() { return tmp_host_ptr_; }
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
  void set_rr_bc_dev(int rr_bc_dev){ rr_bc_dev_ = rr_bc_dev;}
  void set_h2d_df_flag(int dev) { h2d_df_flag_[dev] = true;}
  void unset_h2d_df_flag(int dev) { h2d_df_flag_[dev] = false;}
  bool get_h2d_df_flag(int dev) { return h2d_df_flag_[dev];}
  bool is_symbol() { return is_symbol_; }

  void *host_root_memory() { return host_memory(); }
  void *host_memory();
  void *tmp_host_memory();
  void lock_host_region(int region);
  void unlock_host_region(int region);
  int get_n_regions() { return n_regions_; }
  bool is_regions_enabled() { return n_regions_ != -1; }
  DataMemRegion *get_region(int i) { return regions_[i]; }
  virtual void* arch(int devno, void *host=NULL);
  virtual void* arch(Device* dev, void *host=NULL);
  virtual void** arch_ptr(int devno, void *host=NULL);
  virtual void** arch_ptr(Device *dev, void *host=NULL);
  int update_host_size(size_t *host_size);
  inline void create_dev_mem(Device *dev, int devno, void *host);
  Platform *platform() { return platform_; }
  bool is_pin_memory() { return is_pin_memory_; }
  void set_pin_memory(bool flag=true) { is_pin_memory_ = flag; }
protected:
  bool host_dirty_flag_;
  bool  *dirty_flag_;
  pthread_mutex_t host_mutex_;
  int n_regions_;
  void *host_ptr_;
  void *tmp_host_ptr_;
  size_t off_[DMEM_MAX_DIM];
  size_t host_size_[DMEM_MAX_DIM];
  size_t dev_size_[DMEM_MAX_DIM];
  size_t elem_size_;
  int dim_;
  bool host_ptr_owner_;
  Platform *platform_;
  DataMemRegion **regions_;
  bool tile_enabled_;
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
  bool h2d_df_flag_[16];
  bool is_symbol_;
  bool is_pin_memory_;
  vector<pair<DataMem *, size_t> > child_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEM_H */

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

struct UncoveredRegion {
    size_t off_[DMEM_MAX_DIM];
    size_t dev_size_[DMEM_MAX_DIM];
    int dim_;

    UncoveredRegion(const size_t offset[], const size_t size[], int ndim) : dim_(ndim) {
        memcpy(off_, offset, ndim * sizeof(size_t));
        memcpy(dev_size_, size, ndim * sizeof(size_t));
    }

    void start(size_t out[DMEM_MAX_DIM]) const {
        memcpy(out, off_, dim_ * sizeof(size_t));
    }

    void end(size_t out[DMEM_MAX_DIM]) const {
        for (int i = 0; i < dim_; ++i)
            out[i] = off_[i] + dev_size_[i];
    }
};


class DataMem: public BaseMem {
public:
  DataMem(Platform* platform, void *host, size_t size, int element_type=iris_unknown);
  DataMem(Platform* platform, void *host, size_t size, const char *symbol, int element_type=iris_unknown);
  DataMem(Platform* platform, void *host, size_t *host_size, int dim, size_t elem_size, int element_type=iris_unknown);
  DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type=iris_unknown);
  DataMem *Copy();
  void Init(Platform *platform, void *host_ptr, size_t size, int dim=1);
  virtual ~DataMem();
  void AddChild(DataMem *child, size_t offset);
  void FetchDataFromDevice(void *dst_host_ptr);
  void FetchDataFromDevice(void *dst_host_ptr, size_t size);
  void UpdateHost(void *host);
  void CopyToHost(void *host);
  void RefreshHost();
  void EnableOuterDimensionRegions();
  iris_mem CreateRegion(size_t *off, size_t *size);
  vector<pair<DataMem *, size_t>> & child() { return child_; }
  void init_reset(bool reset=true);
  vector<UncoveredRegion> find_uncovered_tiles();
  bool is_within_bounds(size_t *off, size_t *size);
  int check_overlap(size_t *off, size_t *size);
  virtual bool is_host_dirty(bool check_regions=true);
  virtual void clear_host_dirty();
  virtual void set_host_dirty(bool flag=true);
  bool is_dev_dirty(int devno, bool check_regions=true);
  void set_dev_dirty(int devno, bool flag=true, bool check_regions=true);
  int  get_dev_affinity();
  void clear_dev_dirty(int devno);
  void set_dirty_except(int devno);
  void set_dirty_all(bool flag=true);
  int *get_non_dirty_devices(int *dev);
  bool is_dirty_all();
  void clear();
  size_t *off() { 
    return off_;
  }
  virtual size_t *local_off() { return off(); }
  size_t *host_size() { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem == NULL) return host_size_;
    return source_mem->host_size(); 
  }
  size_t *dev_strides() { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem == NULL) return dev_strides_;
    return source_mem->dev_strides(); 
  }
  size_t *host_strides() { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem == NULL) return host_strides_;
    return source_mem->host_strides(); 
  }
  virtual void** archs() { 
      DataMem *source_mem = (DataMem*)get_source_mem();
      if (source_mem != NULL) return source_mem->archs();
      return archs_; 
  }
  virtual void **get_arch_ptr(int devno) { 
      DataMem *source_mem = (DataMem*)get_source_mem();
      if (source_mem != NULL) return source_mem->get_arch_ptr(devno);
      return &archs_[devno]; 
  }
  virtual void *get_arch(int devno) { 
      DataMem *source_mem = (DataMem*)get_source_mem();
      if (source_mem != NULL) return source_mem->get_arch(devno);
      return archs_[devno]; 
  }
  virtual void set_arch(int devno, void *ptr) { 
      DataMem *source_mem = (DataMem*)get_source_mem();
      if (source_mem != NULL) source_mem->set_arch(devno, ptr);
      else archs_[devno] = ptr; 
  }
  size_t *dev_off() { return dev_off_; }
  size_t *dev_size() { return dev_size_; }
  size_t elem_size() { return elem_size_; }
  void enable_multi_dim_dev_offset() { multi_dim_dev_offset_ = true; }
  bool is_multi_dim_dev_offset() { return multi_dim_dev_offset_; }
  int update_dim(int dim) { dim_ = dim; return IRIS_SUCCESS; }
  int dim() { return dim_; }
  void *host_ptr() { 
      DataMem *source_mem = (DataMem*)get_source_mem();
      if (source_mem == NULL) return host_ptr_; 
      else return source_mem->host_ptr_; 
  }
  void *tmp_host_ptr() { return tmp_host_ptr_; }
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
  void* get_host_ptr_shadow(){return host_ptr_shadow_;}
  void set_host_ptr_shadow(void* host_ptr_shadow){
	  host_ptr_shadow_ = host_ptr_shadow;}
  bool is_host_shadow_dirty() { 
      DataMem *source_mem = (DataMem*)get_source_mem();
      if (source_mem == NULL) return host_dirty_flag_;
      return source_mem->host_dirty_flag_; 
  }
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
  void compute_strides(size_t *strides, size_t *size);
  void compute_dev_strides(size_t *size);
  void compute_host_strides(size_t *size);
  void *tmp_host_memory();
  void lock_host_region(int region);
  void unlock_host_region(int region);
  int get_n_regions() { return n_regions_; }
  bool is_regions_enabled() { return n_regions_ != 0; }
  bool is_dynamic_regions() { return dynamic_regions_; }
  DataMemRegion *get_region(int i, bool dev_region=true) {
    ASSERT((size_t)i < regions_.size()); 
    if (dev_region)
        return regions_[i].second;
    else
        return regions_[i].first;
  }
  virtual void* arch(int devno, void *host=NULL);
  virtual void* arch(Device* dev, void *host=NULL);
  virtual void** arch_ptr(int devno, void *host=NULL);
  virtual void** arch_ptr(Device *dev, void *host=NULL);
  int update_host_size(size_t *host_size);
  int update_dev_size(size_t *dev_size) { 
      memcpy(dev_size_, dev_size, sizeof(size_t)*dim_); 
      return IRIS_SUCCESS;
  }
  void realllocate_space(size_t size);
  int update_off(size_t *off) { 
      memcpy(off_, off, sizeof(size_t)*dim_); 
      return IRIS_SUCCESS;
  }
  inline void create_dev_mem(Device *dev, int devno, void *host);
  Platform *platform() { return platform_; }
  bool is_pin_memory() { return is_pin_memory_; }
  void set_pin_memory(bool flag=true) { is_pin_memory_ = flag; }
  bool tile_enabled() { return tile_enabled_; }
  void Print(bool check_regions=true);
  string ToString(bool check_regions=true);
  void EnableRegions(int axis);
  string GetSizeString(size_t *data, int dim);
  string GetPtrString(void **data, int dim);
protected:
  void start(size_t out[DMEM_MAX_DIM]) const {
      memcpy(out, off_, dim_ * sizeof(size_t));
  }

  void end(size_t out[DMEM_MAX_DIM]) const {
      for (int i = 0; i < dim_; ++i)
          out[i] = off_[i] + dev_size_[i];
  }
  void set_dev_off(size_t *dev_off) {
      memcpy(dev_off_, dev_off, dim_ * sizeof(size_t));
  }
  bool host_dirty_flag_;
  bool  *dirty_flag_;
  bool dynamic_regions_;
  bool host_ptr_owner_;
  bool tile_enabled_;
  bool bc_; // for BC distribution
  bool h2d_df_flag_[16];
  bool is_symbol_;
  bool is_pin_memory_;
  bool multi_dim_dev_offset_;
  pthread_mutex_t host_mutex_;
  int n_regions_;
  void *host_ptr_;
  void *tmp_host_ptr_;
  size_t dev_off_[DMEM_MAX_DIM];
  size_t off_[DMEM_MAX_DIM];
  size_t dev_strides_[DMEM_MAX_DIM];
  size_t host_strides_[DMEM_MAX_DIM];
  size_t host_size_[DMEM_MAX_DIM];
  size_t dev_size_[DMEM_MAX_DIM];
  size_t elem_size_;
  int dim_;
  Platform *platform_;
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
  vector<pair<DataMemRegion*, DataMemRegion*> > regions_;
  vector<pair<DataMem *, size_t> > child_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEM_H */

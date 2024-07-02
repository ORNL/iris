#include "DataMem.h"
#include "DataMemRegion.h"
#include "Debug.h"
#include "Platform.h"
#include "Device.h"
#include <stdlib.h>

#define USE_MEMRANGE

namespace iris {
namespace rt {

DataMem::DataMem(Platform* platform, void *host_ptr, size_t size, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) {
    Init(platform, host_ptr, size);
    set_element_type(element_type);
}
DataMem::DataMem(Platform* platform, void *host_ptr, size_t *host_size, int dim, size_t elem_size, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) {
    size_t size = elem_size;
    set_element_type(element_type);
    ASSERT(dim < DMEM_MAX_DIM);
    for(int i=0; i<dim; i++) {
        size = size * host_size[i];
    }
    Init(platform, host_ptr, size);
    dim_ = dim;
    memcpy(dev_size_, host_size, sizeof(size_t)*dim_);
    memcpy(host_size_, host_size, sizeof(size_t)*dim_);
    elem_size_ = elem_size;
    set_element_type(element_type);
}
DataMem::DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) 
{
    size_t size = elem_size;
    set_element_type(element_type);
    for(int i=0; i<dim; i++) {
        size = size * dev_size[i];
    }
    ASSERT(dim < DMEM_MAX_DIM);
    if (dim == 1) {
        _trace("DataMem host_ptr:%p off:%ld host_size:%ld dev_size:%ld elem_size:%ld dim:%d size:%ld", host_ptr, off[0], host_size[0], dev_size[0], elem_size, dim, size);
    }
    else if (dim == 2) {
        _trace("DataMem host_ptr:%p off:[%ld,%ld] host_size:[%ld,%ld] dev_size:[%ld,%ld] elem_size:%ld dim:%d size:%ld", host_ptr, off[0], off[1], host_size[0], host_size[1], dev_size[0], dev_size[1], elem_size, dim, size);
    }
    else if (dim == 3) {
        _trace("DataMem host_ptr:%p off:[%ld,%ld,%ld] host_size:[%ld,%ld,%ld] dev_size:[%ld,%ld,%ld] elem_size:%ld dim:%d size:%ld", host_ptr, off[0], off[1], off[2], host_size[0], host_size[1], host_size[2], dev_size[0], dev_size[1], dev_size[2], elem_size, dim, size);
    }
    Init(platform, host_ptr, size);
    dim_ = dim;
    memcpy(off_, off, sizeof(size_t)*dim_);
    memcpy(dev_size_, dev_size, sizeof(size_t)*dim_);
    memcpy(host_size_, host_size, sizeof(size_t)*dim_);
    elem_size_ = elem_size;
}
void DataMem::Init(Platform *platform, void *host_ptr, size_t size)
{
    platform_ = platform;
    host_ptr_owner_ = false;
    size_ = size;
    n_regions_ = -1;
    regions_ = NULL;
    host_dirty_flag_ = false;
    dirty_flag_ = new bool[ndevs_];
    for (int i = 0; i < ndevs_; i++) {
        archs_[i] = NULL;
        archs_dev_[i] = platform->device(i);
        dirty_flag_[i] = true;
        //dev_ranges_[i] = NULL;
    }
    elem_size_ = size_;
    for(int i=0; i<DMEM_MAX_DIM; i++) {
        host_size_[i] = 1;
        dev_size_[i] = 1;
        off_[i] = 0;
    }
    dim_ = 1;
    host_ptr_ = host_ptr;
    //printf("host pointer %p\n", host_ptr_);
#ifdef AUTO_PAR
  current_writing_task_ = NULL;
#ifdef AUTO_FLUSH
  flush_task_ = NULL;
#endif
#ifdef AUTO_SHADOW
  host_shadow_dirty_flag_ = true; //initially the shadow is dirty
  current_dmem_shadow_ = NULL; //initially the shadow is dirty
  is_shadow_ = false; // 0: not a shadow, 1: a shadow
  main_dmem_ = NULL; // 0: not a shadow, 1: a shadow
  has_shadow_ = false; // 0: does not have a shadow, 1: has a shadow
#endif
#endif
  bc_ = false;
  row_ = -1;
  col_ = -1;
  rr_bc_dev_ = -1;
  // for keeping track whether any device initiated h2d at the beginning
  for (int i = 0; i < platform_->ndevs(); i++){
    h2d_df_flag_[i] = false;
  }
}
DataMem::~DataMem() {
    if (host_ptr_owner_) free(host_ptr_);
    for (int i = 0; i < ndevs_; i++) {
        if (archs_[i] && !is_usm(i)) archs_dev_[i]->MemFree(this, archs_[i]);
    }
    delete [] dirty_flag_;
    for(int i=0; i<n_regions_; i++) {
        delete regions_[i];
    }
    if (regions_) delete [] regions_;
}
void DataMem::UpdateHost(void *host_ptr)
{
    if (host_ptr_owner_ && host_ptr_ != NULL) free(host_ptr_);
    host_ptr_ = host_ptr;
    host_ptr_owner_ = false;
    host_dirty_flag_ = false;
    for(int i=0; i<ndevs_; i++) {
        dirty_flag_[i] = true;
    }
}
void DataMem::init_reset(bool reset)
{
    reset_ = reset;
    host_dirty_flag_ = reset;
    for(int i=0;  i<ndevs_; i++) {
        dirty_flag_[i] = !reset;
    }
}
void DataMem::clear() {
  host_dirty_flag_ = false;
  for(int i=0;  i<ndevs_; i++) {
      dirty_flag_[i] = true;
  }
  for (int i = 0; i < ndevs_; i++) {
      if (archs_[i]) {
          if (! is_usm(i)) archs_dev_[i]->MemFree(this, archs_[i]);
          archs_[i] = NULL;
      }
  }
}
void *DataMem::host_memory() {
    if (!host_ptr_)  {
        host_ptr_ = malloc(size_);
        host_ptr_owner_ = true;
    }
    return host_ptr_;
}
void DataMem::EnableOuterDimensionRegions()
{
    if (regions_ != NULL) return;
    int outer_dim = dev_size_[dim_-1];
    n_regions_ = outer_dim;
    regions_ = new DataMemRegion*[n_regions_];
    size_t dev_size[DMEM_MAX_DIM], off[DMEM_MAX_DIM], loff[DMEM_MAX_DIM];
    memcpy(dev_size, dev_size_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(off, off_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(loff, off_, sizeof(size_t)*DMEM_MAX_DIM);
    dev_size[dim_-1] = 1;
    size_t dev_offset = 1;
    for(int i=0; i<dim_-1; i++) 
        dev_offset *= dev_size_[i];
    for(int i=0; i<n_regions_; i++) {
        off[dim_-1] = off_[dim_-1]+i; 
        loff[dim_-1] = 0;
        size_t dev_offset_from_root = i * dev_offset * elem_size_;
        regions_[i] = new DataMemRegion(this, i, off, loff, host_size_, dev_size, elem_size_, dim_, dev_offset_from_root);
    }
}

void DataMem::create_dev_mem(Device *dev, int devno, void *host)
{
    //printf(" Dev: %d is shared:%d host:%p host_ptr_:%p\n", devno, dev->is_shared_memory_buffers(), host, host_ptr_);
    if (is_usm(devno) && dev->is_shared_memory_buffers() && 
            (host != NULL || host_ptr_ != NULL)) {
        archs_[devno] = dev->GetSharedMemPtr((host == NULL) ? host_ptr_ : host, size());
    }
    else {
        set_usm_flag(devno, false);
        dev->MemAlloc(this, archs_ + devno, size_, is_reset());
        if (is_reset()) {
            dirty_flag_[devno] = false;
            host_dirty_flag_ = true;
        }
    }
}

void** DataMem::arch_ptr(Device *dev, void *host) {
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        create_dev_mem(dev, devno, host);
    }
    return &archs_[dev->devno()];
}

void** DataMem::arch_ptr(int devno, void *host) {
    if (archs_[devno] == NULL) {
        Device *dev = archs_dev_[devno];
        create_dev_mem(dev, devno, host);
    }
    return &archs_[devno];
}

void* DataMem::arch(int devno, void *host) {
    if (archs_[devno] == NULL) {
        Device *dev = archs_dev_[devno];
        create_dev_mem(dev, devno, host);
    }
    return archs_[devno];
}

void* DataMem::arch(Device* dev, void *host) {
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        create_dev_mem(dev, devno, host);
    }
    return archs_[devno];
}

} /* namespace rt */
} /* namespace iris */


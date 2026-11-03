#include "DataMemRegion.h"
#include "DataMem.h"
#include "Debug.h"
#include "Platform.h"
#include "Device.h"
#include <stdlib.h>

#define USE_MEMRANGE

namespace iris {
namespace rt {

DataMemRegion::DataMemRegion(DataMem *mem, int region, size_t *off, size_t *loff, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, size_t dev_offset_from_root) : DataMem(mem->platform(), NULL, off, host_size, dev_size, elem_size, dim) 
{
    set_element_type(mem->element_type());
    SetMemHandlerType(IRIS_DMEM_REGION);
    mem_ = mem;
    region_ = region;
    memcpy(loff_, loff, sizeof(size_t)*DMEM_MAX_DIM);
    own_dev_memory_ = false;
    dev_offset_from_root_ = dev_offset_from_root;
    if (mem_->is_reset()) init_reset(true);
    set_host_dirty(mem->is_host_dirty(false));
    for(int i=0; i<ndevs_; i++) {
        set_dev_dirty(i, mem->is_dev_dirty(i, false), false);
    }
    host_ptr_ = mem->host_ptr();
	host_ptr_owner_ = false;
}
void *DataMemRegion::host_root_memory() { return mem_->host_memory(); }

bool DataMemRegion::is_host_dirty(bool check_regions) { 
  if (!own_dev_memory_) return DataMem::is_host_dirty(check_regions);
  return get_clone_region()->is_host_dirty(check_regions);
}
void DataMemRegion::clear_host_dirty() { 
  if (!own_dev_memory_) return DataMem::clear_host_dirty();
  return get_clone_region()->clear_host_dirty();
}
void DataMemRegion::set_host_dirty(bool flag) { 
  if (!own_dev_memory_) return DataMem::set_host_dirty(flag);
  return get_clone_region()->set_host_dirty(flag);
}
void** DataMemRegion::arch_ptr(int devno, void *host) {
    if (archs_[devno] == NULL) {
        if (!own_dev_memory_) {
            uint8_t *dev_mem = (uint8_t*)mem_->arch(devno, host);
            dev_mem = dev_mem + dev_offset_from_root_;
            archs_[devno] = dev_mem;
        }
        else {
            return DataMem::arch_ptr(devno, host);
        }
    }
    return &archs_[devno];
}
void** DataMemRegion::arch_ptr(Device* dev, void *host) {
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        if (!own_dev_memory_) {
            uint8_t *dev_mem = (uint8_t*)mem_->arch(dev, host);
            dev_mem = dev_mem + dev_offset_from_root_;
            archs_[devno] = dev_mem;
        }
        else {
            return DataMem::arch_ptr(dev, host);
        }
    }
    return &archs_[devno];
}
void* DataMemRegion::arch(Device* dev, void *host) {
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        if (!own_dev_memory_) {
            uint8_t *dev_mem = (uint8_t*)mem_->arch(dev, host);
            dev_mem = dev_mem + dev_offset_from_root_;
            archs_[devno] = dev_mem;
        }
        else {
            return DataMem::arch(dev, host);
        }
    }
    return archs_[devno];
}
void* DataMemRegion::arch(int devno, void *host) {
    if (archs_[devno] == NULL) {
        if (!own_dev_memory_) {
            uint8_t *dev_mem = (uint8_t*)mem_->arch(devno, host);
            dev_mem = dev_mem + dev_offset_from_root_;
            archs_[devno] = dev_mem;
        }
        else {
            return DataMem::arch(devno, host);
        }
    }
    return archs_[devno];
}
bool DataMemRegion::overlaps_with(size_t *b_off, size_t *b_size) const {
    for (int i = 0; i < dim_; ++i) {
        size_t b_end = b_off[i] + b_size[i];
        size_t a_end = off_[i] + dev_size_[i];
        if (a_end <= b_off[i] || b_end <= off_[i]) {
            return false;
        }
    }
    return true;
}

void *DataMemRegion::host_memory() { 
    if (!own_dev_memory_ && !is_multi_dim_dev_offset()) {
        size_t linear_offset = 0;
        for (int i = 0; i < dim_; ++i) {
            linear_offset += off_[i] * host_strides_[i];
        }
        uint8_t *host_start = (uint8_t *)mem_->host_memory() + 
            linear_offset * elem_size_; 
        return host_start;
    }
    else {
        return mem_->host_memory();
    }
}
}
}

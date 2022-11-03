#include "DataMemRegion.h"
#include "DataMem.h"
#include "Debug.h"
#include "Platform.h"
#include "Device.h"
#include <stdlib.h>

#define USE_MEMRANGE

namespace iris {
namespace rt {

DataMemRegion::DataMemRegion(DataMem *mem, int region, size_t *off, size_t *loff, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, size_t dev_offset_from_root) : BaseMem(IRIS_DMEM_REGION, mem->platform()->ndevs()) 
{
    region_ = region;
    memcpy(off_, off, sizeof(size_t)*3);
    memcpy(loff_, loff, sizeof(size_t)*3);
    memcpy(dev_size_, dev_size, sizeof(size_t)*3);
    memcpy(host_size_, host_size, sizeof(size_t)*3);
    mem_ = mem;
    elem_size_ = elem_size;
    dim_ = dim;
    dev_offset_from_root_ = dev_offset_from_root;
    //host_dirty_flag_ = mem_->is_host_dirty();
    host_dirty_flag_ = false;
    dirty_flag_ = new bool[ndevs_];
    pthread_mutex_init(&host_mutex_, NULL);
    dev_mutex_ = new pthread_mutex_t[ndevs_];
    for(int i=0;  i<ndevs_; i++) {
        dirty_flag_[i] = true;
        pthread_mutex_init(&dev_mutex_[i], NULL);
    }
    size_t size = elem_size;
    for(int i=0; i<dim; i++) {
        size = size * dev_size[i];
    }
    size_ = size;
    if (mem_->is_reset()) init_reset(true);
}
void DataMemRegion::init_reset(bool reset)
{
    reset_ = reset;
    if (reset) {
        host_dirty_flag_ = true;
        for(int i=0;  i<ndevs_; i++) {
            dirty_flag_[i] = false;
        }
    }
    else {
        host_dirty_flag_ = false;
        for(int i=0;  i<ndevs_; i++) {
            dirty_flag_[i] = true;
        }
    }
}
void *DataMemRegion::host_root_memory() { return mem_->host_memory(); }

void** DataMemRegion::arch_ptr(int devno, void *host) {
    if (archs_[devno] == NULL) {
        uint8_t *dev_mem = (uint8_t*)mem_->arch(devno, host);
        dev_mem = dev_mem + dev_offset_from_root_;
        archs_[devno] = dev_mem;
    }
    return &archs_[devno];
}
void** DataMemRegion::arch_ptr(Device* dev, void *host) {
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        uint8_t *dev_mem = (uint8_t*)mem_->arch(dev, host);
        dev_mem = dev_mem + dev_offset_from_root_;
        archs_[devno] = dev_mem;
    }
    return &archs_[devno];
}
void* DataMemRegion::arch(Device* dev, void *host) {
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        uint8_t *dev_mem = (uint8_t*)mem_->arch(dev, host);
        dev_mem = dev_mem + dev_offset_from_root_;
        archs_[devno] = dev_mem;
    }
    return archs_[devno];
}
void* DataMemRegion::arch(int devno, void *host) {
    if (archs_[devno] == NULL) {
        uint8_t *dev_mem = (uint8_t*)mem_->arch(devno, host);
        dev_mem = dev_mem + dev_offset_from_root_;
        archs_[devno] = dev_mem;
    }
    return archs_[devno];
}
void *DataMemRegion::host_memory() { 
    size_t host_row_pitch = elem_size_ * host_size_[0];
    size_t host_slice_pitch   = host_size_[1] * host_row_pitch;
    uint8_t *host_start = (uint8_t *)mem_->host_memory() + off_[0]*elem_size_ + off_[1] * host_row_pitch + off_[2] * host_slice_pitch;
    return host_start;
}
}
}

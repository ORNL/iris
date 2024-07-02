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
    dev_offset_from_root_ = dev_offset_from_root;
    if (mem_->is_reset()) init_reset(true);
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

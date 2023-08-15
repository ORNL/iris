#include "DataMem.h"
#include "DataMemRegion.h"
#include "Debug.h"
#include "Platform.h"
#include "Device.h"
#include <stdlib.h>

#define USE_MEMRANGE

namespace iris {
namespace rt {

DataMem::DataMem(Platform* platform, void *host_ptr, size_t size) : BaseMem(IRIS_DMEM, platform->ndevs()) {
    Init(platform, host_ptr, size);
}
DataMem::DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) : BaseMem(IRIS_DMEM, platform->ndevs()) 
{
    size_t size = elem_size;
    for(int i=0; i<dim; i++) {
        size = size * dev_size[i];
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
    device_map_ = new DataMemDevice[ndevs_+1];
    for (int i = 0; i < ndevs_; i++) {
        archs_[i] = NULL;
        archs_dev_[i] = platform->device(i);
        dirty_flag_[i] = true;
        device_map_[i].Init(this, i);
        //dev_ranges_[i] = NULL;
    }
    device_map_[ndevs_].Init(this, ndevs_);
    elem_size_ = size_;
    for(int i=0; i<3; i++) {
        host_size_[i] = 1;
        dev_size_[i] = 1;
        off_[i] = 0;
    }
    dim_ = 1;
    host_ptr_ = host_ptr;
}
DataMem::~DataMem() {
    if (host_ptr_owner_) free(host_ptr_);
    for (int i = 0; i < ndevs_; i++) {
        if (archs_[i]) archs_dev_[i]->MemFree(archs_[i]);
    }
    delete [] dirty_flag_;
    for(int i=0; i<n_regions_; i++) {
        delete regions_[i];
    }
    if (regions_) delete [] regions_;
    for(int i=0; i<ndevs_; i++) {
        if (GetCompletionEvent(i) != NULL) 
            archs_dev_[i]->DestroyEvent(GetCompletionEvent(i));
    }
    delete [] device_map_;
}
void DataMem::CompleteCallback(void *stream, int status, DataMemDevice *data)
{
    data->EnableCompleted();
}
void DataMem::RecordEvent(int devno, int stream) {
    if (GetCompletionEvent(devno) != NULL)
        archs_dev_[devno]->CreateEvent(GetCompletionEventPtr(devno), iris_event_disable_timing);
    archs_dev_[devno]->RecordEvent(GetCompletionEvent(devno), stream);
}
void DataMem::WaitForEvent(int devno, int stream, int dep_devno) {
    assert(GetCompletionEvent(devno) != NULL);
    archs_dev_[devno]->WaitForEvent(GetCompletionEvent(devno), stream, iris_event_disable_timing);
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
          archs_dev_[i]->MemFree(archs_[i]);
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
    size_t dev_size[3], off[3], loff[3];
    memcpy(dev_size, dev_size_, sizeof(size_t)*3);
    memcpy(off, off_, sizeof(size_t)*3);
    memcpy(loff, off_, sizeof(size_t)*3);
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
    if (dev->is_shared_memory_buffers() && (host != NULL || host_ptr_ != NULL)) 
        archs_[devno] = (host == NULL) ? host_ptr_ : host;
    else {
        dev->MemAlloc(archs_ + devno, size_, is_reset());
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


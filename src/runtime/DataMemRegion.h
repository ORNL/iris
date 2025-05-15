#ifndef IRIS_SRC_RT_DATAMEMREGION_H
#define IRIS_SRC_RT_DATAMEMREGION_H

#include "Config.h"
#include "Retainable.h"
#include "BaseMem.h"
#include "DataMem.h"
#include <pthread.h>
#include <set>
#include <vector>
#include <assert.h>

namespace iris {
namespace rt {

class Device;

class DataMemRegion : public DataMem
{
  public:
      DataMemRegion(DataMem *mem, int region, size_t *off, size_t *loff, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, size_t dev_offset_from_root);
      virtual ~DataMemRegion() { }
      inline DataMem *get_dmem() {
        return mem_;
      }
      void* arch(int devno, void *host=NULL);
      int region_index() { return region_; }
      void* arch(Device* dev, void *host=NULL);
      void** arch_ptr(int devno, void *host=NULL);
      void** arch_ptr(Device *dev, void *host=NULL);
      size_t dev_offset_from_root() { return dev_offset_from_root_; }
      void set_own_dev_memory(bool flag=true) { own_dev_memory_ = flag; }
      bool own_dev_memory() { return own_dev_memory_; }
      size_t *local_off() { 
          if (own_dev_memory_) return off_; 
          else return loff_; 
      }
      bool is_host_dirty(bool check_regions=true);
      void clear_host_dirty();
      void set_host_dirty(bool flag);
      //void *host_ptr() { return host_ptr_; }
      void *host_memory();
      void *host_root_memory();
      bool overlaps_with(size_t *b_off, size_t *b_size) const;
      DataMemRegion *get_clone_region() {
        DataMemRegion *adj = mem_->get_region(region_, false);
        return adj;
      }
      virtual void HostUnlock() {
          DataMemRegion *region = get_clone_region();
          region->DataMem::HostUnlock(); 
      }
      virtual void HostLock() {
          DataMemRegion *region = get_clone_region();
          region->DataMem::HostLock(); 
      }
      virtual void HostWriteLock(int devno) {
          DataMemRegion *region = get_clone_region();
          region->DataMem::HostWriteLock(devno); 
      }
      virtual void HostWriteUnLock(int devno) {
          DataMemRegion *region = get_clone_region();
          region->DataMem::HostWriteUnLock(devno); 
      }
      virtual void HardHostWriteEventSynchronize(Device *dev, void *event) 
      {
          DataMemRegion *region = get_clone_region();
          region->DataMem::HardHostWriteEventSynchronize(dev, event); 
      }
      virtual int GetHostWriteStream() { 
          DataMemRegion *region = get_clone_region();
          return region->DataMem::GetHostWriteStream(); 
      }
      virtual void SetHostWriteStream(int stream) { 
          DataMemRegion *region = get_clone_region();
          region->DataMem::SetHostWriteStream(stream); 
      }
      virtual int GetHostWriteDevice() { 
          DataMemRegion *region = get_clone_region();
          return region->DataMem::GetHostWriteDevice(); 
      }
      virtual void SetHostWriteDevice(int dev) { 
          DataMemRegion *region = get_clone_region();
          region->DataMem::SetHostWriteDevice(dev); 
      }
      virtual void HostRecordEvent(int devno, int stream)
      {
          DataMemRegion *region = get_clone_region();
          region->DataMem::HostRecordEvent(devno, stream); 
      }
      virtual void *GetHostCompletionEvent() { 
          DataMemRegion *region = get_clone_region();
          return region->DataMem::GetHostCompletionEvent(); 
      }
      virtual void **GetHostCompletionEventPtr(bool new_entry=false) { 
          DataMemRegion *region = get_clone_region();
          return region->DataMem::GetHostCompletionEventPtr(new_entry); 
      }
      virtual void *GetDeviceSpecificHostCompletionEvent(int devno) {
          DataMemRegion *region = get_clone_region();
          return region->DataMem::GetDeviceSpecificHostCompletionEvent(devno); 
      }

  private:
      DataMem *mem_;
      int region_;
      bool own_dev_memory_;
      size_t dev_offset_from_root_;
      size_t loff_[DMEM_MAX_DIM];
      //void *host_ptr_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEMREGION_H */

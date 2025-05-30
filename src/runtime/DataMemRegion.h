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
      void* arch(Device* dev, void *host=NULL);
      void** arch_ptr(int devno, void *host=NULL);
      void** arch_ptr(Device *dev, void *host=NULL);
      size_t dev_offset_from_root() { return dev_offset_from_root_; }
      size_t *local_off() { return loff_; }
      //void *host_ptr() { return host_ptr_; }
      void *host_memory();
      void *host_root_memory();

  private:
      DataMem *mem_;
      int region_;
      size_t dev_offset_from_root_;
      size_t loff_[DMEM_MAX_DIM];
      //void *host_ptr_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DATAMEMREGION_H */

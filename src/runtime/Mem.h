#ifndef IRIS_SRC_RT_MEM_H
#define IRIS_SRC_RT_MEM_H

#include "Config.h"
#include "Retainable.h"
#include "MemRange.h"
#include <pthread.h>
#include <set>

namespace iris {
namespace rt {

class Platform;

class Mem: public Retainable<struct _iris_mem, Mem> {
public:
  Mem(size_t size, Platform* platform);
  virtual ~Mem();

  void AddOwner(size_t off, size_t size, Device* dev);
  void SetOwner(size_t off, size_t size, Device* dev);
  void SetOwner(Device* dev);
  bool IsOwner(size_t off, size_t size, Device* dev);
  bool IsOwner(Device* dev);
  Device* Owner(size_t off, size_t size);
  Device* Owner();
  void Reduce(int mode, int type);
  void Expand(int expansion);

  void SetMap(void* host, size_t size);

  size_t size() { return size_; }
  int mode() { return mode_; }
  int type() { return type_; }
  int type_size() { return type_size_; }
  int expansion() { return expansion_; }
  void* host_inter();
  void* mapped_host() { return mapped_host_; }
  size_t mapped_size() { return mapped_size_; }
  bool mapped() { return mapped_host_ != NULL; }

  void** archs() { return archs_; }
  void* arch(Device* dev, void *host=NULL);
  void** archs_off() { return archs_off_; }

private:
  size_t size_;
  int mode_;
  Platform* platform_;
  std::set<MemRange*> ranges_;
  std::set<Device*> owners_;
  void* host_inter_;
  int ndevs_;
  int type_;
  int type_size_;
  int expansion_;
  void* mapped_host_;
  size_t mapped_size_;
  void* archs_[IRIS_MAX_NDEVS];
  void* archs_off_[IRIS_MAX_NDEVS];
  Device* archs_dev_[IRIS_MAX_NDEVS];

  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_MEM_H */

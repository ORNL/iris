#ifndef IRIS_SRC_RT_MEM_H
#define IRIS_SRC_RT_MEM_H

#include "Config.h"
#include "Retainable.h"
#include "MemRange.h"
#include <pthread.h>
#include <set>
#include <vector>

namespace iris {
namespace rt {

class Platform;
class Command;

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

  void set_intermediate(bool flag=true) { intermediate_=true; }
  bool is_intermediate() { return intermediate_; }
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
  std::vector<Command *> & get_h2d_cmds() { return h2d_cmds_; }
  std::vector<Command *> & get_h2dnp_cmds() { return h2dnp_cmds_; }
  std::vector<Command *> & get_d2h_cmds() { return d2h_cmds_; }

private:
  bool intermediate_;
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
  std::vector<Command *> h2d_cmds_;
  std::vector<Command *> h2dnp_cmds_;
  std::vector<Command *> d2h_cmds_;
  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_MEM_H */

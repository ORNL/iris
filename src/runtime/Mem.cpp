#include "Mem.h"
#include "Debug.h"
#include "Platform.h"
#include "Device.h"
#include <stdlib.h>

#define USE_MEMRANGE

namespace iris {
namespace rt {

Mem::Mem(size_t size, Platform* platform) : BaseMem(IRIS_MEM, platform->ndevs()) {
  size_ = size;
  mode_ = iris_normal;
  expansion_ = 1;
  platform_ = platform;
  host_inter_ = NULL;
  mapped_host_ = NULL;
  for (int i = 0; i < ndevs_; i++) {
    archs_dev_[i] = platform->device(i);
    archs_[i] = NULL;
  }
  pthread_mutex_init(&mutex_, NULL);
}

Mem::~Mem() {
  for (int i = 0; i < ndevs_; i++) {
    if (archs_[i]) archs_dev_[i]->MemFree(archs_[i]);
  }
  if (!host_inter_) free(host_inter_);
  pthread_mutex_destroy(&mutex_);
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
      MemRange *mr = *I;
      delete mr;
  }
}

void** Mem::arch_ptr(int devno, void *host) {
  arch(devno, host);
  return &archs_[devno];
}

void** Mem::arch_ptr(Device *dev, void *host) {
  arch(dev, host);
  return &archs_[dev->devno()];
}

void* Mem::arch(int devno, void *host) {
  if (archs_[devno] == NULL) {
      Device *dev = archs_dev_[devno];
      if (host == NULL || !dev->is_shared_memory_buffers()) 
          dev->MemAlloc(archs_ + devno, expansion_ * size_);
      else
          archs_[devno] = host;
  }
  return archs_[devno];
}

void* Mem::arch(Device* dev, void *host) {
  int devno = dev->devno();
  return arch(devno, host);
}

void* Mem::host_inter() {
  if (!host_inter_) host_inter_ = malloc(expansion_ * size_);
  return host_inter_;
}

#ifdef USE_MEMRANGE
void Mem::AddOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  //_trace("mem[%lu] off[%lu] size[%lu] dev[%d]", uid(), off, size, dev->devno());
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Overlap(off, size)) {
      //_trace("old[%lu,%lu,%d] new[%lu,%lu,%d]", r->off(), r->size(), r->dev()->devno(), off, size, dev->devno());
    }
  }
  ranges_.insert(new MemRange(off, size, dev));
  pthread_mutex_unlock(&mutex_);
}
#else
void Mem::AddOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  for (std::set<Device*>::iterator I = owners_.begin(), E = owners_.end(); I != E; ++I) {
    Device* owner = *I;
    if (owner == dev) {
      pthread_mutex_unlock(&mutex_);
      return;
    }
  }
  owners_.insert(dev);
  pthread_mutex_unlock(&mutex_);
}
#endif

#ifdef USE_MEMRANGE
void Mem::SetOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  bool found = false;
  //_trace("mem[%lu] off[%lu] size[%lu] dev[%d]", uid(), off, size, dev->devno());
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E;) {
    MemRange* r = *I;
    if (r->Contain(off, size) && r->dev() == dev) {
      found = true;
      ++I;
      continue;
    } else if (r->Overlap(off, size)) {
      //_todo("old[%lu,%lu,%d] new[%lu,%lu,%d]", r->off(), r->size(), r->dev()->devno(), off, size, dev->devno());
      ranges_.erase(I);
      I = ranges_.begin();
    } else ++I;
  }
  if (!found) ranges_.insert(new MemRange(off, size, dev));
  pthread_mutex_unlock(&mutex_);
}
#else
void Mem::SetOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  owners_.clear();
  owners_.insert(dev);
  pthread_mutex_unlock(&mutex_);
}

#endif

void Mem::SetOwner(Device* dev) {
  return SetOwner(0, size_, dev);
}

#ifdef USE_MEMRANGE
Device* Mem::Owner(size_t off, size_t size) {
  pthread_mutex_lock(&mutex_);
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Contain(off, size)) {
      pthread_mutex_unlock(&mutex_);
      return r->dev();
    }
  }
  pthread_mutex_unlock(&mutex_);
  return NULL;
}
#else
Device* Mem::Owner(size_t off, size_t size) {
  Device* ret = NULL;
  pthread_mutex_lock(&mutex_);
  if (!owners_.empty()) {
    ret = *(owners_.begin());
  }
  pthread_mutex_unlock(&mutex_);
  return ret;
}
#endif

Device* Mem::Owner() {
  return Owner(0, size_);
}

#ifdef USE_MEMRANGE
bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  for (std::set<MemRange*>::iterator I = ranges_.begin(), E = ranges_.end(); I != E; ++I) {
    MemRange* r = *I;
    if (r->Contain(off, size) && r->dev() == dev) {
      pthread_mutex_unlock(&mutex_);
      return true;
    }
  }
  pthread_mutex_unlock(&mutex_);
  return false;
}
#else
bool Mem::IsOwner(size_t off, size_t size, Device* dev) {
  pthread_mutex_lock(&mutex_);
  bool bret = owners_.find(dev) != owners_.end();
  pthread_mutex_unlock(&mutex_);
  return bret;
}
#endif

bool Mem::IsOwner(Device* dev) {
  return IsOwner(0, size_, dev);
}

void Mem::Reduce(int mode, int type) {
  mode_ = mode;
  type_ = type;
  switch (type_) {
    case iris_int:      type_size_ = sizeof(int);       break;
    case iris_long:     type_size_ = sizeof(long);      break;
    case iris_float:    type_size_ = sizeof(float);     break;
    case iris_double:   type_size_ = sizeof(double);    break;
    default: _error("not support type[0x%x]", type_);
  }
}

void Mem::Expand(int expansion) {
  expansion_ = expansion;
}

void Mem::SetMap(void* host, size_t size) {
  mapped_host_ = host;
  mapped_size_ = size;
}

} /* namespace rt */
} /* namespace iris */


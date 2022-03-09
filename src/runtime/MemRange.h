#ifndef IRIS_SRC_RT_MEM_RANGE_H
#define IRIS_SRC_RT_MEM_RANGE_H

#include <stddef.h>
#include <set>

namespace iris {
namespace rt {

class Device;

class MemRange {
public:
  MemRange(size_t off, size_t size, Device* dev);
  ~MemRange();

  bool Distinct(size_t off, size_t size);
  bool Overlap(size_t off, size_t size);
  bool Contain(size_t off, size_t size);

  size_t off() { return off_; }
  size_t size() { return size_; }
  Device* dev() { return dev_; }

  bool operator <(const MemRange& r) const { return off_ < r.off_; }

private:
  size_t off_;
  size_t size_;
  Device* dev_;

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_MEM_RANGE_H */


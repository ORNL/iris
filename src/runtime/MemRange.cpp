#include "MemRange.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

MemRange::MemRange(size_t off, size_t size, Device* dev) {
  off_ = off;
  size_ = size;
  dev_ = dev;
}

MemRange::~MemRange() {

}

bool MemRange::Distinct(size_t off, size_t size) {
  return (off_ > off + size - 1) || (off_ + size_ - 1 < off);
}

bool MemRange::Overlap(size_t off, size_t size) {
  return !Distinct(off, size);
}

bool MemRange::Contain(size_t off, size_t size) {
  return (off_ <= off) && (off_ + size_ >= off + size);
}

} /* namespace rt */
} /* namespace brisbane */


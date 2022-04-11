#include "QueueReady.h"

namespace iris {
namespace rt {

QueueReady::QueueReady(unsigned long size) {
  size_ = size;
  idx_r_ = 0;
  idx_w_ = 0;
  idx_w_cas_ = 0;
  elements_ = (volatile Task**)(new Task*[size_]);
}

QueueReady::~QueueReady() {
  delete[] elements_;
}

bool QueueReady::Enqueue(Task* task) {
  while (true) {
    unsigned long prev_idx_w = idx_w_cas_;
    unsigned long next_idx_w = (prev_idx_w + 1) % this->size_;
    if (next_idx_w == this->idx_r_) return false;
    if (__sync_bool_compare_and_swap(&idx_w_cas_, prev_idx_w, next_idx_w)) {
      this->elements_[prev_idx_w] = task;
      while (!__sync_bool_compare_and_swap(&this->idx_w_, prev_idx_w, next_idx_w)) {}
      break;
    }
  }
  return true;
}

bool QueueReady::Dequeue(Task** task) {
  if (idx_r_ == idx_w_) return false;
  unsigned long next_idx_r = (idx_r_ + 1) % size_;
  *task = (Task*) elements_[idx_r_];
  idx_r_ = next_idx_r;
  return true;
}

size_t QueueReady::Size() {
  if (idx_w_ >= idx_r_) return idx_w_ - idx_r_;
  return size_ - idx_r_ + idx_w_;
}

bool QueueReady::Empty() {
  return Size() == 0UL;
}

} /* namespace rt */
} /* namespace iris */


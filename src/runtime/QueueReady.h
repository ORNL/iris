#ifndef IRIS_SRC_RT_QUEUE_READY_H
#define IRIS_SRC_RT_QUEUE_READY_H

#include "Queue.h"

namespace iris {
namespace rt {

class QueueReady : public Queue {
public:
  QueueReady(unsigned long size);
  virtual ~QueueReady();

  bool Enqueue(Task* task);
  bool Dequeue(Task** task);
  size_t Size();
  bool Empty();

private:
  unsigned long size_;
  volatile Task** elements_;
  volatile unsigned long idx_r_;
  volatile unsigned long idx_w_;
  volatile unsigned long idx_w_cas_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_QUEUE_READY_H */

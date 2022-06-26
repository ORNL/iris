#ifndef IRIS_SRC_RT_QUEUE_READY_H
#define IRIS_SRC_RT_QUEUE_READY_H

#include "Queue.h"

#include <queue>
#include <mutex>

namespace iris {
namespace rt {

class QueueReady : public Queue {
public:
  QueueReady();
  virtual ~QueueReady();

  bool Peek(Task** task, int index);
  bool Enqueue(Task* task);
  bool Dequeue(Task** task);
  size_t Size();
  bool Empty();

private:
  std::deque<Task*> pqueue_, queue_;
  mutable std::mutex mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_QUEUE_READY_H */

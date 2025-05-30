#ifndef IRIS_SRC_RT_QUEUE_READY_H
#define IRIS_SRC_RT_QUEUE_READY_H

#include "Queue.h"

#include <queue>
#include <mutex>
namespace iris {
namespace rt {
using namespace std;
class QueueReady : public Queue {
public:
  QueueReady();
  virtual ~QueueReady();

  bool Peek(Task** task, int index);
  bool Enqueue(Task* task);
  bool Dequeue(Task** task);
  bool Dequeue(Task** task, Device *device);
  size_t Size();
  bool Empty();
  void Print(int devno=-1);

private:
  std::deque<Task*> pqueue_, queue_, mqueue_;
  mutable std::mutex mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_QUEUE_READY_H */

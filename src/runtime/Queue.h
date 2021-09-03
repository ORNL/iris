#ifndef BRISBANE_SRC_RT_QUEUE_H
#define BRISBANE_SRC_RT_QUEUE_H

#include "Config.h"

namespace brisbane {
namespace rt {

class Task;

class Queue {
public:
  virtual ~Queue() {}

  virtual bool Enqueue(Task* task) = 0;
  virtual bool Dequeue(Task** task) = 0;
  virtual size_t Size() = 0;
  virtual bool Empty() = 0;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_QUEUE_H */

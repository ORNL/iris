#ifndef IRIS_SRC_RT_QUEUE_H
#define IRIS_SRC_RT_QUEUE_H

#include "Config.h"
#include <utility>

namespace iris {
namespace rt {

class Task;
class Device;

class Queue {
public:
  virtual ~Queue() {}
  virtual bool Peek(Task** task, int index) = 0;
  virtual bool Enqueue(Task* task) = 0;
  virtual bool Dequeue(Task** task) = 0;
  virtual bool Dequeue(Task** task, Device *dev) { return Dequeue(task); }
  virtual size_t Size() = 0;
  virtual bool Empty() = 0;
  virtual void Print(int devno=-1) { }
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_QUEUE_H */

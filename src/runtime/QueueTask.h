#ifndef IRIS_SRC_RT_QUEUE_TASK_H
#define IRIS_SRC_RT_QUEUE_TASK_H

#include "Task.h"
#include "Queue.h"
#include <pthread.h>
#include <list>

namespace iris {
namespace rt {

class Platform;

class QueueTask : public Queue {
public:
  QueueTask(Platform* platform);
  ~QueueTask();
  bool Peek(Task** task, int index);
  bool Enqueue(Task* task);
  bool Dequeue(Task** task);
  bool Dequeue(pair<unsigned long, Task*>* task) { return Dequeue(task); }
  size_t Size();
  bool Empty();

private:
  Platform* platform_;
  std::list<Task*> tasks_;
  pthread_mutex_t mutex_;
  Task* last_sync_task_;
  bool enable_profiler_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_QUEUE_TASK_H */

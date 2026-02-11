#ifndef IRIS_SRC_RT_WORKER_H
#define IRIS_SRC_RT_WORKER_H

#include "Thread.h"
#include "Timer.h"
#include "Queue.h"

namespace iris {
namespace rt {

class Consistency;
class Device;
class Platform;
class Queue;
class ReadyQueue;
class Scheduler;
class Task;

class Worker : public Thread {
public:
  Worker(Device* dev, Platform* platform, bool single = false);
  virtual ~Worker();

  void Enqueue(Task* task);
  void TaskComplete(Task* task);

  bool busy() { return busy_; }
  unsigned long ntasks();
  size_t size() { return queue_->Size(); }
  Device* device() { return dev_; }
  Platform* platform() { return platform_; }
  double Now() { return timer_->Now(); }
  Timer *timer() { return timer_; }
private:
  void Execute(Task* task);
  virtual void Run();

private:
  Platform* platform_;
  Queue* queue_;
  Consistency* consistency_;
  Device* dev_;
  Scheduler* scheduler_;
  bool single_;
  bool busy_;
protected:
  Timer* timer_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_WORKER_H */

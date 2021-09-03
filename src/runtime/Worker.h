#ifndef BRISBANE_SRC_RT_WORKER_H
#define BRISBANE_SRC_RT_WORKER_H

#include "Thread.h"

namespace brisbane {
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
  Device* device() { return dev_; }

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
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_WORKER_H */

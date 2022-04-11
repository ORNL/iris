#ifndef IRIS_SRC_RT_SCHEDULER_H
#define IRIS_SRC_RT_SCHEDULER_H

#include "Config.h"
#include "Thread.h"
#include <pthread.h>

namespace iris {
namespace rt {

class Consistency;
class Device;
class HubClient;
class Task;
class Timer;
class Platform;
class Policies;
class Profiler;
class Queue;
class Worker;

class Scheduler : public Thread {
public:
  Scheduler(Platform* platform);
  virtual ~Scheduler();

  void Enqueue(Task* task);
  void SubmitTaskDirect(Task* task, Device* dev);

  Platform* platform() { return platform_; }
  Device** devices() { return devs_; }
  Worker** workers() { return workers_; }
  Worker* worker(int i) { return workers_[i]; }
  Consistency* consistency() { return consistency_; }
  Policies* policies() { return policies_; }
  int ndevs() { return ndevs_; }
  int nworkers() { return ndevs_; }
  void StartTask(Task* task, Worker* worker);
  void CompleteTask(Task* task, Worker* worker);
  bool hub_available() { return hub_available_; }
  bool enable_profiler() { return enable_profiler_; }
  int RefreshNTasksOnDevs();
  size_t NTasksOnDev(int i);

private:
  void Submit(Task* task);
  void SubmitTask(Task* task);
  virtual void Run();

  void InitHubClient();

private:
  Queue* queue_;
  Platform* platform_;

  Policies* policies_;
  Device** devs_;
  Worker** workers_;
  Consistency* consistency_;
  size_t ntasks_on_devs_[IRIS_MAX_NDEVS];
  int ndevs_;
  HubClient* hub_client_;
  bool hub_available_;
  bool enable_profiler_;
  int nprofilers_;
  Profiler** profilers_;
  Timer* timer_;
  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_SCHEDULER_H */

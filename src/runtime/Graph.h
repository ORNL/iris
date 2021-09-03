#ifndef BRISBANE_SRC_RT_GRAPH_H
#define BRISBANE_SRC_RT_GRAPH_H

#include "Retainable.h"
#include "Command.h"
#include "Platform.h"
#include <pthread.h>
#include <vector>

namespace brisbane {
namespace rt {

class Graph: public Retainable<struct _brisbane_graph, Graph> {
public:
  Graph(Platform* platform);
  virtual ~Graph();

  void AddTask(Task* task);
  void Submit();
  void Complete();
  void Wait();

  Platform* platform() { return platform_; }
  std::vector<Task*>* tasks() { return &tasks_; }

private:
  Platform* platform_;
  Scheduler* scheduler_;
  std::vector<Task*> tasks_;

  Task* start_;
  Task* end_;

  int status_;
  
  pthread_mutex_t mutex_complete_;
  pthread_cond_t complete_cond_;

public:
  static Graph* Create(Platform* platform);
  Task* end() { return end_; }
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_GRAPH_H */

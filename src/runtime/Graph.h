#ifndef IRIS_SRC_RT_GRAPH_H
#define IRIS_SRC_RT_GRAPH_H

#include "Retainable.h"
#include "Command.h"
#include "Platform.h"
#include <pthread.h>
#include <vector>

namespace iris {
namespace rt {

class Graph: public Retainable<struct _iris_graph, Graph> {
public:
  Graph(Platform* platform);
  virtual ~Graph();

  void AddTask(Task* task);
  void Submit();
  void Complete();
  void Wait();

  Platform* platform() { return platform_; }
  std::vector<Task*>* tasks() { return &tasks_; }
  int iris_tasks(iris_task *pv);
  int tasks_count() { return tasks_.size(); }
  bool is_retainable() { return retain_tasks_; }
  void enable_retainable() { retain_tasks_ = true; }
  void disable_retainable() { retain_tasks_ = false; }
private:
  Platform* platform_;
  Scheduler* scheduler_;
  std::vector<Task*> tasks_;

  //Task* start_;
  Task* end_;

  bool retain_tasks_;
  int status_;
  
  pthread_mutex_t mutex_complete_;
  pthread_cond_t complete_cond_;

public:
  static Graph* Create(Platform* platform);
  Task* end() { return end_; }
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_GRAPH_H */

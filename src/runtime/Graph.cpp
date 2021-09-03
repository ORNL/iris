#include "Graph.h"
#include "Debug.h"
#include "Scheduler.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Graph::Graph(Platform* platform) {
  platform_ = platform;
  if (platform) scheduler_ = platform_->scheduler();
  status_ = BRISBANE_NONE;

  end_ = Task::Create(platform_, BRISBANE_TASK_PERM, NULL);
  tasks_.push_back(end_);

  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_cond_init(&complete_cond_, NULL);
}

Graph::~Graph() {
  pthread_mutex_destroy(&mutex_complete_);
  pthread_cond_destroy(&complete_cond_);
  if (end_) delete end_;
}

void Graph::AddTask(Task* task) {
  tasks_.push_back(task);
  end_->AddDepend(task);
}

void Graph::Submit() {
  status_ = BRISBANE_SUBMITTED;
}

void Graph::Complete() {
  pthread_mutex_lock(&mutex_complete_);
  status_ = BRISBANE_COMPLETE;
  pthread_cond_broadcast(&complete_cond_);
  pthread_mutex_unlock(&mutex_complete_);
}

void Graph::Wait() {
  end_->Wait();
}

Graph* Graph::Create(Platform* platform) {
  return new Graph(platform);
}

} /* namespace rt */
} /* namespace brisbane */

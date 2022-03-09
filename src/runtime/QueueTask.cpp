#include "QueueTask.h"
#include "Platform.h"
#include "Debug.h"

namespace iris {
namespace rt {

QueueTask::QueueTask(Platform* platform) {
  platform_ = platform;
  enable_profiler_ = platform->enable_profiler();
  last_sync_task_ = NULL;
  pthread_mutex_init(&mutex_, NULL);
}

QueueTask::~QueueTask() {
  pthread_mutex_destroy(&mutex_);
}

bool QueueTask::Dequeue(Task** task) {
  pthread_mutex_lock(&mutex_);
  if (tasks_.empty()) {
    pthread_mutex_unlock(&mutex_);
    return false;
  }
  for (std::list<Task*>::iterator I = tasks_.begin(), E = tasks_.end(); I != E; ++I) {
    Task* t = *I;
    if (!t->Dispatchable()) continue;
    if (t->marker() && I != tasks_.begin()) continue;
    *task = t;
    //todo: debug this!
    tasks_.erase(I);
    pthread_mutex_unlock(&mutex_);
    return true;
  }
  pthread_mutex_unlock(&mutex_);
  return false;
}

bool QueueTask::Enqueue(Task* task) {
  pthread_mutex_lock(&mutex_);
  tasks_.push_back(task);
  if (enable_profiler_) {
    if (last_sync_task_) task->AddDepend(last_sync_task_);
    if (last_sync_task_ && task->sync()) last_sync_task_->Release();
    if (task->sync()) {
      last_sync_task_ = task;
      last_sync_task_->Retain();
    }
  }
  pthread_mutex_unlock(&mutex_);
  return true;
}

size_t QueueTask::Size() {
  pthread_mutex_lock(&mutex_);
  size_t size = tasks_.size();
  pthread_mutex_unlock(&mutex_);
  return size;
}

bool QueueTask::Empty() {
  pthread_mutex_lock(&mutex_);
  bool empty = tasks_.empty();
  pthread_mutex_unlock(&mutex_);
  return empty;
}

} /* namespace rt */
} /* namespace iris */

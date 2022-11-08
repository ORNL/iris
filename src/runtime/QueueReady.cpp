#include "QueueReady.h"
#include "Task.h"

namespace iris {
namespace rt {

QueueReady::QueueReady() {
}

QueueReady::~QueueReady() {
}

bool QueueReady::Peek(Task** task, int target_index){
  std::lock_guard<std::mutex> lock(mutex_);
  if (target_index <= (int)pqueue_.size()){
    *task = pqueue_[target_index];
  }
  else{
    target_index -= pqueue_.size();
    *task = queue_[target_index];
  }
  return true;
}

bool QueueReady::Enqueue(Task* task) {
  std::lock_guard<std::mutex> lock(mutex_);
  //if the task to be enqueued is a memory transfer it should be prioritized
  if (task->ncmds_memcpy() == task->ncmds()) {
    pqueue_.push_back(task);
  }
  else{
    queue_.push_back(task);
  }
  return true;
}

bool QueueReady::Dequeue(Task** task) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pqueue_.empty()){
    *task = (Task*) pqueue_.front();
    pqueue_.pop_front();
    return true;
  }
  if (!queue_.empty()){
    *task = (Task*) queue_.front();
    queue_.pop_front();
    return true;
  }
  return false;
}

size_t QueueReady::Size() {
  return (size_t)(pqueue_.size() + queue_.size());
}

bool QueueReady::Empty() {
  return pqueue_.empty() && queue_.empty();
}

} /* namespace rt */
} /* namespace iris */


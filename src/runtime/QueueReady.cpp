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

void QueueReady::Print(int devno) {
  printf("Queue data: (%lu) devno:%d:%p --- ", pqueue_.size(), devno, this);
  for(size_t i=0; i<pqueue_.size(); i++) {
    printf("%lu:%p ", pqueue_[i]->uid(), pqueue_[i]);
  }
  printf("\n");
}

bool QueueReady::Enqueue(Task* task) {
  std::lock_guard<std::mutex> lock(mutex_);
  //if the task to be enqueued is a memory transfer it should be prioritized
  if (task->marker()) {
    queue_.push_back(task);
    _trace("Pushed marker task:%lu:%s to queue pq:%lu q:%lu", task->uid(), task->name(), pqueue_.size(), queue_.size());
  }
  else if (task->ncmds_memcpy() == task->ncmds()) {
    pqueue_.push_back(task);
    _trace("Pushed task:%lu:%s to pqueue pq:%lu q:%lu", task->uid(), task->name(), pqueue_.size(), queue_.size());
  }
  else{
    queue_.push_back(task);
    _trace("Pushed task:%lu:%s to queue pq:%lu q:%lu", task->uid(), task->name(), pqueue_.size(), queue_.size());
  }
  return true;
}

bool QueueReady::Dequeue(Task** task) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pqueue_.empty()){
    *task = (Task*) pqueue_.front();
    _trace("Popped task:%lu:%s to pqueue pq:%lu q:%lu", (*task)->uid(), (*task)->name(), pqueue_.size(), queue_.size());
    pqueue_.pop_front();
    return true;
  }
  if (!queue_.empty()){
    *task = (Task*) queue_.front();
    _trace("Popped task:%lu:%s to queue pq:%lu q:%lu", (*task)->uid(), (*task)->name(), pqueue_.size(), queue_.size());
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


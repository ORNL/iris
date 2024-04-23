#include "QueueReady.h"
#include "Task.h"
#include "Device.h"
#include <utility>

#define FLUSH_INDEPENDENT_QUEUE
using namespace std;
namespace iris {
namespace rt {

QueueReady::QueueReady() {
}

QueueReady::~QueueReady() {
}

bool QueueReady::Peek(Task** task, int target_index){
  std::lock_guard<std::mutex> lock(mutex_);
  if (target_index <= (int)pqueue_.size()){
    auto & data = pqueue_[target_index];
    *task = data;
  }
  else{
    target_index -= pqueue_.size();
    auto & data = queue_[target_index];
    *task = data;
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
    mqueue_.push_back(task);
    _trace("Pushed marker task:%lu:%s to queue pq:%lu q:%lu", task->uid(), task->name(), pqueue_.size(), mqueue_.size());
  }
#ifdef FLUSH_INDEPENDENT_QUEUE
  else if (task->ncmds_memcpy() == task->ncmds()) {
    pqueue_.push_back(task);
    _trace("Pushed task:%lu:%s to pqueue pq:%lu q:%lu", task->uid(), task->name(), pqueue_.size(), queue_.size());
  }
#endif
  else{
    queue_.push_back(task);
    _trace("Pushed task:%lu:%s to queue pq:%lu q:%lu", task->uid(), task->name(), pqueue_.size(), queue_.size());
  }
  return true;
}

bool QueueReady::Dequeue(Task **task) {
  std::lock_guard<std::mutex> lock(mutex_);
#ifdef FLUSH_INDEPENDENT_QUEUE
  if (!pqueue_.empty()){
    auto &data = pqueue_.front();
    *task = data;
    _trace("Popped task:%lu:%s to pqueue pq:%lu q:%lu", (*task)->uid(), (*task)->name(), pqueue_.size(), queue_.size());
    pqueue_.pop_front();
    return true;
  }
#endif
  if (!queue_.empty()){
    auto &data = queue_.front();
    *task = data;
    _trace("Popped task:%lu:%s to queue pq:%lu q:%lu", (*task)->uid(), (*task)->name(), pqueue_.size(), queue_.size());
    queue_.pop_front();
    return true;
  }
  if (!mqueue_.empty()){
    auto &data = mqueue_.front();
    *task = data;
    _trace("Popped task:%lu:%s to mqueue mq:%lu q:%lu", (*task)->uid(), (*task)->name(), mqueue_.size(), queue_.size());
    mqueue_.pop_front();
    return true;
  }
  return false;
}
bool QueueReady::Dequeue(Task **task, Device *device) {
  std::lock_guard<std::mutex> lock(mutex_);
#ifdef FLUSH_INDEPENDENT_QUEUE
  if (!pqueue_.empty()){
    auto &data = pqueue_.front();
    *task = data; //(Task*) data.second;
    _trace("Popped task:%lu:%s to pqueue pq:%lu q:%lu", data->uid(), data->name(), pqueue_.size(), queue_.size());
    pqueue_.pop_front();
    return true;
  }
#endif
  if (!queue_.empty()){
    auto &data = queue_.front();
    *task = data; //(Task*) data.second;
    _trace("Popped task:%lu:%s to queue pq:%lu q:%lu", data->uid(), data->name(), pqueue_.size(), queue_.size());
    queue_.pop_front();
    return true;
  }
  if (!mqueue_.empty() && device->IsFree()){
    auto &data = mqueue_.front();
    *task = data; //(Task*) data.second;
    _trace("Popped task:%lu:%s to mqueue pq:%lu q:%lu", data->uid(), data->name(), mqueue_.size(), queue_.size());
    mqueue_.pop_front();
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


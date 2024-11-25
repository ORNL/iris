#include "Worker.h"
#include "Debug.h"
#include "Consistency.h"
#include "Device.h"
#include "Platform.h"
#include "QueueReady.h"
#include "Scheduler.h"
#include "Task.h"
#include "Utils.h"
#include "Timer.h"

namespace iris {
namespace rt {

Worker::Worker(Device* dev, Platform* platform, bool single) {
  dev_ = dev;
  platform_ = platform;
  //Utils::SetThreadAffinity(dev_->devno());
  dev_->set_worker(this);
  scheduler_ = platform_->scheduler();
  if (scheduler_) consistency_ = scheduler_->consistency();
  else consistency_ = NULL;
  single_ = single;
  if (single) queue_ = platform->queue();
  else queue_ = new QueueReady();
  busy_ = false;
}

Worker::~Worker() {
  _trace("Worker is destroyed");
  //if (sleeping_) {
      running_ = false;
      Invoke();
      while(running_);
  //}
  if (!single_) delete queue_;
}

void Worker::TaskComplete(Task* task) {
  _debug2("task:%lu:%s ref_cnt:%d\n", task->uid(), task->name(), task->ref_cnt());
  _debug2("now invoke scheduler after task:%lu:%s qsize:%lu", task->uid(), task->name(), queue_->Size());
  if (scheduler_) scheduler_->Invoke();
  _debug2("task:%lu:%s ref_cnt:%d\n", task->uid(), task->name(), task->ref_cnt());
  //task->set_devno(dev_->devno());
  //_trace("now invoke worker after task:%lu:%s qsize:%lu", task->uid(), task->name(), queue_->Size());
  Invoke();
  _debug2("task:%lu:%s ref_cnt:%d\n", task->uid(), task->name(), task->ref_cnt());
}

void Worker::Enqueue(Task* task) {
    _debug2("Inside enqueue of task:%lu:%s\n", task->uid(), task->name());
  //if we're running an internal task (such as an internal memory movement for consistency) skip the queue.
  if (task->is_internal_memory_transfer()) {
    dev_->Synchronize();
    Execute(task);
    return;
  }
  //printf("Enqueuing task:%lu:%s:%p devno:%d\n", task->uid(), task->name(), task, dev_->devno());
  _debug2("Enqueuing task:%lu:%s:%p devno:%d", task->uid(), task->name(), task, dev_->devno());
  while (!queue_->Enqueue(task)) { }
  _debug2("Invoking worker for task:%lu:%s qsize:%lu", task->uid(), task->name(), queue_->Size());
  Invoke();
}

void Worker::Execute(Task* task) {
  _debug2("check executable for task:%lu:%s:%p qsize:%lu dev:%d ref_cnt:%d\n", task->uid(), task->name(), task, queue_->Size(), dev_->devno(), task->ref_cnt());
  //queue_->Print(dev_->devno());
  if (!task->Executable()) { task->Release(); return; }
  task->ChangeToProcessMode();
  task->Release(); // This is for Scheduler::Submit task retain
  task->set_dev(dev_);
  if (task->marker()) {
    dev_->FreeDestroyEvents();
    dev_->Synchronize();
    task->Complete();
    //task->TryReleaseTask();
    return;
  }
  busy_ = true;
  if (scheduler_) scheduler_->StartTask(task, this);
  if (consistency_) consistency_->Resolve(task);
  bool task_cmd_last = task->cmd_last();
  bool user_task = task->user();
  
  task->Retain();
  _debug2("Task %s:%lu refcnt:%d", task->name(), task->uid(), task->ref_cnt());
  task->set_devno(dev_->devno());
  dev_->Execute(task);
  _debug2("Task %s:%lu refcnt:%d after execute", task->name(), task->uid(), task->ref_cnt());
  if (task_cmd_last) {
    if (scheduler_) scheduler_->CompleteTask(task, this);
    //task->Complete();
  }
#ifdef _DEBUG2_ENABLE
  unsigned long uid = task->uid(); string name = task->name(); _debug2("Task %s:%lu refcnt:%d before release", name.c_str(), uid, task->ref_cnt());
#endif
  int ref_cnt = task->Release(); //Device::Execute Retain call
  _debug2("Task %s:%lu refcnt:%d after release", name.c_str(), uid, ref_cnt);
  //task->TryReleaseTask();
  busy_ = false;
}

void Worker::Run() {
  while (true) {
    _debug2("Device:%d:%s Worker entering into sleep mode", dev_->devno(), dev_->name());
    //printf("1Device:%d:%s Queue size:%lu\n", dev_->devno(), dev_->name(), queue_->Size());
    _debug2("Device:%d:%s Worker thread invoked now running:%d", dev_->devno(), dev_->name(), running_);
    //printf("2Device:%d:%s Queue size:%lu\n", dev_->devno(), dev_->name(), queue_->Size());
    if (!running_) break;
    Task* task;
    _debug2("Device:%d:%s Queue size:%lu", dev_->devno(), dev_->name(), queue_->Size());
    while (running_ && queue_->Dequeue(&task, dev_)){
      //printf("Device:%d:%s Qsize:%lu dequeued task:%lu:%s:%p\n", dev_->devno(), dev_->name(), queue_->Size(), task->uid(), task->name(), task);
      _debug2("Device:%d:%s Qsize:%lu dequeued task:%lu:%s:%p", dev_->devno(), dev_->name(), queue_->Size(), task.first, task->name(), task);
      //FIX: This check is not needed. For the policies like ALL, when the task is in some worker queue, it couldn't be deleted unless all worker queues release them
      //if (!Platform::GetPlatform()->is_task_exist(task.first)) continue;
#ifdef _DEBUG2_ENABLE
      task->Retain();
#endif
      Execute(task);
      _debug2("Completed task Device:%d:%s Qsize:%lu dequeued, task:%p", dev_->devno(), dev_->name(), queue_->Size(), task);
#ifdef _DEBUG2_ENABLE
      task->Release(); // For Worker::Run Retain call
#endif
    }
    Sleep();
  }
  _debug2("Device:%d:%s Worker thread exited\n\n", dev_->devno(), dev_->name());
}

unsigned long Worker::ntasks() {
  return queue_->Size() + (busy_ ? 1 : 0);
}

} /* namespace rt */
} /* namespace iris */

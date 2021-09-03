#include "Scheduler.h"
#include "Debug.h"
#include "Consistency.h"
#include "Device.h"
#include "HubClient.h"
#include "Platform.h"
#include "Policies.h"
#include "Policy.h"
#include "Profiler.h"
#include "QueueTask.h"
#include "Task.h"
#include "Timer.h"
#include "Worker.h"

namespace brisbane {
namespace rt {

Scheduler::Scheduler(Platform* platform) {
  platform_ = platform;
  devs_ = platform_->devices();
  ndevs_ = platform_->ndevs();
  queue_ = platform->queue();
  workers_ = platform_->workers();
  enable_profiler_ = platform->enable_profiler();
  nprofilers_ = platform->nprofilers();
  profilers_ = platform->profilers();
  pthread_mutex_init(&mutex_, NULL);
  consistency_ = new Consistency(this);
  policies_ = new Policies(this);
  hub_client_ = new HubClient(this);
  timer_ = new Timer();
  InitHubClient();
}

Scheduler::~Scheduler() {
  delete consistency_;
  delete policies_;
  delete hub_client_;
  pthread_mutex_destroy(&mutex_);
}

void Scheduler::InitHubClient() {
  hub_available_ = hub_client_->Init() == BRISBANE_OK;
}

void Scheduler::StartTask(Task* task, Worker* worker) {
  task->set_time_start(timer_->Now());
}

void Scheduler::CompleteTask(Task* task, Worker* worker) {
  Device* dev = worker->device();
  int devno = dev->devno();
  if (hub_available_) hub_client_->TaskDec(devno, 1);
  if (enable_profiler_ & !task->system()) {
    task->set_time_end(timer_->Now());
    pthread_mutex_lock(&mutex_);
    _todo("remove lock profile[%d]", enable_profiler_);
    for (int i = 0; i < nprofilers_; i++) profilers_[i]->CompleteTask(task); 
    pthread_mutex_unlock(&mutex_);
  }
}

int Scheduler::RefreshNTasksOnDevs() {
  if (!hub_available_) {
    for (int i = 0; i < ndevs_; i++) ntasks_on_devs_[i] = workers_[i]->ntasks();
    return BRISBANE_OK;
  }
  hub_client_->TaskAll(ntasks_on_devs_, ndevs_);
  return BRISBANE_OK;
}

size_t Scheduler::NTasksOnDev(int i) {
  return ntasks_on_devs_[i];
}

void Scheduler::Enqueue(Task* task) {
  while (!queue_->Enqueue(task)) { }
  Invoke();
}

void Scheduler::Run() {
  while (true) {
    Sleep();
    if (!running_) break;
    Task* task = NULL;
    while (queue_->Dequeue(&task)) Submit(task);
  }
}

void Scheduler::SubmitTaskDirect(Task* task, Device* dev) {
  dev->worker()->Enqueue(task);
  if (hub_available_) hub_client_->TaskInc(dev->devno(), 1);
}

void Scheduler::Submit(Task* task) {
  if (!ndevs_) {
    if (!task->marker()) _error("%s", "no device");
    task->Complete();
    return;
  }
  if (task->marker()) {
    std::vector<Task*>* subtasks = task->subtasks();
    for (std::vector<Task*>::iterator I = subtasks->begin(), E = subtasks->end(); I != E; ++I) {
      Task* subtask = *I;
      int dev = subtask->devno();
      workers_[dev]->Enqueue(subtask);
    }
    return;
  }
  if (!task->HasSubtasks()) {
    SubmitTask(task);
    return;
  }
  std::vector<Task*>* subtasks = task->subtasks();
  for (std::vector<Task*>::iterator I = subtasks->begin(), E = subtasks->end(); I != E; ++I)
    SubmitTask(*I);
}

void Scheduler::SubmitTask(Task* task) {
  int brs_policy = task->brs_policy();
  char* opt = task->opt();
  int ndevs = 0;
  Device* devs[BRISBANE_MAX_NDEVS];
  if (brs_policy < BRISBANE_MAX_NDEVS) {
    if (brs_policy >= ndevs_) ndevs = 0;
    else {
      ndevs = 1;
      devs[0] = devs_[brs_policy];
    }
  } else policies_->GetPolicy(brs_policy, opt)->GetDevices(task, devs, &ndevs);
  if (ndevs == 0) {
    int dev_default = platform_->device_default();
    _trace("no device for policy[0x%x], run the task on device[%d]", brs_policy, dev_default);
    ndevs = 1;
    devs[0] = devs_[dev_default];
  }
  for (int i = 0; i < ndevs; i++) {
    devs[i]->worker()->Enqueue(task);
    if (hub_available_) hub_client_->TaskInc(devs[i]->devno(), 1);
  }
}

} /* namespace rt */
} /* namespace brisbane */

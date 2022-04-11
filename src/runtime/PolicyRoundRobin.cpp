#include "PolicyRoundRobin.h"
#include "Debug.h"
#include "Device.h"
#include "Scheduler.h"
#include "Task.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace iris {
namespace rt {

PolicyRoundRobin::PolicyRoundRobin(Scheduler* scheduler) {
  SetScheduler(scheduler);
  index_ = 0;
}

PolicyRoundRobin::~PolicyRoundRobin() {
}

void PolicyRoundRobin::GetDevices(Task* task, Device** devs, int* ndevs) {
  int policy = task->brs_policy();
  if (policy == iris_roundrobin) return GetDevice(task, devs, ndevs);
  return GetDeviceType(task, devs, ndevs);
}

void PolicyRoundRobin::GetDevice(Task* task, Device** devs, int* ndevs) {
  devs[0] = devs_[index_];
  *ndevs = 1;
  if (++index_ == ndevs_) index_ = 0;
}

void PolicyRoundRobin::GetDeviceType(Task* task, Device** devs, int* ndevs) {
  int policy = task->brs_policy();
  for (int i = 0; i < ndevs_; i++) {
    if (devs_[index_]->type() & policy) {
      devs[0] = devs_[index_];
      *ndevs = 1;
      if (++index_ == ndevs_) index_ = 0;
      return;
    }
    if (++index_ == ndevs_) index_ = 0;
  }
  *ndevs = 0;
}

} /* namespace rt */
} /* namespace iris */


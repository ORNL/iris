#include "PolicyShortestDeviceQueue.h"
#include "Debug.h"
#include "Device.h"
#include "Scheduler.h"
#include "Platform.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace iris {
namespace rt {

PolicyShortestDeviceQueue::PolicyShortestDeviceQueue(Scheduler* scheduler) {//Shortest-Device-Queue
  SetScheduler(scheduler);
}

PolicyShortestDeviceQueue::~PolicyShortestDeviceQueue() {
}

void PolicyShortestDeviceQueue::GetDevices(Task* task, Device** devs, int* ndevs) {
  unsigned long min = ULONG_MAX;
  int min_dev = 0;
  scheduler_->RefreshNTasksOnDevs();
  for (int i = 0; i < ndevs_; i++) {
    if (!IsKernelSupported(task, devs_[i])) continue;
    int active_tasks = 0; 
    if (scheduler_->platform()->is_async()) active_tasks = devs_[i]->active_tasks();
    unsigned long n = scheduler_->NTasksOnDev(i) + active_tasks;
    if (n == 0) {
      min_dev = i;
      break;
    }
    if (n < min) {
      min = n;
      min_dev = i;
    }
  }
  devs[0] = devs_[min_dev];
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace iris */


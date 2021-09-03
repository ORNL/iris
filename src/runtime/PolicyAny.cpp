#include "PolicyAny.h"
#include "Debug.h"
#include "Scheduler.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace brisbane {
namespace rt {

PolicyAny::PolicyAny(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyAny::~PolicyAny() {
}

void PolicyAny::GetDevices(Task* task, Device** devs, int* ndevs) {
  unsigned long min = ULONG_MAX;
  int min_dev = 0;
  scheduler_->RefreshNTasksOnDevs();
  for (int i = 0; i < ndevs_; i++) {
    unsigned long n = scheduler_->NTasksOnDev(i);
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
} /* namespace brisbane */


#include "PolicyAll.h"
#include "Debug.h"
#include "Scheduler.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace brisbane {
namespace rt {

PolicyAll::PolicyAll(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyAll::~PolicyAll() {
}

void PolicyAll::GetDevices(Task* task, Device** devs, int* ndevs) {
  for (int i = 0; i < ndevs_; i++) devs[i] = devs_[i];
  *ndevs = ndevs_;
}

} /* namespace rt */
} /* namespace brisbane */


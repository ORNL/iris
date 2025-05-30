#include "PolicyFirstToFinish.h"
#include "Debug.h"
#include "Scheduler.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace iris {
namespace rt {

PolicyFirstToFinish::PolicyFirstToFinish(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyFirstToFinish::~PolicyFirstToFinish() {
}

void PolicyFirstToFinish::GetDevices(Task* task, Device** devs, int* ndevs) {
  int n = 0;
  for (int i = 0; i < ndevs_; i++) 
      if (IsKernelSupported(task, devs_[i]))
          devs[n++] = devs_[i];
  *ndevs = n;
}

} /* namespace rt */
} /* namespace iris */


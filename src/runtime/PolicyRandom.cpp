#include "PolicyRandom.h"
#include "Debug.h"
#include <stdlib.h>
#include <time.h>
#include "Task.h"
#include <unistd.h>

namespace iris {
namespace rt {

PolicyRandom::PolicyRandom(Scheduler* scheduler) {
  SetScheduler(scheduler);
  srand(time(NULL));
}

PolicyRandom::~PolicyRandom() {
}

void PolicyRandom::GetDevices(Task* task, Device** devs, int* ndevs) {
  int selected = 0;
  int supported = 0;
  devs[0] = devs_[0];
  for(int i=0; i<ndevs_; i++) {
      if (IsKernelSupported(task, devs_[i])) 
          devs[supported++] = devs_[i];
  }
  if (supported > 0)
      devs[0] = devs[rand() % supported];
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace iris */

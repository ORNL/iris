#include "PolicyRandom.h"
#include "Debug.h"
#include <stdlib.h>
#include <time.h>
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
  for(int i=0; i<ndevs_; i++) {
      selected = rand() % ndevs_;
      if (IsKernelSupported(task, devs_[selected])) break;
  }
  devs[0] = devs_[selected];
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace iris */

#include "PolicyRandom.h"
#include "Debug.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

namespace brisbane {
namespace rt {

PolicyRandom::PolicyRandom(Scheduler* scheduler) {
  SetScheduler(scheduler);
  srand(time(NULL));
}

PolicyRandom::~PolicyRandom() {
}

void PolicyRandom::GetDevices(Task* task, Device** devs, int* ndevs) {
  devs[0] = devs_[rand() % ndevs_];
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace brisbane */

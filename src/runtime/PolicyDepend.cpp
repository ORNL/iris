#include "PolicyDepend.h"
#include "Debug.h"
#include "Device.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicyDepend::PolicyDepend(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyDepend::~PolicyDepend() {
}

void PolicyDepend::GetDevices(Task* task, Device** devs, int* ndevs) {
  int ntargets = 0;
  int ndepends = task->ndepends();
  Task** depends = task->depends();
  for (int i = 0; i < ndepends; i++) {
    Task* dep = depends[i];
    Device* dev = dep->dev();
    bool found = false;
    for (int j = 0; j < ntargets; j++) {
      if (devs[j] == dev) {
        found = true;
        break;
      }
    }
    if (!found) {
      devs[ntargets] = dev;
      *ndevs = ++ntargets;
    }
  }
}

} /* namespace rt */
} /* namespace brisbane */

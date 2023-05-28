#include "PolicyDepend.h"
#include "Debug.h"
#include "Device.h"
#include "Task.h"
#include "Platform.h"

namespace iris {
namespace rt {

PolicyDepend::PolicyDepend(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyDepend::~PolicyDepend() {
}

void PolicyDepend::GetDevices(Task* task, Device** devs, int* ndevs) {
  int ntargets = 0;
  int ndepends = task->ndepends();
  unsigned long *depends = task->depends();
  int local_ndepends = ndepends;
  for (int i = 0; i < ndepends; i++) {
    unsigned long dep_uid = depends[i];
    Task *dep = Platform::GetPlatform()->get_task_object(dep_uid);
    if (dep == NULL) {
        local_ndepends--;
        continue;
    }
    Device* dev = dep->dev();
    bool found = false;
    for (int j = 0; j < ntargets; j++) {
      if (devs[j] == dev) {
        found = true;
        break;
      }
    }
    if (!found && IsKernelSupported(task, dev)) {
      devs[ntargets] = dev;
      *ndevs = ++ntargets;
    }
  }
  if (local_ndepends == 0) {
      int selected = 0;
      for(selected=0; selected<ndevs_; selected++) {
          if (IsKernelSupported(task, devs_[selected])) break;
      }
      if (selected == ndevs_) selected = 0;
      devs[0] = devs_[selected];
      *ndevs = 1;
  }
}

} /* namespace rt */
} /* namespace iris */

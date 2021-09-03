#include "PolicyDevice.h"
#include "Debug.h"
#include "Device.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicyDevice::PolicyDevice(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyDevice::~PolicyDevice() {
}

void PolicyDevice::GetDevices(Task* task, Device** devs, int* ndevs) {
  int brs_policy = task->brs_policy();
  int n = 0;
  for (int i = 0; i < ndevs_; i++) {
    Device* dev = devs_[i];
    if ((dev->type() & brs_policy) == dev->type()) {
      devs[n++] = dev;
    }
  }
  *ndevs = n;
}

} /* namespace rt */
} /* namespace brisbane */

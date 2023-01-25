#include "PolicyDefault.h"
#include "Debug.h"
#include "Platform.h"

namespace iris {
namespace rt {

PolicyDefault::PolicyDefault(Scheduler* scheduler) {
  SetScheduler(scheduler);
}

PolicyDefault::~PolicyDefault() {
}

void PolicyDefault::GetDevices(Task* task, Device** devs, int* ndevs) {
  int selected = 0;
  for(selected=0; selected<ndevs_; selected++) {
      if (IsKernelSupported(task, devs_[selected])) break;
  }
  if (selected == ndevs_) selected = 0;
  devs[0] = devs_[selected];
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace iris */

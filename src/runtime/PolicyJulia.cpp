#include "PolicyJulia.h"
#include "Debug.h"
#include "Device.h"
#include "Scheduler.h"
#include "Task.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace iris {
namespace rt {

PolicyJulia::PolicyJulia(Scheduler* scheduler) {
  out_devs_ = NULL;
  j_devs_ = NULL;
  if (!scheduler->platform()->is_julia_enabled()) return;
  SetScheduler(scheduler);
  index_ = 0;
  out_devs_ = new int32_t[ndevs_+1];
  j_devs_ = new iris_device[ndevs_];
  for(int i=0; i<ndevs_; i++) {
    j_devs_[i].class_obj = devs_[i];
    j_devs_[i].uid = i;
  }
}

PolicyJulia::~PolicyJulia() {
  if (out_devs_ != NULL) delete [] out_devs_;
  if (j_devs_ != NULL) delete [] j_devs_;
}

void PolicyJulia::GetDevices(Task* task, Device** devs, int* ndevs) {
  *ndevs = 0;
  if (!scheduler_->platform()->is_julia_enabled()) return;
  julia_policy_t j_policy = iris_get_julia_policy_func();
  const char *jpolicy_name_ = task->julia_policy();
  if (j_policy == NULL) return;
  //printf("Julia Policy name: %s\n", jpolicy_name_);
  iris_task brs_task = *(task->struct_obj());
  *ndevs = j_policy(brs_task, jpolicy_name_, j_devs_, ndevs_, out_devs_);
  //printf("Returned with ndevs: %d\n", *ndevs);
  ASSERT(*ndevs < ndevs_);
  for (int i=0; i<*ndevs; i++) {
    int dev_index = out_devs_[i];
    ASSERT(dev_index <= ndevs_);
    devs[i] = devs_[dev_index-1];
  }
}

} /* namespace rt */
} /* namespace iris */


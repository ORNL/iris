#include "Policy.h"
#include "Command.h"
#include "Task.h"
#include "Debug.h"
#include "Scheduler.h"

namespace iris {
namespace rt {

Policy::Policy() {
}

Policy::~Policy() {
}

bool Policy::IsKernelSupported(Task *task, Device *dev) {
    return task->IsKernelSupported(dev);
}

void Policy::SetScheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
  devs_ = scheduler_->devices();
  ndevs_ = scheduler_->ndevs();
}

} /* namespace rt */
} /* namespace iris */


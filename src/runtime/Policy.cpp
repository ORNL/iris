#include "Policy.h"
#include "Debug.h"
#include "Scheduler.h"

namespace brisbane {
namespace rt {

Policy::Policy() {
}

Policy::~Policy() {
}

void Policy::SetScheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
  devs_ = scheduler_->devices();
  ndevs_ = scheduler_->ndevs();
}

} /* namespace rt */
} /* namespace brisbane */

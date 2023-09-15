#include "TGPolicy.h"
#include "Command.h"
#include "Task.h"
#include "Debug.h"
#include "Scheduler.h"

namespace iris {
namespace rt {

TGPolicy::TGPolicy() {
}

TGPolicy::~TGPolicy() {
}

void TGPolicy::SetScheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
  devs_ = scheduler_->devices();
  ndevs_ = scheduler_->ndevs();
}

} /* namespace rt */
} /* namespace iris */



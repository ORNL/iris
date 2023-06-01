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
    Command *cmd = task->cmd_kernel();
    if (cmd == NULL) return true;
    Kernel *kernel = cmd->kernel();
    if (kernel == NULL) return true;
    if (kernel->isSupported(dev)) return true;
    return false;
}

void Policy::SetScheduler(Scheduler* scheduler) {
  scheduler_ = scheduler;
  devs_ = scheduler_->devices();
  ndevs_ = scheduler_->ndevs();
}

} /* namespace rt */
} /* namespace iris */


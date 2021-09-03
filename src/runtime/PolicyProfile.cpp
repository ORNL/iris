#include "PolicyProfile.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Policies.h"
#include "Kernel.h"
#include "Task.h"

namespace brisbane {
namespace rt {

PolicyProfile::PolicyProfile(Scheduler* scheduler, Policies* policies) {
  SetScheduler(scheduler);
  policies_ = policies;
}

PolicyProfile::~PolicyProfile() {
}

void PolicyProfile::GetDevices(Task* task, Device** devs, int* ndevs) {
  Command* cmd = task->cmd_kernel();
  if (!cmd) return policies_->GetPolicy(brisbane_default, NULL)->GetDevices(task, devs, ndevs);
  History* history = cmd->kernel()->history();
  devs[0] = history->OptimalDevice(task);
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace brisbane */

#include "PolicyProfile.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Policies.h"
#include "Kernel.h"
#include "Task.h"
#include <memory>
using namespace std;
namespace iris {
namespace rt {

PolicyProfile::PolicyProfile(Scheduler* scheduler, Policies* policies) {
  SetScheduler(scheduler);
  policies_ = policies;
}

PolicyProfile::~PolicyProfile() {
}

void PolicyProfile::GetDevices(Task* task, Device** devs, int* ndevs) {
  Command* cmd = task->cmd_kernel();
  if (!cmd) return policies_->GetPolicy(iris_default, NULL)->GetDevices(task, devs, ndevs);
  shared_ptr<History> history = cmd->kernel()->history();
  devs[0] = history->OptimalDevice(task);
  *ndevs = 1;
}

} /* namespace rt */
} /* namespace iris */

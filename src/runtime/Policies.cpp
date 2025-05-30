#include "Policies.h"
#include "Debug.h"
#include "Platform.h"
#include "Scheduler.h"
#include "LoaderPolicy.h"
#include "PolicyBlockCycle.h"
#include "PolicyFirstToFinish.h"
#include "PolicyData.h"
#include "PolicyDefault.h"
#include "PolicyDepend.h"
#include "PolicyDevice.h"
#include "PolicyJulia.h"
#include "PolicyShortestDeviceQueue.h"
#include "PolicyProfile.h"
#include "PolicyRandom.h"
#include "PolicyRoundRobin.h"
#include "Platform.h"

namespace iris {
namespace rt {

Policies::Policies(Scheduler* scheduler) {
  scheduler_ = scheduler;
  policy_block_cycle_ = new PolicyBlockCycle(scheduler_);
  policy_ftf_         = new PolicyFirstToFinish(scheduler_);
  policy_data_        = new PolicyData(scheduler_);
  policy_default_     = new PolicyDefault(scheduler_);
  policy_depend_      = new PolicyDepend(scheduler_);
  policy_device_      = new PolicyDevice(scheduler_);
  policy_sdq_         = new PolicyShortestDeviceQueue(scheduler_);
  policy_profile_     = new PolicyProfile(scheduler_, this);
  policy_random_      = new PolicyRandom(scheduler_);
  policy_roundrobin_  = new PolicyRoundRobin(scheduler_);
  policy_julia_       = new PolicyJulia(scheduler_);
}

Policies::~Policies() {
  delete policy_block_cycle_;
  delete policy_ftf_;
  delete policy_data_;
  delete policy_default_;
  delete policy_depend_;
  delete policy_device_;
  delete policy_sdq_;
  delete policy_profile_;
  delete policy_random_;
  delete policy_roundrobin_;
  delete policy_julia_;
  for (std::map<std::string, LoaderPolicy*>::iterator I = policy_customs_.begin(), E = policy_customs_.end(); I != E; ++I)
    delete I->second;
}

Policy* Policies::GetPolicy(int brs_policy, const char* opt) {
  if (brs_policy &  iris_roundrobin) return policy_roundrobin_;
  if (brs_policy &  iris_julia_policy) return policy_julia_;
  if (brs_policy &  iris_cpu    ||
      brs_policy &  iris_nvidia ||
      brs_policy &  iris_amd    ||
      brs_policy &  iris_gpu    ||
      brs_policy &  iris_phi    ||
      brs_policy &  iris_hexagon||
      brs_policy &  iris_dsp    ||
      brs_policy &  iris_fpga)          return policy_device_;
  if (brs_policy == iris_block_cycle)   return policy_block_cycle_;
  if (brs_policy == iris_ftf)           return policy_ftf_;
  if (brs_policy == iris_data)          return policy_data_;
  if (brs_policy == iris_depend)        return policy_depend_;
  if (brs_policy == iris_default)       return policy_default_;
  if (brs_policy == iris_sdq)           return policy_sdq_;
  if (brs_policy == iris_profile)       return policy_profile_;
  if (brs_policy == iris_random)        return policy_random_;
  if (brs_policy == iris_pending)       return policy_data_;//to get to this point with the pending policy it can only be for D2H tranfers (so just default to data policy to minimize memory movement)
  if (brs_policy == iris_custom) {
    if (policy_customs_.find(std::string(opt)) != policy_customs_.end()) {
      Policy* policy = policy_customs_[opt]->policy();
      policy->SetScheduler(scheduler_);
      return policy;
    }
  }
  _warning("unknown policy [%d] [0x%x] [%s]", brs_policy, brs_policy, opt);
  return policy_sdq_;
}

int Policies::Register(const char* lib, const char* name, void* params) {
  LoaderPolicy* loader = new LoaderPolicy(lib, name);
  std::string namestr = std::string(name);
  if (policy_customs_.find(namestr) != policy_customs_.end()) {
    scheduler_->platform()->IncrementErrorCount();
    _error("existing policy name[%s]", name);
    return IRIS_ERROR;
  }
  if (loader->Load() != IRIS_SUCCESS) {
    scheduler_->platform()->IncrementErrorCount();
    _error("cannot load custom policy[%s]", name);
    return IRIS_ERROR;
  }
  loader->Init(params);
  _trace("lib[%s] name[%s]", lib, name);
  policy_customs_.insert(std::pair<std::string, LoaderPolicy*>(namestr, loader));
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


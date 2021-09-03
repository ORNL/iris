#ifndef BRISBANE_SRC_RT_POLICY_PROFILE_H
#define BRISBANE_SRC_RT_POLICY_PROFILE_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class Policies;

class PolicyProfile : public Policy {
public:
  PolicyProfile(Scheduler* scheduler, Policies* policies);
  virtual ~PolicyProfile();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

private:
  Policies* policies_;

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_PROFILE_H */

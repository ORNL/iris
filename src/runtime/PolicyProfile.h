#ifndef IRIS_SRC_RT_POLICY_PROFILE_H
#define IRIS_SRC_RT_POLICY_PROFILE_H

#include "Policy.h"

namespace iris {
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
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_PROFILE_H */

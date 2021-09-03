#ifndef BRISBANE_SRC_RT_POLICY_ANY_H
#define BRISBANE_SRC_RT_POLICY_ANY_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyAny : public Policy {
public:
  PolicyAny(Scheduler* scheduler);
  virtual ~PolicyAny();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_ANY_H */


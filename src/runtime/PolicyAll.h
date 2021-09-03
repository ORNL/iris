#ifndef BRISBANE_SRC_RT_POLICY_ALL_H
#define BRISBANE_SRC_RT_POLICY_ALL_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyAll : public Policy {
public:
  PolicyAll(Scheduler* scheduler);
  virtual ~PolicyAll();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_ALL_H */


#ifndef BRISBANE_SRC_RT_POLICY_DEPEND_H
#define BRISBANE_SRC_RT_POLICY_DEPEND_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyDepend: public Policy {
public:
  PolicyDepend(Scheduler* scheduler);
  virtual ~PolicyDepend();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_DEPEND_H */


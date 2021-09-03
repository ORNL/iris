#ifndef BRISBANE_SRC_RT_POLICY_DEFAULT_H
#define BRISBANE_SRC_RT_POLICY_DEFAULT_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyDefault : public Policy {
public:
  PolicyDefault(Scheduler* scheduler);
  virtual ~PolicyDefault();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_DEFAULT_H */

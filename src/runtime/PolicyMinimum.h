#ifndef IRIS_SRC_RT_POLICY_ANY_H
#define IRIS_SRC_RT_POLICY_ANY_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyMinimum : public Policy {
public:
  PolicyMinimum(Scheduler* scheduler);
  virtual ~PolicyMinimum();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_ANY_H */


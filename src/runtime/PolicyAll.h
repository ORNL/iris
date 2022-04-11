#ifndef IRIS_SRC_RT_POLICY_ALL_H
#define IRIS_SRC_RT_POLICY_ALL_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyAll : public Policy {
public:
  PolicyAll(Scheduler* scheduler);
  virtual ~PolicyAll();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_ALL_H */


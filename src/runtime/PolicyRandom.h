#ifndef IRIS_SRC_RT_POLICY_RANDOM_H
#define IRIS_SRC_RT_POLICY_RANDOM_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyRandom : public Policy {
public:
  PolicyRandom(Scheduler* scheduler);
  virtual ~PolicyRandom();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_RANDOM_H */

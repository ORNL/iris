#ifndef IRIS_SRC_RT_POLICY_FIRST_TO_FINISH_H
#define IRIS_SRC_RT_POLICY_FIRST_TO_FINISH_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyFirstToFinish : public Policy {
public:
  PolicyFirstToFinish(Scheduler* scheduler);
  virtual ~PolicyFirstToFinish();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_FIRST_TO_FINISH_H */


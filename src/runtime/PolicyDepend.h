#ifndef IRIS_SRC_RT_POLICY_DEPEND_H
#define IRIS_SRC_RT_POLICY_DEPEND_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyDepend: public Policy {
public:
  PolicyDepend(Scheduler* scheduler);
  virtual ~PolicyDepend();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_DEPEND_H */


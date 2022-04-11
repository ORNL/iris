#ifndef IRIS_SRC_RT_POLICY_DEFAULT_H
#define IRIS_SRC_RT_POLICY_DEFAULT_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyDefault : public Policy {
public:
  PolicyDefault(Scheduler* scheduler);
  virtual ~PolicyDefault();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_DEFAULT_H */

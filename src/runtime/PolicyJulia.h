#ifndef IRIS_SRC_RT_POLICY_JULIA_H
#define IRIS_SRC_RT_POLICY_JULIA_H

#include "Policy.h"
#include "iris/iris.h"

namespace iris {
namespace rt {

class PolicyJulia: public Policy {
public:
  PolicyJulia(Scheduler* scheduler);
  virtual ~PolicyJulia();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

private:
  int index_;
  int32_t *out_devs_;
  iris_device *j_devs_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_JULIA_H */


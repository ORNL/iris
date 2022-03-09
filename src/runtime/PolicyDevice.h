#ifndef IRIS_SRC_RT_POLICY_DEVICE_H
#define IRIS_SRC_RT_POLICY_DEVICE_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyDevice : public Policy {
public:
  PolicyDevice(Scheduler* scheduler);
  virtual ~PolicyDevice();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_DEVICE_H */

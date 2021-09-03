#ifndef BRISBANE_SRC_RT_POLICY_DEVICE_H
#define BRISBANE_SRC_RT_POLICY_DEVICE_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyDevice : public Policy {
public:
  PolicyDevice(Scheduler* scheduler);
  virtual ~PolicyDevice();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_DEVICE_H */

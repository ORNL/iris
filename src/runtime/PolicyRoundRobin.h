#ifndef BRISBANE_SRC_RT_POLICY_ROUND_ROBIN_H
#define BRISBANE_SRC_RT_POLICY_ROUND_ROBIN_H

#include "Policy.h"

namespace brisbane {
namespace rt {

class PolicyRoundRobin: public Policy {
public:
  PolicyRoundRobin(Scheduler* scheduler);
  virtual ~PolicyRoundRobin();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

private:
  void GetDevice(Task* task, Device** devs, int* ndevs);
  void GetDeviceType(Task* task, Device** devs, int* ndevs);

private:
  int index_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POLICY_ROUND_ROBIN_H */


#ifndef IRIS_SRC_RT_POLICY_ROUND_ROBIN_H
#define IRIS_SRC_RT_POLICY_ROUND_ROBIN_H

#include "Policy.h"

namespace iris {
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
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_ROUND_ROBIN_H */


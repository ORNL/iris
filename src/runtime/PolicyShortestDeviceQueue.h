#ifndef IRIS_SRC_RT_POLICY_SHORTEST_DEVICE_QUEUE_H
#define IRIS_SRC_RT_POLICY_SHORTEST_DEVICE_QUEUE_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyShortestDeviceQueue : public Policy {
public:
  PolicyShortestDeviceQueue(Scheduler* scheduler);
  virtual ~PolicyShortestDeviceQueue();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_SHORTEST_DEVICE_QUEUE_H */


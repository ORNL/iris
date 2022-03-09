#ifndef IRIS_SRC_RT_POLICY_DATA_H
#define IRIS_SRC_RT_POLICY_DATA_H

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyData : public Policy {
public:
  PolicyData(Scheduler* scheduler);
  virtual ~PolicyData();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_DATA_H */

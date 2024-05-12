#pragma once

#include "Policy.h"

namespace iris {
namespace rt {

class PolicyBlockCycle : public Policy {
public:
  PolicyBlockCycle(Scheduler* scheduler);
  virtual ~PolicyBlockCycle();

  virtual void GetDevices(Task* task, Device** devs, int* ndevs);
};

} /* namespace rt */
} /* namespace iris */




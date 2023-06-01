#include <iris/rt/Policy.h>
#include <iris/rt/Debug.h>
#include <iris/rt/Device.h>
#include <iris/rt/Scheduler.h>
#include <iris/rt/Task.h>

namespace iris {
namespace rt {

class PolicyFirst: public Policy {
public:
  PolicyFirst() {}
  virtual ~PolicyFirst() {}
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    _info("ndevs[%d]", ndevs_);
    devs[0] = devs_[0];
    *ndevs = 1;
  }
};

class PolicyLast : public Policy {
public:
  PolicyLast() {}
  virtual ~PolicyLast() {}
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    _info("ndevs[%d]", ndevs_);
    devs[0] = devs_[ndevs_ - 1];
    *ndevs = 1;
  }
};

} /* namespace rt */
} /* namespace iris */

REGISTER_CUSTOM_POLICY(PolicyLast, policy_last)
REGISTER_CUSTOM_POLICY(PolicyFirst, policy_first)


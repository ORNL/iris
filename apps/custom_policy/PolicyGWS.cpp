#include <iris/iris.h>
#include <iris/rt/Policy.h>
#include <iris/rt/Command.h>
#include <iris/rt/Device.h>
#include <iris/rt/Task.h>

namespace iris {
namespace rt {

class PolicyGWS: public Policy {
public:
  PolicyGWS() {}
  virtual ~PolicyGWS() {}
  virtual void Init(void* params) {
    threshold_ = (size_t) params;
  }
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    Command* cmd = task->cmd_kernel();
    size_t* gws = cmd->gws();
    size_t total_work_items = gws[0] * gws[1] * gws[2];
    int target_dev = total_work_items > threshold_ ? iris_gpu : iris_cpu;
    int devid = 0;
    for (int i = 0; i < ndevices(); i++)
      if (device(i)->type() & target_dev) devs[devid++] = device(i);
    *ndevs = devid;
  }

  size_t threshold_;
};

} /* namespace runtime */
} /* namespace iris */

REGISTER_CUSTOM_POLICY(PolicyGWS, custom_gws)


#include <iris/iris.h>
#include <iris/rt/Policy.h>
#include <iris/rt/Debug.h>
#include <iris/rt/Device.h>
#include <iris/rt/Command.h>
#include <iris/rt/Device.h>
#include <iris/rt/Task.h>

namespace iris {
namespace rt {

int task_hook_pre(void* task) {
  Task* t = (Task*) task;
  printf("[%s:%d] enter task[%lu]\n", __FILE__, __LINE__, t->uid());
  return 0;
}

int task_hook_post(void* task) {
  Task* t = (Task*) task;
  printf("[%s:%d] exit task[%lu] time[%lf]\n", __FILE__, __LINE__, t->uid(), t->time());
  return 0;
}

int cmd_hook_pre(void* cmd) {
  Command* c = (Command*) cmd;
  printf("[%s:%d] enter cmd[%x]\n", __FILE__, __LINE__, c->type());
  return 0;
}

int cmd_hook_post(void* cmd) {
  Command* c = (Command*) cmd;
  if (c->type() == IRIS_CMD_KERNEL)
    printf("[%s:%d] exit cmd[%x] policy[%s] kernel[%s] gws[%lu][%lu][%lu] dev[%d][%s] time[%lf]\n", __FILE__, __LINE__, c->type(), c->task()->opt(), c->kernel()->name(), c->gws(0), c->gws(1), c->gws(2), c->task()->dev()->devno(), c->task()->dev()->name(), c->time());
  else printf("[%s:%d] exit cmd[%x] time[%lf]\n", __FILE__, __LINE__, c->type(), c->time());
  return 0;
}

class PolicyGWSHook: public Policy {
public:
  PolicyGWSHook() {
    iris_register_hooks_task(task_hook_pre, task_hook_post);
    iris_register_hooks_command(cmd_hook_pre, cmd_hook_post);
  }
  virtual ~PolicyGWSHook() {}
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

} /* namespace rt */
} /* namespace iris */

REGISTER_CUSTOM_POLICY(PolicyGWSHook, policy_gws_hook)


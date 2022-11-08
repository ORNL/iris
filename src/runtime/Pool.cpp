#include "Pool.h"
#include "Debug.h"
#include "Command.h"
#include "Task.h"

namespace iris {
namespace rt {

Pool::Pool(Platform* platform) {
  platform_ = platform;
#if IRIS_POOL_ENABLED
  for (int i = 0; i < IRIS_POOL_MAX_TASK; i++) {
    tasks_[i] = new Task(platform, IRIS_TASK, NULL);
  }
  for (int i = 0; i < IRIS_POOL_MAX_CMD; i++) {
    cmds_[i] = new Command();
  }
  tid_ = 0;
  cid_ = 0;
#endif
}

Pool::~Pool() {
}

Task* Pool::GetTask() {
#if IRIS_POOL_ENABLED
  return tasks_[tid_++];
#else
  const char *pool_tn = "Pool";
  return new Task(platform_, IRIS_TASK, pool_tn);
#endif
}

Command* Pool::GetCommand(Task* task, int type) {
#if IRIS_POOL_ENABLED
  Command* cmd = cmds_[cid_++];
  cmd->Set(task, type);
  return cmd;
#else
  return new Command(task, type);
#endif
}

} /* namespace rt */
} /* namespace iris */


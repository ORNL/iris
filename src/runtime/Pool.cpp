#include "Pool.h"
#include "Debug.h"
#include "Command.h"
#include "Task.h"

namespace brisbane {
namespace rt {

Pool::Pool(Platform* platform) {
  platform_ = platform;
#if BRISBANE_POOL_ENABLED
  for (int i = 0; i < BRISBANE_POOL_MAX_TASK; i++) {
    tasks_[i] = new Task(platform, BRISBANE_TASK, NULL);
  }
  for (int i = 0; i < BRISBANE_POOL_MAX_CMD; i++) {
    cmds_[i] = new Command();
  }
  tid_ = 0;
  cid_ = 0;
#endif
}

Pool::~Pool() {
}

Task* Pool::GetTask() {
#if BRISBANE_POOL_ENABLED
  return tasks_[tid_++];
#else
  return new Task(platform_, BRISBANE_TASK, NULL);
#endif
}

Command* Pool::GetCommand(Task* task, int type) {
#if BRISBANE_POOL_ENABLED
  Command* cmd = cmds_[cid_++];
  cmd->Set(task, type);
  return cmd;
#else
  return new Command(task, type);
#endif
}

} /* namespace rt */
} /* namespace brisbane */


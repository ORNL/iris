#ifndef BRISBANE_SRC_RT_POOL_H
#define BRISBANE_SRC_RT_POOL_H

#define BRISBANE_POOL_ENABLED     0
#define BRISBANE_POOL_MAX_TASK    1100
#define BRISBANE_POOL_MAX_CMD     1100

namespace brisbane {
namespace rt {

class Command;
class Platform;
class Task;

class Pool {
public:
  Pool(Platform* platform);
  ~Pool();

  Task* GetTask();
  Command* GetCommand(Task* task, int type);

private:
  Platform* platform_;
  Task* tasks_[BRISBANE_POOL_MAX_TASK];
  Command* cmds_[BRISBANE_POOL_MAX_CMD];

  int tid_;
  int cid_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_POOL_H */


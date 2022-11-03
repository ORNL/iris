#ifndef IRIS_SRC_RT_POOL_H
#define IRIS_SRC_RT_POOL_H

#define IRIS_POOL_ENABLED     0
#define IRIS_POOL_MAX_TASK    1100
#define IRIS_POOL_MAX_CMD     1100

namespace iris {
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
#if IRIS_POOL_ENABLED
  Task* tasks_[IRIS_POOL_MAX_TASK];
  Command* cmds_[IRIS_POOL_MAX_CMD];
  int tid_;
  int cid_;
#endif

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POOL_H */


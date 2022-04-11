#ifndef IRIS_SRC_RT_HUB_H
#define IRIS_SRC_RT_HUB_H

#define IRIS_HUB_MQ_PATH        "/tmp/iris_hub.mq"
#define IRIS_HUB_MQ_PID         52
#define IRIS_HUB_FIFO_PATH      "/tmp/iris_hub.fifo"
#define IRIS_HUB_PERM           0644

#define IRIS_HUB_MQ_MSG_SIZE    256
#define IRIS_HUB_MQ_STOP        0x1000
#define IRIS_HUB_MQ_STATUS      0x1001
#define IRIS_HUB_MQ_REGISTER    0x1002
#define IRIS_HUB_MQ_DEREGISTER  0x1003
#define IRIS_HUB_MQ_TASK_INC    0x1004
#define IRIS_HUB_MQ_TASK_ALL    0x1005

#define IRIS_HUB_FIFO_MSG_SIZE  256
#define IRIS_HUB_FIFO_STOP      0x2000
#define IRIS_HUB_FIFO_STATUS    0x2001
#define IRIS_HUB_FIFO_TASK_ALL  0x2005

#include "Config.h"
#include <sys/msg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>

namespace iris {
namespace rt {

class Message;

class Hub {
public:
  Hub();
  ~Hub();

  int Run();
  bool Running();
  int CloseMQ();

private:
  int OpenMQ();

  int SendFIFO(Message& msg, int pid);

  int ExecuteStop(Message& msg, int pid);
  int ExecuteStatus(Message& msg, int pid);
  int ExecuteRegister(Message& msg, int pid);
  int ExecuteDeregister(Message& msg, int pid);
  int ExecuteTaskInc(Message& msg, int pid);
  int ExecuteTaskAll(Message& msg, int pid);

private:
  key_t key_;
  int mq_;
  bool running_;
  std::map<int, int> fifos_;
  size_t ntasks_[IRIS_MAX_NDEVS];
  int ndevs_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_HUB_H */

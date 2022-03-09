#include <iris/iris.h>
#include "Hub.h"
#include "HubClient.h"
#include "Debug.h"
#include "Message.h"
#include <stdlib.h>

namespace iris {
namespace rt {

Hub::Hub() {
  running_ = false;
  mq_ = -1;
  ndevs_ = 0;
  for (int i = 0; i < IRIS_MAX_NDEVS; i++) ntasks_[i] = 0;
}

Hub::~Hub() {
  if (mq_ != -1) CloseMQ();
}

int Hub::OpenMQ() {
#if !USE_HUB
  return IRIS_ERROR;
#else
  char cmd[64];
  memset(cmd, 0, 64);
  sprintf(cmd, "touch %s", IRIS_HUB_MQ_PATH);
  if (system(cmd) == -1) perror(cmd);
  if ((key_ = ftok(IRIS_HUB_MQ_PATH, IRIS_HUB_MQ_PID)) == -1) {
    perror("ftok");
    return IRIS_ERROR;
  }
  if ((mq_ = msgget(key_, IRIS_HUB_PERM | IPC_CREAT)) == -1) {
    perror("msgget");
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
#endif
}

int Hub::CloseMQ() {
  char cmd[64];
  memset(cmd, 0, 64);
  sprintf(cmd, "rm -f %s %s*", IRIS_HUB_MQ_PATH, IRIS_HUB_FIFO_PATH);
  if (system(cmd) == -1) perror(cmd);
  return IRIS_SUCCESS;
}

int Hub::SendFIFO(Message& msg, int pid) {
  int fd = fifos_[pid];
  ssize_t ssret = write(fd, msg.buf(), IRIS_HUB_FIFO_MSG_SIZE);
  if (ssret != IRIS_HUB_FIFO_MSG_SIZE) {
    _error("ssret[%zd]", ssret);
    perror("write");
  }
  return IRIS_SUCCESS;
}

int Hub::Run() {
#if !USE_HUB
  return IRIS_ERROR;
#else
  running_ = OpenMQ() == IRIS_SUCCESS;
  while (running_) {
    Message msg;
    if (msgrcv(mq_, msg.buf(), IRIS_HUB_MQ_MSG_SIZE, 0, 0) == -1) {
      perror("msgrcv");
      continue;
    }
    int header = msg.ReadHeader();
    int pid = msg.ReadPID();
    int ret = IRIS_SUCCESS;
    switch (header) {
      case IRIS_HUB_MQ_STOP:          ret = ExecuteStop(msg, pid);        break;
      case IRIS_HUB_MQ_STATUS:        ret = ExecuteStatus(msg, pid);      break;
      case IRIS_HUB_MQ_REGISTER:      ret = ExecuteRegister(msg, pid);    break;
      case IRIS_HUB_MQ_DEREGISTER:    ret = ExecuteDeregister(msg, pid);  break;
      case IRIS_HUB_MQ_TASK_INC:      ret = ExecuteTaskInc(msg, pid);     break;
      case IRIS_HUB_MQ_TASK_ALL:      ret = ExecuteTaskAll(msg, pid);     break;
      default: _error("not supported msg header[0x%x]", header);
    }
    if (ret != IRIS_SUCCESS) _error("header[0x%x] ret[%d]", header, ret);
  }
  return IRIS_SUCCESS;
#endif
}

bool Hub::Running() {
  return access(IRIS_HUB_MQ_PATH, F_OK) != -1;
}

int Hub::ExecuteStop(Message& msg, int pid) {
  running_ = false;
  Message fmsg(IRIS_HUB_FIFO_STOP);
  SendFIFO(fmsg, pid);
  return IRIS_SUCCESS;
}

int Hub::ExecuteStatus(Message& msg, int pid) {
  Message fmsg(IRIS_HUB_FIFO_STATUS);
  fmsg.WriteInt(ndevs_);
  for (int i = 0; i < ndevs_; i++) {
    fmsg.WriteULong(ntasks_[i]);
  }
  SendFIFO(fmsg, pid);
  return IRIS_SUCCESS;
}

int Hub::ExecuteRegister(Message& msg, int pid) {
  int ndevs = msg.ReadInt();
  if (ndevs != -1) {
    if (ndevs_ != 0 && ndevs_ != ndevs) _error("ndevs_[%d] ndev[%d]", ndevs_, ndevs);
    ndevs_ = ndevs;
  }
  char path[64];
  sprintf(path, "%s.%d", IRIS_HUB_FIFO_PATH, pid);
  int fd = open(path, O_RDWR);
  fifos_[pid] = fd;
  return IRIS_SUCCESS;
}

int Hub::ExecuteDeregister(Message& msg, int pid) {
  char path[64];
  sprintf(path, "%s.%d", IRIS_HUB_FIFO_PATH, pid);
  int fifo = fifos_[pid];
  int iret = close(fifo);
  if (iret == -1) {
    _error("iret[%d]", iret);
    perror("close");
  }
  iret = remove(path);
  if (iret == -1) {
    _error("iret[%d][%s]", iret, path);
    perror("remove");
  }
  fifos_.erase(pid);
  return IRIS_SUCCESS;
}

int Hub::ExecuteTaskInc(Message& msg, int pid) {
  int dev = msg.ReadInt();
  int i = msg.ReadInt();
  ntasks_[dev] += i;
  return IRIS_SUCCESS;
}

int Hub::ExecuteTaskAll(Message& msg, int pid) {
  int ndevs = msg.ReadInt();
  Message fmsg(IRIS_HUB_FIFO_TASK_ALL);
  fmsg.Write(ntasks_, ndevs * sizeof(size_t));
  SendFIFO(fmsg, pid);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


#include <brisbane/brisbane.h>
#include "Hub.h"
#include "HubClient.h"
#include "Debug.h"
#include "Message.h"
#include <stdlib.h>

namespace brisbane {
namespace rt {

Hub::Hub() {
  running_ = false;
  mq_ = -1;
  ndevs_ = 0;
  for (int i = 0; i < BRISBANE_MAX_NDEVS; i++) ntasks_[i] = 0;
}

Hub::~Hub() {
  if (mq_ != -1) CloseMQ();
}

int Hub::OpenMQ() {
#if !USE_HUB
  return BRISBANE_ERR;
#else
  char cmd[64];
  memset(cmd, 0, 64);
  sprintf(cmd, "touch %s", BRISBANE_HUB_MQ_PATH);
  if (system(cmd) == -1) perror(cmd);
  if ((key_ = ftok(BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_MQ_PID)) == -1) {
    perror("ftok");
    return BRISBANE_ERR;
  }
  if ((mq_ = msgget(key_, BRISBANE_HUB_PERM | IPC_CREAT)) == -1) {
    perror("msgget");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
#endif
}

int Hub::CloseMQ() {
  char cmd[64];
  memset(cmd, 0, 64);
  sprintf(cmd, "rm -f %s %s*", BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_FIFO_PATH);
  if (system(cmd) == -1) perror(cmd);
  return BRISBANE_OK;
}

int Hub::SendFIFO(Message& msg, int pid) {
  int fd = fifos_[pid];
  ssize_t ssret = write(fd, msg.buf(), BRISBANE_HUB_FIFO_MSG_SIZE);
  if (ssret != BRISBANE_HUB_FIFO_MSG_SIZE) {
    _error("ssret[%zd]", ssret);
    perror("write");
  }
  return BRISBANE_OK;
}

int Hub::Run() {
#if !USE_HUB
  return BRISBANE_ERR;
#else
  running_ = OpenMQ() == BRISBANE_OK;
  while (running_) {
    Message msg;
    if (msgrcv(mq_, msg.buf(), BRISBANE_HUB_MQ_MSG_SIZE, 0, 0) == -1) {
      perror("msgrcv");
      continue;
    }
    int header = msg.ReadHeader();
    int pid = msg.ReadPID();
    int ret = BRISBANE_OK;
    switch (header) {
      case BRISBANE_HUB_MQ_STOP:          ret = ExecuteStop(msg, pid);        break;
      case BRISBANE_HUB_MQ_STATUS:        ret = ExecuteStatus(msg, pid);      break;
      case BRISBANE_HUB_MQ_REGISTER:      ret = ExecuteRegister(msg, pid);    break;
      case BRISBANE_HUB_MQ_DEREGISTER:    ret = ExecuteDeregister(msg, pid);  break;
      case BRISBANE_HUB_MQ_TASK_INC:      ret = ExecuteTaskInc(msg, pid);     break;
      case BRISBANE_HUB_MQ_TASK_ALL:      ret = ExecuteTaskAll(msg, pid);     break;
      default: _error("not supported msg header[0x%x]", header);
    }
    if (ret != BRISBANE_OK) _error("header[0x%x] ret[%d]", header, ret);
  }
  return BRISBANE_OK;
#endif
}

bool Hub::Running() {
  return access(BRISBANE_HUB_MQ_PATH, F_OK) != -1;
}

int Hub::ExecuteStop(Message& msg, int pid) {
  running_ = false;
  Message fmsg(BRISBANE_HUB_FIFO_STOP);
  SendFIFO(fmsg, pid);
  return BRISBANE_OK;
}

int Hub::ExecuteStatus(Message& msg, int pid) {
  Message fmsg(BRISBANE_HUB_FIFO_STATUS);
  fmsg.WriteInt(ndevs_);
  for (int i = 0; i < ndevs_; i++) {
    fmsg.WriteULong(ntasks_[i]);
  }
  SendFIFO(fmsg, pid);
  return BRISBANE_OK;
}

int Hub::ExecuteRegister(Message& msg, int pid) {
  int ndevs = msg.ReadInt();
  if (ndevs != -1) {
    if (ndevs_ != 0 && ndevs_ != ndevs) _error("ndevs_[%d] ndev[%d]", ndevs_, ndevs);
    ndevs_ = ndevs;
  }
  char path[64];
  sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid);
  int fd = open(path, O_RDWR);
  fifos_[pid] = fd;
  return BRISBANE_OK;
}

int Hub::ExecuteDeregister(Message& msg, int pid) {
  char path[64];
  sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid);
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
  return BRISBANE_OK;
}

int Hub::ExecuteTaskInc(Message& msg, int pid) {
  int dev = msg.ReadInt();
  int i = msg.ReadInt();
  ntasks_[dev] += i;
  return BRISBANE_OK;
}

int Hub::ExecuteTaskAll(Message& msg, int pid) {
  int ndevs = msg.ReadInt();
  Message fmsg(BRISBANE_HUB_FIFO_TASK_ALL);
  fmsg.Write(ntasks_, ndevs * sizeof(size_t));
  SendFIFO(fmsg, pid);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


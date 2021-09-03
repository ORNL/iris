#include <brisbane/brisbane.h>
#include "HubClient.h"
#include "Hub.h"
#include "Debug.h"
#include "Message.h"
#include "Scheduler.h"
#include <stdlib.h>
#include <unistd.h>

namespace brisbane {
namespace rt {

HubClient::HubClient(Scheduler* scheduler) {
  ndevs_ = -1;
  if (scheduler) {
    scheduler_  = scheduler;
    ndevs_ = scheduler->ndevs();
  }
  pid_ = getpid();
  fifo_ = -1;
  available_ = false;
  stop_hub_ = false;
}

HubClient::~HubClient() {
  if (!stop_hub_) CloseMQ();
}

int HubClient::Init() {
  int ret = OpenMQ();
  if (ret == BRISBANE_OK) ret = OpenFIFO();
  if (ret == BRISBANE_OK) Register();
  available_ = ret == BRISBANE_OK;
  return ret;
}

int HubClient::StopHub() {
  if (!available_) return BRISBANE_ERR;
  Message msg(BRISBANE_HUB_MQ_STOP);
  msg.WritePID(pid_);
  SendMQ(msg);
  msg.Clear();
  RecvFIFO(msg);
  stop_hub_ = true;
  return BRISBANE_OK;
}

int HubClient::Status() {
  if (!available_) return BRISBANE_ERR;
  Message msg(BRISBANE_HUB_MQ_STATUS);
  msg.WritePID(pid_);
  SendMQ(msg);
  msg.Clear();
  RecvFIFO(msg);
  int header = msg.ReadHeader();
  int ndevs = msg.ReadInt();
  for (int i = 0; i < ndevs; i++) {
    _info("Device[%d] ntasks[%lu]", i, msg.ReadULong());
  }
  return BRISBANE_OK;
}

int HubClient::OpenMQ() {
#if !USE_HUB
  return BRISBANE_ERR;
#else
  key_t key;
  if ((key = ftok(BRISBANE_HUB_MQ_PATH, BRISBANE_HUB_MQ_PID)) == -1) return BRISBANE_ERR;
  if ((mq_ = msgget(key, BRISBANE_HUB_PERM | IPC_CREAT)) == -1) return BRISBANE_ERR;
  return BRISBANE_OK;
#endif
}

int HubClient::CloseMQ() {
  if (fifo_ != -1) {
    CloseFIFO();
    Deregister();
  }
  return BRISBANE_OK;
}

int HubClient::SendMQ(Message& msg) {
#if !USE_HUB
  return BRISBANE_ERR;
#else
  int iret = msgsnd(mq_, msg.buf(), BRISBANE_HUB_MQ_MSG_SIZE, 0);
  if (iret == -1) {
    _error("msgsnd err[%d]", iret);
    perror("msgsnd");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
#endif
}

int HubClient::OpenFIFO() {
  char path[64];
  sprintf(path, "%s.%d", BRISBANE_HUB_FIFO_PATH, pid_);
  int iret = mknod(path, S_IFIFO | BRISBANE_HUB_PERM, 0);
  if (iret == -1) {
    _error("iret[%d]", iret);
    perror("mknod");
    return BRISBANE_ERR;
  }
  fifo_ = open(path, O_RDWR);
  if (fifo_ == -1) {
    _error("path[%s]", path);
    perror("read");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int HubClient::CloseFIFO() {
  int iret = close(fifo_);
  if (iret == -1) {
    _error("iret[%d]", iret);
    perror("close");
  }
  return BRISBANE_OK;
}

int HubClient::RecvFIFO(Message& msg) {
  ssize_t ssret = read(fifo_, msg.buf(), BRISBANE_HUB_FIFO_MSG_SIZE);
  if (ssret != BRISBANE_HUB_FIFO_MSG_SIZE) {
    _error("ssret[%zd]", ssret);
    perror("read");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int HubClient::Register() {
  Message msg(BRISBANE_HUB_MQ_REGISTER);
  msg.WritePID(pid_);
  msg.WriteInt(ndevs_);
  SendMQ(msg);
  return BRISBANE_OK;
}

int HubClient::Deregister() {
  Message msg(BRISBANE_HUB_MQ_DEREGISTER);
  msg.WritePID(pid_);
  SendMQ(msg);
  return BRISBANE_OK;
}

int HubClient::TaskInc(int dev, int i) {
  if (!available_) return BRISBANE_OK;
  Message msg(BRISBANE_HUB_MQ_TASK_INC);
  msg.WritePID(pid_);
  msg.WriteInt(dev);
  msg.WriteInt(i);
  SendMQ(msg);
  return BRISBANE_OK;
}

int HubClient::TaskDec(int dev, int i) {
  return TaskInc(dev, -i);
}

int HubClient::TaskAll(size_t* ntasks, int ndevs) {
  Message msg(BRISBANE_HUB_MQ_TASK_ALL);
  msg.WritePID(pid_);
  msg.WriteInt(ndevs);
  SendMQ(msg);

  msg.Clear();
  RecvFIFO(msg);
  int header = msg.ReadHeader();
  if (header != BRISBANE_HUB_FIFO_TASK_ALL) {
    _error("header[0x%x]", header);
  }
  for (int i = 0; i < ndevs; i++) {
    ntasks[i] = msg.ReadULong();
    _trace("dev[%d] ntasks[%lu]", i, ntasks[i]);
  }
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

#include "Profiler.h"
#include "Debug.h"
#include "Device.h"
#include "Message.h"
#include "Platform.h"
#include "Task.h"
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace brisbane {
namespace rt {

Profiler::Profiler(Platform* platform) {
  platform_ = platform;
  fd_ = -1;
  msg_ = new Message();
}

Profiler::~Profiler() {
  CloseFD();
  if (msg_) delete msg_;
}

int Profiler::OpenFD() {
  time_t t = time(NULL);
  char s[64];
  strftime(s, 64, "%Y%m%d-%H%M%S", localtime(&t));
  sprintf(path_, "%s-%s-%s.%s", platform_->app(), platform_->host(), s, FileExtension());
  fd_ = open(path_, O_CREAT | O_WRONLY, 0666);
  if (fd_ == -1) {
    _error("open profiler file[%s]", path_);
    perror("open");
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

int Profiler::Main() {
  return BRISBANE_OK;
}

int Profiler::Exit() {
  return BRISBANE_OK;
}

int Profiler::Write(const char* s, int tab) {
  if (!msg_->WriteString(s)) {
    Flush();
    if (!msg_->WriteString(s)) {
      _error("s[%s]", s);
      return BRISBANE_ERR;
    }
  }
  return BRISBANE_OK;
}

int Profiler::Flush() {
    size_t off = msg_->offset();
    ssize_t ssret = write(fd_, msg_->buf(), off);
    if ((size_t) ssret != off) {
      _error("path[%s] ssret[%zd] off[%zu]", path_, ssret, off);
      perror("write");
      return BRISBANE_ERR;
    }
    msg_->Clear();
    return BRISBANE_OK;
}

int Profiler::CloseFD() {
  Flush();
  if (fd_ != -1) {
    int iret = close(fd_);
    if (iret == -1) {
      _error("close profiler file[%s]", path_);
      perror("close");
    }
  }
  return BRISBANE_OK;
}

const char* Profiler::policy_str(int policy) {
  switch (policy) {
    case brisbane_default:  return "default";
    case brisbane_cpu:      return "cpu";
    case brisbane_nvidia:   return "nvidia";
    case brisbane_amd:      return "amd";
    case brisbane_gpu:      return "gpu";
    case brisbane_phi:      return "phi";
    case brisbane_fpga:     return "fpga";
    case brisbane_data:     return "data";
    case brisbane_profile:  return "profile";
    case brisbane_random:   return "random";
    case brisbane_any:      return "any";
  }
  return policy & brisbane_all ? "all" : "?";
}

} /* namespace rt */
} /* namespace brisbane */


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

namespace iris {
namespace rt {

Profiler::Profiler(Platform* platform, const char *profiler_name) {
  platform_ = platform;
  fd_ = -1;
  msg_ = new Message();
  strcpy(profiler_name_, profiler_name);
}

Profiler::~Profiler() {
  CloseFD();
  if (msg_) delete msg_;
}

int Profiler::OpenFD(const char *path) {
  time_t t = time(NULL);
  if (path != NULL) {
      strcpy(path_, path);
  } else {
      char s[64];
      strftime(s, 64, "%Y%m%d-%H%M%S", localtime(&t));
      sprintf(path_, "%s-%s-%s.%s", platform_->app(), platform_->host(), s, FileExtension());
  }
  fd_ = open(path_, O_CREAT | O_WRONLY, 0666);
  if (fd_ == -1) {
    _error("open profiler file[%s]", path_);
    perror("open");
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

int Profiler::Main() {
  return IRIS_SUCCESS;
}

int Profiler::Exit() {
  return IRIS_SUCCESS;
}

int Profiler::Write(const char* s, int tab) {
  if (!msg_->WriteString(s)) {
    Flush();
    if (!msg_->WriteString(s)) {
      _error("s[%s]", s);
      return IRIS_ERROR;
    }
  }
  return IRIS_SUCCESS;
}

int Profiler::Flush() {
    size_t off = msg_->offset();
    ssize_t ssret = write(fd_, msg_->buf(), off);
    if ((size_t) ssret != off) {
      _error("path[%s] ssret[%zd] off[%zu]", path_, ssret, off);
      perror("write");
      return IRIS_ERROR;
    }
    msg_->Clear();
    return IRIS_SUCCESS;
}

int Profiler::CloseFD() {
  Flush();
  if (fd_ != -1) {
    int iret = close(fd_);
    _info("Profiler %s output in file: %s", profiler_name_,  path_);
    if (iret == -1) {
      _error("close profiler file[%s]", path_);
      perror("close");
    }
  }
  return IRIS_SUCCESS;
}

const char* Profiler::policy_str(int policy) {
  switch (policy) {
    case iris_default:    return "default";
    case iris_cpu:        return "cpu";
    case iris_nvidia:     return "nvidia";
    case iris_amd:        return "amd";
    case iris_gpu_intel:  return "gpu intel";
    case iris_gpu:        return "gpu";
    case iris_phi:        return "phi";
    case iris_fpga:       return "fpga";
    case iris_dsp:        return "dsp";
    case iris_roundrobin: return "roundrobin";
    case iris_depend:     return "depend";
    case iris_data:       return "data";
    case iris_profile:    return "profile";
    case iris_random:     return "random";
    case iris_pending:    return "pending";
    case iris_any:        return "any";
    case iris_custom:     return "custom";
    default: break;
  }
  return policy & iris_all ? "all" : "?";
}

} /* namespace rt */
} /* namespace iris */


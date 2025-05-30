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
      string app_path = platform_->app();
      string app_base_name = app_path;
      size_t last_slash = app_path.find_last_of("/\\");
      if (last_slash != std::string::npos) {
          app_base_name = app_path.substr(last_slash + 1);
      } 
      strftime(s, 64, "%Y%m%d-%H%M%S", localtime(&t));
      sprintf(path_, "%s-%s-%s.%s", app_base_name.c_str(), platform_->host(), s, FileExtension());
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

int Profiler::Write(string s, int tab) {
  //Flush();
  size_t space_left = msg_->free_buf_size();
  while(space_left < s.length()) {
     string s1 = s.substr(0, space_left);
     s = s.substr(space_left);
     if (!msg_->WriteString(s1.c_str())) {
         _error("s[%s]", s.c_str());
         return IRIS_ERROR;
     }
     Flush();
     space_left = msg_->free_buf_size();
  }
  if (s.length() <= space_left) {
      if (!msg_->WriteString(s.c_str())) {
          _error("s[%s]", s.c_str());
          return IRIS_ERROR;
      }
  }
  return IRIS_SUCCESS;
}

int Profiler::Write(const char* s, int tab) {
  string s_str = s;
  return Write(s_str, tab);
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
    _printf("Profiler %s output in file: %s", profiler_name_,  path_);
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
    case iris_julia_policy: return "julia policy";
    case iris_depend:     return "depend";
    case iris_data:       return "data";
    case iris_profile:    return "profile";
    case iris_random:     return "random";
    case iris_pending:    return "pending";
    case iris_sdq:        return "sdq";
    case iris_custom:     return "custom";
    default: break;
  }
  return policy & iris_ftf ? "ftf" : "?";
}

} /* namespace rt */
} /* namespace iris */


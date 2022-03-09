#include "SigHandler.h"
#include "Config.h"
#include "Debug.h"
#if USE_SIGHANDLER
#include <execinfo.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#endif

namespace iris {
namespace rt {

struct sigaction SigHandler::sa_;

SigHandler::SigHandler() {
#if USE_SIGHANDLER
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = Handle;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, &sa_);
#endif
}

SigHandler::~SigHandler() {
}

void SigHandler::Handle(int signum, siginfo_t* si, void* arg) {
#if USE_SIGHANDLER
  void* buf[128];
  size_t size = backtrace(buf, sizeof(buf) / sizeof(void*));
  _error("signum[%d][%s]", signum, strsignal(signum));
  backtrace_symbols_fd(buf, size, STDERR_FILENO);
  sigaction(SIGSEGV, &sa_, NULL);
#endif
}

} /* namespace rt */
} /* namespace iris */


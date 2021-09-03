#ifndef BRISBANE_SRC_RT_SIGHANDLER_H
#define BRISBANE_SRC_RT_SIGHANDLER_H

#include <signal.h>

namespace brisbane {
namespace rt {

class SigHandler {
public:
  SigHandler();
  ~SigHandler();

public:
  static void Handle(int signum, siginfo_t* si, void* arg);

public:
  static struct sigaction sa_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_SIGHANDLER_H */

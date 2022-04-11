#ifndef IRIS_SRC_RT_SIGHANDLER_H
#define IRIS_SRC_RT_SIGHANDLER_H

#include <signal.h>

namespace iris {
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
} /* namespace iris */

#endif /* IRIS_SRC_RT_SIGHANDLER_H */

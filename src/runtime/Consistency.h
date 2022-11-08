#ifndef IRIS_SRC_RT_CONSISTENCY_H
#define IRIS_SRC_RT_CONSISTENCY_H

#include <iris/iris_poly_types.h>
#include "Kernel.h"

namespace iris {
namespace rt {

class Command;
class Mem;
class Scheduler;
class Task;
class Worker;

class Consistency {
public:
  Consistency(Scheduler* scheduler);
  ~Consistency();

  void Resolve(Task* task);
  void Disable() { disable_ = true; }
  void Enable() { disable_ = false; }

private:
  void ResolveKernel(Task* task, Command* cmd);
  void ResolveKernelWithPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg, iris_poly_mem* polymem);
  void ResolveKernelWithoutPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg);
  void ResolveD2H(Task* task, Command* cmd);
private:
  Scheduler* scheduler_;
  pthread_mutex_t mutex_;
  bool disable_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_CONSISTENCY_H */

#ifndef BRISBANE_SRC_RT_CONSISTENCY_H
#define BRISBANE_SRC_RT_CONSISTENCY_H

#include <brisbane/brisbane_poly_types.h>
#include "Kernel.h"

namespace brisbane {
namespace rt {

class Command;
class Mem;
class Scheduler;
class Task;

class Consistency {
public:
  Consistency(Scheduler* scheduler);
  ~Consistency();

  void Resolve(Task* task);

private:
  void ResolveKernel(Task* task, Command* cmd);
  void ResolveKernelWithPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg, brisbane_poly_mem* polymem);
  void ResolveKernelWithoutPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg);
  void ResolveD2H(Task* task, Command* cmd);

private:
  Scheduler* scheduler_;

};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_CONSISTENCY_H */

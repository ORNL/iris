#include "Consistency.h"
#include "Debug.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Scheduler.h"
#include "Task.h"

namespace iris {
namespace rt {

Consistency::Consistency(Scheduler* scheduler) {
  scheduler_ = scheduler;
}

Consistency::~Consistency() {
}

void Consistency::Resolve(Task* task) {
  if (task->system()) return;
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    switch (cmd->type()) {
      case IRIS_CMD_KERNEL:       ResolveKernel(task, cmd);     break;
      case IRIS_CMD_D2H:          ResolveD2H(task, cmd);        break;
    }
  }
}

void Consistency::ResolveKernel(Task* task, Command* cmd) {
//  if (task->parent()) return;
  Device* dev = task->dev();
  iris_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  KernelArg* args = cmd->kernel_args();
  int mem_idx = 0;
  for (int i = 0; i < cmd->kernel_nargs(); i++) {
    KernelArg* arg = args + i;
    Mem* mem = arg->mem;
    if (!mem) continue;
    if (npolymems) ResolveKernelWithPolymem(task, cmd, mem, arg, polymems + mem_idx);
    else ResolveKernelWithoutPolymem(task, cmd, mem, arg);
    mem_idx++;
  }
}

void Consistency::ResolveKernelWithPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg, iris_poly_mem* polymem) {
  Device* dev = task->dev();
  Kernel* kernel = cmd->kernel();
  size_t off = 0UL;
  size_t size = 0UL;
  if (arg->mode == iris_r) {
    off = polymem->typesz * polymem->r0;
    size = polymem->typesz * (polymem->r1 - polymem->r0 + 1);
  } else if (arg->mode == iris_w) {
    off = polymem->typesz * polymem->w0;
    size = polymem->typesz * (polymem->w1 - polymem->w0 + 1);
  } else if (arg->mode == iris_rw) {
    off = polymem->r0 < polymem->w0 ? polymem->r0 : polymem->w0;
    size = polymem->typesz * (polymem->r1 > polymem->w1 ? polymem->r1 - off + 1 : polymem->w1 - off + 1);
    off *= polymem->typesz;
  } else _error("not supprt mode[%d]", arg->mode);

  Device* owner = mem->Owner(off, size);
  if (!owner || mem->IsOwner(off, size, dev)) return;

  Task* task_d2h = new Task(scheduler_->platform());
  task_d2h->set_system();
  Command* d2h = Command::CreateD2H(task, mem, off, size, (char*) mem->host_inter() + off);
  task_d2h->AddCommand(d2h);
  scheduler_->SubmitTaskDirect(task_d2h, owner);
  task_d2h->Wait();

  Command* h2d = arg->mode == iris_r ?
    Command::CreateH2DNP(task, mem, off, size, (char*) mem->host_inter() + off) :
    Command::CreateH2D(task, mem, off, size, (char*) mem->host_inter() + off);
  dev->ExecuteH2D(h2d);

  _trace("kernel[%s] memcpy[%lu] [%s] -> [%s]", kernel->name(), mem->uid(), owner->name(), dev->name());

  task_d2h->Release();
  Command::Release(h2d);
}

void Consistency::ResolveKernelWithoutPolymem(Task* task, Command* cmd, Mem* mem, KernelArg* arg) {
  Device* dev = task->dev();
  Kernel* kernel = cmd->kernel();
  Device* owner = mem->Owner();

  if (!owner || dev == owner || mem->IsOwner(0, mem->size(), dev)) return;

  Task* task_d2h = new Task(scheduler_->platform());
  task_d2h->set_system();
  Command* d2h = Command::CreateD2H(task_d2h, mem, 0, mem->size(), mem->host_inter());
  task_d2h->AddCommand(d2h);
  scheduler_->SubmitTaskDirect(task_d2h, owner);
  //_todo("HERE task[%lu] dev[%d] owner[%d]", task_d2h->uid(), dev->devno(), owner->devno());
  task_d2h->Wait();
  //_todo("THERE task[%lu] dev[%d] owner[%d]", task_d2h->uid(), dev->devno(), owner->devno());

  Command* h2d = Command::CreateH2DNP(task, mem, 0, mem->size(), mem->host_inter());
  dev->ExecuteH2D(h2d);

  _trace("kernel[%s] mem[%lu] [%s][%d] -> [%s][%d]", kernel->name(), mem->uid(), owner->name(), owner->devno(), dev->name(), dev->devno());

  task_d2h->Release();
  Command::Release(h2d);
}

void Consistency::ResolveD2H(Task* task, Command* cmd) {
  Device* dev = task->dev();
  Mem* mem = cmd->mem();
  Device* owner = mem->Owner();
  if (!owner || dev == owner || mem->IsOwner(0, mem->size(), dev)) return;
  Task* task_d2h = new Task(scheduler_->platform());
  task_d2h->set_system();
  Command* d2h = Command::CreateD2H(task_d2h, mem, 0, mem->size(), mem->host_inter());
  task_d2h->AddCommand(d2h);
  scheduler_->SubmitTaskDirect(task_d2h, owner);
  task_d2h->Wait();

  Command* h2d = Command::CreateH2DNP(task, mem, 0, mem->size(), mem->host_inter());
  dev->ExecuteH2D(h2d);

  _trace("mem[%lu] [%s][%d] -> [%s][%d]", mem->uid(), owner->name(), owner->devno(), dev->name(), dev->devno());

  task_d2h->Release();
  Command::Release(h2d);
}

} /* namespace rt */
} /* namespace iris */


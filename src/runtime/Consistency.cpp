#include "Consistency.h"
#include "Debug.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Scheduler.h"
#include "Task.h"
#include "Worker.h"
#include "Thread.h"

#include <cstring>
#include <unistd.h>

namespace iris {
namespace rt {

Consistency::Consistency(Scheduler* scheduler) {
  scheduler_ = scheduler;
  pthread_mutex_init(&mutex_, NULL);
  disable_ = false;
}

Consistency::~Consistency() {
  pthread_mutex_destroy(&mutex_);
}

void Consistency::Resolve(Task* task) {
  if (task->system()) return;
  if (disable_) return;
  if (task->disable_consistency()) return;
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
  //Device* dev = task->dev();
  iris_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  KernelArg* args = cmd->kernel_args();
  int mem_idx = 0;
  for (int i = 0; i < cmd->kernel_nargs(); i++) {
    KernelArg* arg = args + i;
    BaseMem* bmem = (BaseMem *)arg->mem;
    if (!bmem) continue;
    if (bmem->GetMemHandlerType() != IRIS_MEM) continue;
    Mem* mem = (Mem *)bmem;
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
  } else _error("not in supported mode[%d]", arg->mode);

  Device* owner = mem->Owner(off, size);
  if (!owner || mem->IsOwner(off, size, dev)) return;

  const char *d2h_tn = "Internal-D2H";
  Task* task_d2h = new Task(scheduler_->platform(), IRIS_TASK, d2h_tn);
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
  pthread_mutex_lock(&mutex_);
  //issue the first stage (d2h); get the data from the other device
  const char* d2h_tn = "Internal-D2H";
  Task* task_d2h = new Task(scheduler_->platform(), IRIS_TASK, d2h_tn);
  Command* d2h = Command::CreateD2H(task, mem, 0, mem->size(), mem->host_inter());
  d2h->set_name(d2h_tn);
  task_d2h->set_name(d2h_tn);
  task_d2h->set_system();
  task_d2h->AddCommand(d2h);
  task_d2h->set_internal_memory_transfer();
  d2h->set_internal_memory_transfer();
  bool context_shift = owner->IsContextChangeRequired();
  if (context_shift) owner->ResetContext();
  scheduler_->SubmitTaskDirect(task_d2h,owner);
  task_d2h->Wait();
  if (context_shift) dev->ResetContext();

  const char *h2d_tn = "Internal-H2D";
  Task* task_h2d = new Task(scheduler_->platform(), IRIS_TASK, h2d_tn);
  Command* h2d = Command::CreateH2D(task, mem, 0, mem->size(), mem->host_inter());
  h2d->set_name(h2d_tn);
  h2d->set_internal_memory_transfer();
  task_h2d->set_name(h2d_tn);
  task_h2d->set_system();
  task_h2d->AddCommand(h2d);
  task_h2d->set_internal_memory_transfer();
  scheduler_->SubmitTaskDirect(task_h2d,dev);
  task_h2d->Wait();
  pthread_mutex_unlock(&mutex_);

  _trace("kernel[%s] mem[%lu] [%s][%d] -> [%s][%d]", kernel->name(), mem->uid(), owner->name(), owner->devno(), dev->name(), dev->devno());

  task_d2h->Release();
  task_h2d->Release();
}

void Consistency::ResolveD2H(Task* task, Command* cmd) {
  Device* dev = task->dev();
  BaseMem* dmem = (BaseMem *)cmd->mem();
  if (dmem && dmem->GetMemHandlerType() == IRIS_DMEM) {
    //we're using datamem so there is no need to execute this memory transfer --- just flush
    dev->ExecuteMemFlushOut(cmd);
    return;
  }
  Mem* mem = (Mem *)cmd->mem();
  Device* owner = mem->Owner();
  if (!owner || dev == owner || mem->IsOwner(0, mem->size(), dev)) return;
  const char *d2h_tn = "Internal-D2H";
  Task* task_d2h = new Task(scheduler_->platform(), IRIS_TASK, d2h_tn);
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


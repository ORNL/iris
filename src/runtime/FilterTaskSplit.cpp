#include "FilterTaskSplit.h"
#include "Config.h"
#include "Debug.h"
#include "Command.h"
#include "Kernel.h"
#include "Mem.h"
#include "Polyhedral.h"
#include "Task.h"
#include <alloca.h>

namespace iris {
namespace rt {

FilterTaskSplit::FilterTaskSplit(Polyhedral* polyhedral, Platform* platform) {
  polyhedral_ = polyhedral;
  platform_ = platform;
}

FilterTaskSplit::~FilterTaskSplit() {
}

int FilterTaskSplit::Execute(Task* task) {
  Command* cmd_kernel = task->cmd_kernel();
  if (!cmd_kernel) return IRIS_SUCCESS;
  Kernel* kernel = cmd_kernel->kernel();

  int poly_available = polyhedral_->Kernel(kernel->name());
  if (!poly_available) return IRIS_ERROR;
  int nmems = 0;
  KernelArg* args = cmd_kernel->kernel_args();
  for (int idx = 0; idx < cmd_kernel->kernel_nargs(); idx++) {
    KernelArg* arg = args + idx;
    Mem* mem = (Mem *)arg->mem;
    if (mem) nmems++;
    else polyhedral_->SetArg(idx, arg->size, arg->value); 
  }

  int dim = cmd_kernel->dim();
  size_t off[3];
  size_t gws[3];
  size_t lws[3];
  for (int i = 0; i < 3; i++) {
    off[i] = cmd_kernel->off(i);
    gws[i] = cmd_kernel->gws(i);
    lws[i] = cmd_kernel->lws(i);
  }

  size_t new_gws[3] = { gws[0], gws[1], gws[2] };

  iris_poly_mem* plmems = new iris_poly_mem[nmems];
  Mem** plmems_mem = (Mem**) alloca(sizeof(Mem*) * nmems);
  size_t chunk_size = gws[0] / (platform_->ndevs());
  _debug("gws[%lu] dim[%d] chunk_size[%lu]", gws[0], dim, chunk_size);
  size_t gws0 = gws[0];
  bool left_gws = gws[0] % chunk_size;
  size_t nchunks = gws[0] / chunk_size + (left_gws ? 1 : 0);
  Task** subtasks = new Task*[nchunks];
  for (size_t i = 0; i < nchunks; i++) {
    subtasks[i] = new Task(platform_);
    off[0] = i * chunk_size;
    if (left_gws && i == nchunks - 1) gws[0] = gws0 - i * chunk_size;
    else gws[0] = chunk_size;

    polyhedral_->Launch(dim, off, gws, new_gws, lws);
    int mem_idx = 0;
    for (int idx = 0; idx < cmd_kernel->kernel_nargs(); idx++) {
      KernelArg* arg = args + idx;
      Mem* mem = (Mem *)arg->mem;
      if (mem) {
        polyhedral_->GetMem(idx, plmems + mem_idx);
        _trace("kernel[%s] idx[%d] mem[%lu] typesz[%lu] read[%lu,%lu] write[%lu,%lu]", kernel->name(), idx, mem->uid(), plmems[mem_idx].typesz, plmems[mem_idx].r0, plmems[mem_idx].r1, plmems[mem_idx].w0, plmems[mem_idx].w1);
        plmems_mem[mem_idx] = mem;
        mem_idx++;
      }
    }
    for (int j = 0; j < task->ncmds(); j++) {
      Command* cmd = task->cmd(j);
      if (cmd->type_h2d() || cmd->type_h2broadcast()) {
        Mem* mem = (Mem *)cmd->mem();
        for (int k = 0; k < nmems; k++) {
          if (plmems_mem[k] == mem) {
            iris_poly_mem* plmem = plmems + k; 
            if (plmem->r0 > plmem->r1) _error("invalid poly_mem r0[%lu] r1[%lu]", plmem->r0, plmem->r1);
            Command* sub_cmd = Command::CreateH2DNP(subtasks[i], mem, plmem->typesz * plmem->r0, plmem->typesz * (plmem->r1 - plmem->r0 + 1), (char*) cmd->host() + plmem->typesz * plmem->r0);
            subtasks[i]->AddCommand(sub_cmd);
          }
        }
      } else if (cmd->type_d2h()) {
        Mem* mem = (Mem *)cmd->mem();
        for (int k = 0; k < nmems; k++) {
          if (plmems_mem[k] == mem) {
            iris_poly_mem* plmem = plmems + k; 
            if (plmem->w0 > plmem->w1) _error("invalid poly_mem w0[%lu] w1[%lu]", plmem->w0, plmem->w1);
            Command* sub_cmd = Command::CreateD2H(subtasks[i], mem, plmem->typesz * plmem->w0, plmem->typesz * (plmem->w1 - plmem->w0 + 1), (char*) cmd->host() + plmem->typesz * plmem->w0);
            subtasks[i]->AddCommand(sub_cmd);
          }
        }
      } else if (cmd->type_kernel()) {
        Command* sub_cmd = Command::CreateKernelPolyMem(subtasks[i], cmd, off, gws, plmems, nmems);
        subtasks[i]->AddCommand(sub_cmd);
      }
    }
    task->AddSubtask(subtasks[i]);
  }
  task->ClearCommands();
  delete[] plmems;

  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


#include "Command.h"
#include "Debug.h"
#include "Platform.h"
#include "Pool.h"
#include "Task.h"
#include "Timer.h"
#include "Mem.h"
#include "DataMem.h"

namespace iris {
namespace rt {

Command::Command() {
  Clear(true);
}

Command::Command(Task* task, int type) {
  Clear(true);
  Set(task, type);
}

Command::~Command() {
  if (kernel_args_) delete[] kernel_args_;
  if (kernel_) kernel_->Release();
  if (npolymems_ && polymems_) delete polymems_;
  if (selector_kernel_params_) free(selector_kernel_params_);
  if (params_map_) delete [] params_map_; 
}

void Command::set_params_map(int *pmap) { 
    params_map_ = new int[kernel_nargs_]; 
    memcpy(params_map_, pmap, sizeof(int)*kernel_nargs_); 
}

void Command::Clear(bool init) {
  host_ = NULL;
  params_map_ = NULL;
  kernel_ = NULL;
  task_ = NULL;
  mem_ = NULL;
  platform_ = NULL;
  kernel_args_ = NULL;
  polymems_ = NULL;
  func_params_ = NULL;
  params_ = NULL;
  type_name_ = NULL;
  name_ = NULL;
  selector_kernel_params_ = NULL;
  time_ = 0.0;
  internal_memory_transfer_ = false;
  kernel_args_ = NULL;
  kernel_nargs_ = 0;
  last_ = false;
  selector_kernel_ = NULL;
  selector_kernel_params_ = NULL;
  polymems_ = NULL;
  npolymems_ = 0;
  params_map_ = NULL;
  name_ = NULL;
  off_[0] = 0; off_[1] = 0; off_[2] = 0;
  gws_[0] = 0; gws_[1] = 1; gws_[2] = 1;
  lws_[0] = 0; lws_[1] = 1; lws_[2] = 1;
  dim_ = 1;
  elem_size_ = 0;
  if (init) {
    kernel_nargs_max_ = IRIS_CMD_KERNEL_NARGS_MAX;
    kernel_args_ = new KernelArg[kernel_nargs_max_];
    for (int i = 0; i < kernel_nargs_max_; i++) {
      kernel_args_[i].mem = NULL;
    }
  }
}

void Command::Set(Task* task, int type) {
  task_ = task;
  type_ = type;
  switch(type){
    case IRIS_CMD_INIT:        type_name_= const_cast<char*>("Init");    break;
    case IRIS_CMD_KERNEL:      type_name_= const_cast<char*>("Kernel");  break;
    case IRIS_CMD_MALLOC:      type_name_= const_cast<char*>("Malloc");  break;
    case IRIS_CMD_H2D:         type_name_= const_cast<char*>("H2D");     break;
    case IRIS_CMD_D2D:         type_name_= const_cast<char*>("D2D");     break;
    case IRIS_CMD_H2BROADCAST: type_name_= const_cast<char*>("H2Broadcast");     break;
    case IRIS_CMD_H2DNP:       type_name_= const_cast<char*>("H2DNP");   break;
    case IRIS_CMD_D2H:         type_name_= const_cast<char*>("D2H");     break;
    case IRIS_CMD_MEM_FLUSH:   type_name_= const_cast<char*>("MemFlush");     break;
    case IRIS_CMD_MAP:         type_name_= const_cast<char*>("Map");     break;
    case IRIS_CMD_RELEASE_MEM: type_name_= const_cast<char*>("Release"); break;
    case IRIS_CMD_HOST:        type_name_= const_cast<char*>("Host");    break;
    case IRIS_CMD_CUSTOM:      type_name_= const_cast<char*>("Custom");  break;
    case IRIS_CMD_RESET_INPUT: type_name_= const_cast<char*>("ResetMem");  break;
    default: _error("cmd type[0x%x]", type);
  }
  if (task->ncmds() == 0 && task->name()){ 
    name_ = task->name();
  }
  platform_ = task->platform();
}

double Command::SetTime(double t, bool incr) {
//  if (time_ != 0.0) _error("double set time[%lf]", t);
  if (incr) {
    time_ += t;
    task_->TimeInc(t);
  }
  else {
    time_ = t;
  }
  return time_;
}

Command* Command::Create(Task* task, int type) {
  return task->platform()->pool()->GetCommand(task, type);
}

Command* Command::CreateInit(Task* task) {
  return Create(task, IRIS_CMD_INIT);
}

Command* Command::CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  Command* cmd = Create(task, IRIS_CMD_KERNEL);
  cmd->kernel_ = kernel;
  if (cmd->kernel_args_) delete[] cmd->kernel_args_;
  cmd->kernel_args_ = kernel->ExportArgs();
  cmd->kernel_nargs_max_ = kernel->nargs();
  cmd->kernel_nargs_ = kernel->nargs();
  cmd->dim_ = dim;
  for (int i = 0; i < dim; i++) {
    cmd->off_[i] = off ? off[i] : 0ULL;
    cmd->gws_[i] = gws[i];
    cmd->lws_[i] = lws ? lws[i] : 0ULL;
  }
  for (int i = dim; i < 3; i++) {
    cmd->off_[i] = 0;
    cmd->gws_[i] = 1;
    cmd->lws_[i] = 1;
  }
  return cmd;
}

Command* Command::CreateKernel(Task* task, Kernel* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges) {
  Command* cmd = Create(task, IRIS_CMD_KERNEL);
  cmd->kernel_ = kernel;
  cmd->dim_ = dim;
  for (int i = 0; i < dim; i++) {
    cmd->off_[i] = off ? off[i] : 0ULL;
    cmd->gws_[i] = gws[i];
    cmd->lws_[i] = lws ? lws[i] : 0ULL;
  }
  for (int i = dim; i < 3; i++) {
    cmd->off_[i] = 0;
    cmd->gws_[i] = 1;
    cmd->lws_[i] = 1;
  }
  if (nparams > cmd->kernel_nargs_max_) {
    delete[] cmd->kernel_args_;
    cmd->kernel_args_ = new KernelArg[nparams];
    cmd->kernel_nargs_max_ = nparams;
  }
  cmd->kernel_nargs_ = nparams;
  KernelArg* args = cmd->kernel_args();
  for (int i = 0; i < nparams; i++) {
    KernelArg* arg = args + i;
    void* param = params[i];
    int param_info = params_info[i];
    if (param_info > 0) {
      arg->mem = NULL;
      arg->off = 0ULL;
      arg->size = param_info;
      if (param) memcpy(arg->value, param, arg->size);
      continue;
    }
    size_t mem_off = 0ULL;
    BaseMem* mem = cmd->platform_->GetMem(*((iris_mem*) param));
    if (!mem) mem = cmd->platform_->GetMem(param, &mem_off);
    if (!mem) {
      _error("no mem[%p] task[%ld:%s]", ((iris_mem*) param), task->uid(), task->name());
      continue;
    }
    if (mem->GetMemHandlerType() == IRIS_DMEM) kernel->add_dmem((DataMem *)mem, i, param_info);
    if (mem->GetMemHandlerType() == IRIS_DMEM_REGION) kernel->add_dmem_region((DataMemRegion *)mem, i, param_info);
    arg->mem = mem;
    arg->off = mem_off;
    if (params_off) arg->off = params_off[i];
    arg->mode = param_info;
    arg->mem_off = memranges ? memranges[i * 2] : 0;
    arg->mem_size = memranges ? memranges[i * 2 + 1] : mem->size();
  }
  return cmd;
}

Command* Command::CreateKernelPolyMem(Task* task, Command* pcmd, size_t* off, size_t* gws, iris_poly_mem* polymems, int npolymems) {
  Kernel* kernel = pcmd->kernel();
  int dim = pcmd->dim();
  size_t* lws = pcmd->lws();
  int nparams = pcmd->kernel_nargs();

  Command* cmd = Create(task, IRIS_CMD_KERNEL);
  cmd->kernel_ = kernel;
  cmd->dim_ = dim;
  for (int i = 0; i < dim; i++) {
    cmd->off_[i] = off[i];
    cmd->gws_[i] = gws[i];
    cmd->lws_[i] = lws[i];
  }
  for (int i = dim; i < 3; i++) {
    cmd->off_[i] = 0;
    cmd->gws_[i] = 1;
    cmd->lws_[i] = 1;
  }
  if (nparams > cmd->kernel_nargs_max_) {
    delete[] cmd->kernel_args_;
    cmd->kernel_args_ = new KernelArg[nparams];
    cmd->kernel_nargs_max_ = nparams;
  }
  cmd->kernel_nargs_ = nparams;
  for (int i = 0; i < nparams; i++) {
    KernelArg* arg = cmd->kernel_arg(i);
    KernelArg* parg = pcmd->kernel_arg(i);
    arg->size = parg->size;
    arg->mem = parg->mem;
#if 0
    arg->mem_off = parg->mem_off;
    arg->mem_size = parg->mem_size;
    arg->off = parg->off;
#else
    arg->mem_off = 0;
    arg->mem_size = parg->mem_size;
    arg->off = 0;
#endif
    arg->mode = parg->mode;
    if (!arg->mem) memcpy(arg->value, parg->value, arg->size);
  }
  cmd->npolymems_ = npolymems;
  cmd->polymems_ = new iris_poly_mem[npolymems];
  memcpy(cmd->polymems_, polymems, sizeof(iris_poly_mem) * npolymems);
  return cmd;
}

Command* Command::CreateMalloc(Task* task, Mem* mem) {
  Command* cmd = Create(task, IRIS_CMD_MALLOC);
  cmd->mem_ = mem;
  return cmd;
}

Command* Command::CreateMemFlushOut(Task* task, DataMem* mem) {
  Command* cmd = Create(task, IRIS_CMD_MEM_FLUSH);
  cmd->mem_ = mem;
  return cmd;
}

Command* Command::CreateMemResetInput(Task* task, BaseMem *mem, uint8_t reset_value) {
  Command* cmd = Create(task, IRIS_CMD_RESET_INPUT);
  cmd->mem_ = mem;
  cmd->reset_value_ = reset_value;
  return cmd;
}

Command* Command::CreateH2Broadcast(Task* task, Mem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Command* cmd = Create(task, IRIS_CMD_H2BROADCAST);
  cmd->mem_ = mem;
  cmd->dim_ = dim;
  size_t size = elem_size;
  for(int i=0; i<dim; i++) {
    cmd->off_[i] = off[i];
    cmd->gws_[i] = host_sizes[i];
    cmd->lws_[i] = dev_sizes[i];
    size *= dev_sizes[i];
  }
  cmd->elem_size_ = elem_size;
  cmd->size_ = size;
  cmd->host_ = host;
  cmd->exclusive_ = true;
  return cmd;
}

Command* Command::CreateH2D(Task* task, Mem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Command* cmd = Create(task, IRIS_CMD_H2D);
  cmd->mem_ = mem;
  cmd->dim_ = dim;
  size_t size = elem_size;
  for(int i=0; i<dim; i++) {
    cmd->off_[i] = off[i];
    cmd->gws_[i] = host_sizes[i];
    cmd->lws_[i] = dev_sizes[i];
    size *= dev_sizes[i];
  }
  cmd->elem_size_ = elem_size;
  cmd->size_ = size;
  cmd->host_ = host;
  cmd->exclusive_ = true;
  return cmd;
}

Command* Command::CreateH2Broadcast(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, IRIS_CMD_H2BROADCAST);
  cmd->dim_ = 1;
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  cmd->exclusive_ = true;
  return cmd;
}
Command* Command::CreateD2D(Task* task, Mem* mem, size_t off, size_t size, void* host, int src_dev) {
  Command* cmd = Create(task, IRIS_CMD_D2D);
  cmd->dim_ = 1;
  cmd->src_dev_ = src_dev;
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  cmd->exclusive_ = true;
  return cmd;
}

Command* Command::CreateH2D(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, IRIS_CMD_H2D);
  cmd->dim_ = 1;
  cmd->mem_ = mem;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  cmd->exclusive_ = true;
  return cmd;
}

Command* Command::CreateH2DNP(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, IRIS_CMD_H2DNP);
  cmd->mem_ = mem;
  cmd->dim_ = 1;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  cmd->exclusive_ = false;
  return cmd;
}

Command* Command::CreateD2H(Task* task, Mem* mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  Command* cmd = Create(task, IRIS_CMD_D2H);
  cmd->mem_ = mem;
  cmd->dim_ = dim;
  size_t size = elem_size;
  for(int i=0; i<dim; i++) {
    cmd->off_[i] = off[i];
    cmd->gws_[i] = host_sizes[i];
    cmd->lws_[i] = dev_sizes[i];
    size *= dev_sizes[i];
  }
  cmd->elem_size_ = elem_size;
  cmd->size_ = size;
  cmd->host_ = host;
  //mem->get_d2h_cmds().push_back(cmd);
  return cmd;
}

Command* Command::CreateD2H(Task* task, Mem* mem, size_t off, size_t size, void* host) {
  Command* cmd = Create(task, IRIS_CMD_D2H);
  cmd->mem_ = mem;
  cmd->dim_ = 1;
  cmd->off_[0] = off;
  cmd->size_ = size;
  cmd->host_ = host;
  //mem->get_d2h_cmds().push_back(cmd);
  return cmd;
}

Command* Command::CreateMap(Task* task, void* host, size_t size) {
  Command* cmd = Create(task, IRIS_CMD_MAP);
  cmd->host_ = host;
  cmd->size_ = size;
  return cmd;
}

Command* Command::CreateMapTo(Task* task, void* host) {
  Command* cmd = Create(task, IRIS_CMD_MAP_TO);
  cmd->host_ = host;
  return cmd;
}

Command* Command::CreateMapFrom(Task* task, void* host) {
  Command* cmd = Create(task, IRIS_CMD_MAP_FROM);
  cmd->host_ = host;
  return cmd;
}

Command* Command::CreateReleaseMem(Task* task, Mem* mem) {
  Command* cmd = Create(task, IRIS_CMD_RELEASE_MEM);
  cmd->mem_ = mem;
  return cmd;
}

Command* Command::CreateHost(Task* task, iris_host_task func, void* params) {
  Command* cmd = Create(task, IRIS_CMD_HOST);
  cmd->func_ = func;
  cmd->func_params_ = params;
  return cmd;
}

Command* Command::CreateCustom(Task* task, int tag, void* params, size_t params_size) {
  Command* cmd = Create(task, IRIS_CMD_CUSTOM);
  cmd->tag_ = tag;
  cmd->params_ = (char*) malloc(params_size);
  memcpy(cmd->params_, params, params_size);
  return cmd;
}

void Command::Release(Command* cmd) {
  delete cmd;
}

void Command::set_selector_kernel(iris_selector_kernel func, void* params, size_t params_size) {
  selector_kernel_ = func;
  selector_kernel_params_ = malloc(params_size);
  memcpy(selector_kernel_params_, params, params_size);
}

} /* namespace rt */
} /* namespace iris */


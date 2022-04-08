#include "Device.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"

namespace iris {
namespace rt {

Device::Device(int devno, int platform) {
  devno_ = devno;
  platform_ = platform;
  busy_ = false;
  enable_ = false;
  shared_memory_buffers_ = false;
  is_vendor_specific_kernel_ = false;
  nqueues_ = 32;
  q_ = 0;
  memset(vendor_, 0, sizeof(vendor_));
  memset(name_, 0, sizeof(name_));
  memset(version_, 0, sizeof(version_));
  memset(kernel_path_, 0, sizeof(kernel_path_));
  timer_ = new Timer();
  hook_task_pre_ = NULL;
  hook_task_post_ = NULL;
  hook_command_pre_ = NULL;
  hook_command_post_ = NULL;
}

Device::~Device() {
  delete timer_;
}

void Device::Execute(Task* task) {
  busy_ = true;
  if (hook_task_pre_) hook_task_pre_(task);
  TaskPre(task);
  task->set_time_start(timer_->Now());
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (hook_command_pre_) hook_command_pre_(cmd);
    cmd->set_time_start(timer_->Now());
    switch (cmd->type()) {
      case IRIS_CMD_INIT:         ExecuteInit(cmd);       break;
      case IRIS_CMD_KERNEL:       ExecuteKernel(cmd);     break;
      case IRIS_CMD_MALLOC:       ExecuteMalloc(cmd);     break;
      case IRIS_CMD_H2D:          ExecuteH2D(cmd);        break;
      case IRIS_CMD_H2DNP:        ExecuteH2DNP(cmd);      break;
      case IRIS_CMD_D2H:          ExecuteD2H(cmd);        break;
      case IRIS_CMD_MAP:          ExecuteMap(cmd);        break;
      case IRIS_CMD_RELEASE_MEM:  ExecuteReleaseMem(cmd); break;
      case IRIS_CMD_HOST:         ExecuteHost(cmd);       break;
      case IRIS_CMD_CUSTOM:       ExecuteCustom(cmd);     break;
      default: _error("cmd type[0x%x]", cmd->type());
    }
    cmd->set_time_end(timer_->Now());
    if (hook_command_post_) hook_command_post_(cmd);
#ifndef IRIS_SYNC_EXECUTION
    if (cmd->last()) AddCallback(task);
#endif
  }
  task->set_time_end(timer_->Now());
  TaskPost(task);
  if (hook_task_post_) hook_task_post_(task);
//  if (++q_ >= nqueues_) q_ = 0;
  if (!task->system()) _trace("task[%lu] complete dev[%d][%s] time[%lf]", task->uid(), devno(), name(), task->time());
#ifdef IRIS_SYNC_EXECUTION
  task->Complete();
#endif
  busy_ = false;
}

void Device::ExecuteInit(Command* cmd) {
  timer_->Start(IRIS_TIMER_INIT);
  cmd->set_time_start(timer_->Now());
  if (SupportJIT()) {
    char* tmpdir = NULL;
    char* src = NULL;
    char* bin = NULL;
    Platform::GetPlatform()->EnvironmentGet("TMPDIR", &tmpdir, NULL);
    Platform::GetPlatform()->EnvironmentGet(kernel_src(), &src, NULL);
    Platform::GetPlatform()->EnvironmentGet(kernel_bin(), &bin, NULL);
    bool stat_src = Utils::Exist(src);
    bool stat_bin = Utils::Exist(bin);
    if (!stat_src && !stat_bin) {
      _error("NO KERNEL SRC[%s] NO KERNEL BIN[%s]", src, bin);
    } else if (!stat_src && stat_bin) {
      strncpy(kernel_path_, bin, strlen(bin));
    } else if (stat_src && !stat_bin) {
      sprintf(kernel_path_, "%s/%s-%d", tmpdir, bin, devno_);
      errid_ = Compile(src);
    } else {
      long mtime_src = Utils::Mtime(src);
      long mtime_bin = Utils::Mtime(bin);
      if (mtime_src > mtime_bin) {
        sprintf(kernel_path_, "%s/%s-%d", tmpdir, bin, devno_);
        errid_ = Compile(src);
      } else strncpy(kernel_path_, bin, strlen(bin));
    }
    if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  }
  errid_ = Init();
  if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_INIT);
  cmd->SetTime(time);
  if (Platform::GetPlatform()->enable_scheduling_history()) Platform::GetPlatform()->scheduling_history()->AddKernel(cmd);
  enable_ = true;
}

void Device::ExecuteKernel(Command* cmd) {
  timer_->Start(IRIS_TIMER_KERNEL);
  cmd->set_time_start(timer_->Now());
  Kernel* kernel = ExecuteSelectorKernel(cmd);
  int dim = cmd->dim();
  size_t* off = cmd->off();
  size_t* gws = cmd->gws();
  size_t gws0 = gws[0];
  size_t* lws = cmd->lws();
  bool reduction = false;
  iris_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  int max_idx = 0;
  int mem_idx = 0;
  set_vendor_specific_kernel(false);
  KernelLaunchInit(kernel);
  KernelArg* args = cmd->kernel_args();
  for (int idx = 0; idx < cmd->kernel_nargs(); idx++) {
    if (idx > max_idx) max_idx = idx;
    KernelArg* arg = args + idx;
    Mem* mem = arg->mem;
    if (mem) {
      if (arg->mode == iris_w || arg->mode == iris_rw) {
        if (npolymems) {
          iris_poly_mem* pm = polymems + mem_idx;
          mem->SetOwner(pm->typesz * pm->w0, pm->typesz * (pm->w1 - pm->w0 + 1), this);
        } else mem->SetOwner(arg->mem_off, arg->mem_size, this);
      }
      if (mem->mode() & iris_reduction) {
        lws = (size_t*) alloca(3 * sizeof(size_t));
        lws[0] = 1;
        lws[1] = 1;
        lws[2] = 1;
        while (max_compute_units_ * lws[0] < gws[0]) lws[0] <<= 1;
        while (max_work_item_sizes_[0] / 4 < lws[0]) lws[0] >>= 1;
        size_t expansion = (gws[0] + lws[0] - 1) / lws[0];
        gws[0] = lws[0] * expansion;
        mem->Expand(expansion);
        KernelSetMem(kernel, idx, mem, arg->off);
        KernelSetArg(kernel, idx + 1, lws[0] * mem->type_size(), NULL);
        reduction = true;
        if (idx + 1 > max_idx) max_idx = idx + 1;
        idx++;
      } else KernelSetMem(kernel, idx, mem, arg->off);
      mem_idx++;
    } else KernelSetArg(kernel, idx, arg->size, arg->value);
  }
#if 0
  if (reduction) {
    _trace("max_idx+1[%d] gws[%lu]", max_idx + 1, gws0);
    KernelSetArg(kernel, max_idx + 1, sizeof(size_t), &gws0);
  }
#endif
  errid_ = KernelLaunch(kernel, dim, off, gws, lws[0] > 0 ? lws : NULL);
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_KERNEL);
  cmd->SetTime(time);
  cmd->kernel()->history()->AddKernel(cmd, this, time);
  if (Platform::GetPlatform()->enable_scheduling_history()) Platform::GetPlatform()->scheduling_history()->AddKernel(cmd);
}

void Device::ExecuteMalloc(Command* cmd) {
  cmd->set_time_start(timer_->Now());
  Mem* mem = cmd->mem();
  void* arch = mem->arch(this);
  cmd->set_time_end(timer_->Now());
  if (Platform::GetPlatform()->enable_scheduling_history()) Platform::GetPlatform()->scheduling_history()->AddH2D(cmd);
  _trace("dev[%d] malloc[%p]", devno_, arch);
}

void Device::ExecuteH2D(Command* cmd) {
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  bool exclusive = cmd->exclusive();
  void* host = cmd->host();
  if (exclusive) mem->SetOwner(off, size, this);
  else mem->AddOwner(off, size, this);
  timer_->Start(IRIS_TIMER_H2D);
  cmd->set_time_start(timer_->Now());
  bool is_mem_xfer_skipped = true;
  //printf("DeviceH2D device no:%d\n", devno_);
  // Decide whether memory transfer required or not
  std::vector<Command *>  d2h_cmds_of_mem = mem->get_d2h_cmds();
  if (d2h_cmds_of_mem.size() == 0 || ! mem->is_intermediate())
      is_mem_xfer_skipped = false;
  if (is_mem_xfer_skipped) for (Command *d2h_cmd : d2h_cmds_of_mem) {
     //printf("DeviceH2D task:%p %p %p %p %d %d %d\n", cmd->task(), d2h_cmd->task(), cmd->task()->dev(), d2h_cmd->task()->dev(), cmd->task()->devno(), d2h_cmd->task()->devno(), devno_);
     if (cmd->task() == d2h_cmd->task())
         is_mem_xfer_skipped = false;
     if (cmd->task()->dev() != d2h_cmd->task()->dev())
         is_mem_xfer_skipped = false;
  }
  if (is_mem_xfer_skipped) {
      _trace("Dev[%d][%s] MemH2D is skipped", devno_, name_);
      return;
  }
  errid_ = MemH2D(mem, off, size, host);
  if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_H2D);
  cmd->SetTime(time);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) cmd_kernel->kernel()->history()->AddH2D(cmd, this, time);
  else Platform::GetPlatform()->null_kernel()->history()->AddH2D(cmd, this, time);
  if (Platform::GetPlatform()->enable_scheduling_history()) Platform::GetPlatform()->scheduling_history()->AddH2D(cmd);
}

void Device::ExecuteH2DNP(Command* cmd) {
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t size = cmd->size();
//  if (mem->IsOwner(off, size, this)) return;
  return ExecuteH2D(cmd);
}

void Device::ExecuteD2H(Command* cmd) {
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t size = cmd->size();
  void* host = cmd->host();
  int mode = mem->mode();
  int expansion = mem->expansion();
  timer_->Start(IRIS_TIMER_D2H);
  cmd->set_time_start(timer_->Now());
  errid_ = IRIS_SUCCESS;
  bool is_mem_xfer_skipped = true;
  // Decide whether memory transfer required or not
  //printf("DeviceH2D device no:%d\n", devno_);
  std::vector<Command *>  h2d_cmds_of_mem = mem->get_h2d_cmds();
  if (h2d_cmds_of_mem.size() == 0 || ! mem->is_intermediate())
      is_mem_xfer_skipped = false;
  if (is_mem_xfer_skipped) for (Command *h2d_cmd : h2d_cmds_of_mem) {
     //printf("DeviceD2H task:%p %p %p %p %d %d %d\n", cmd->task(), h2d_cmd->task(), cmd->task()->dev(), h2d_cmd->task()->dev(), cmd->task()->devno(), h2d_cmd->task()->devno(), devno_);
     if (cmd->task()->dev() != h2d_cmd->task()->dev())
         is_mem_xfer_skipped = false;
  }
  if (is_mem_xfer_skipped) {
      _trace("Dev[%d][%s] MemD2H is skipped", devno_, name_);
      return;
  }

  if (mode & iris_reduction) {
    errid_ = MemD2H(mem, off, mem->size() * expansion, mem->host_inter());
    Reduction::GetInstance()->Reduce(mem, host, size);
  } else errid_ = MemD2H(mem, off, size, host);

  if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_D2H);
  cmd->SetTime(time);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) cmd_kernel->kernel()->history()->AddD2H(cmd, this, time);
  else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, time);
  if (Platform::GetPlatform()->enable_scheduling_history()) Platform::GetPlatform()->scheduling_history()->AddD2H(cmd);
}

void Device::ExecuteMap(Command* cmd) {
  void* host = cmd->host();
  size_t size = cmd->size();
}

void Device::ExecuteReleaseMem(Command* cmd) {
  Mem* mem = cmd->mem();
  mem->Release(); 
}

void Device::ExecuteHost(Command* cmd) {
  iris_host_task func = cmd->func();
  void* params = cmd->func_params();
  const int dev = devno_;
  _trace("dev[%d][%s] func[%p] params[%p]", devno_, name_, func, params);
  func(params, &dev);
}

void Device::ExecuteCustom(Command* cmd) {
  int tag = cmd->tag();
  char* params = cmd->params();
  Custom(tag, params);
}

Kernel* Device::ExecuteSelectorKernel(Command* cmd) {
  Kernel* kernel = cmd->kernel();
  if (!cmd->selector_kernel()) return kernel;
  iris_selector_kernel func = cmd->selector_kernel();
  void* params = cmd->selector_kernel_params();
  char kernel_name[256];
  memset(kernel_name, 0, 256);
  strcpy(kernel_name, kernel->name());
  func(cmd->task()->struct_obj(), params, kernel_name);
  return Platform::GetPlatform()->GetKernel(kernel_name);
}

int Device::RegisterCommand(int tag, command_handler handler) {
  cmd_handlers_[tag] = handler;
  return IRIS_SUCCESS;
}

int Device::RegisterHooks() {
  hook_task_pre_ = Platform::GetPlatform()->hook_task_pre();
  hook_task_post_ = Platform::GetPlatform()->hook_task_post();
  hook_command_pre_ = Platform::GetPlatform()->hook_command_pre();
  hook_command_post_ = Platform::GetPlatform()->hook_command_post();
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


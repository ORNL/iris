#include "Device.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"
#include "Worker.h"

namespace iris {
namespace rt {

Device::Device(int devno, int platform) {
  devno_ = devno;
  current_queue_ = 0;
  platform_ = platform;
  busy_ = false;
  enable_ = false;
  async_ = false;
  native_kernel_not_exists_ = false;
  is_d2d_possible_ = false;
  shared_memory_buffers_ = false;
  can_share_host_memory_ = false;
  nqueues_ = IRIS_MAX_DEVICE_NQUEUES;
  dev_2_child_task_ = NULL; 
  memset(vendor_, 0, sizeof(vendor_));
  memset(name_, 0, sizeof(name_));
  memset(version_, 0, sizeof(version_));
  memset(kernel_path_, 0, sizeof(kernel_path_));
  timer_ = new Timer();
  hook_task_pre_ = NULL;
  hook_task_post_ = NULL;
  hook_command_pre_ = NULL;
  hook_command_post_ = NULL;
  worker_ = NULL;
  stream_policy_ = STREAM_POLICY_DEFAULT;
}

Device::~Device() {
  delete timer_;
}

int Device::GetStream(Task *task) { 
    task->stream_lock();
    int s_index = task->recommended_stream();
    if (s_index == -1) {
        int queue = get_new_stream_queue();
        task->set_recommended_stream(queue);
    }
    task->stream_unlock();
    return task->recommended_stream();
    //return task->uid() % nqueues_; 
}

int Device::GetStream(Task *task, BaseMem *mem) { 
    task->stream_lock();
    int s_index = task->recommended_stream();
    if (s_index == -1) {
        int queue = get_new_stream_queue();
        task->set_recommended_stream(queue);
    }
    task->stream_unlock();
    return task->recommended_stream();
    //return task->uid() % nqueues_; 
}

void Device::Execute(Task* task) {
  busy_ = true;
  if (is_async() && task->user()) task->set_recommended_stream(GetStream(task));
  if (hook_task_pre_) hook_task_pre_(task);
  TaskPre(task);
  task->set_time_start(timer_->Now());
  //printf("================== Task:%s =====================\n", task->name());
  _printf("task[%lu:%s] started execution on dev[%d][%s] time[%lf] start:[%lf] q[%d]", task->uid(), task->name(), devno(), name(), task->time(), task->time_start(), task->recommended_stream());
  _trace("task[%lu:%s] started execution on dev[%d][%s] time[%lf] start:[%lf] q[%d]", task->uid(), task->name(), devno(), name(), task->time(), task->time_start(), task->recommended_stream());
  for(Command *cmd : task->reset_mems()) {
      ExecuteMemResetInput(task, cmd);
  }
  if (task->cmd_kernel()) ExecuteMemIn(task, task->cmd_kernel());     
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (hook_command_pre_) hook_command_pre_(cmd);
    cmd->set_time_start(timer_->Now());
    switch (cmd->type()) {
      case IRIS_CMD_INIT:         ExecuteInit(cmd);       break;
      case IRIS_CMD_KERNEL:       {
                                      ExecuteKernel(cmd);     
                                      ExecuteMemOut(task, task->cmd_kernel());
                                      break;
                                  }
      case IRIS_CMD_MALLOC:       ExecuteMalloc(cmd);     break;
      case IRIS_CMD_H2D:          ExecuteH2D(cmd);        break;
      case IRIS_CMD_H2BROADCAST:  ExecuteH2BroadCast(cmd); break;
      case IRIS_CMD_H2DNP:        ExecuteH2DNP(cmd);      break;
      case IRIS_CMD_D2H:          ExecuteD2H(cmd);        break;
      case IRIS_CMD_MEM_FLUSH:    ExecuteMemFlushOut(cmd);break;
      case IRIS_CMD_RESET_INPUT :                         break;
      case IRIS_CMD_MAP:          ExecuteMap(cmd);        break;
      case IRIS_CMD_RELEASE_MEM:  ExecuteReleaseMem(cmd); break;
      case IRIS_CMD_HOST:         ExecuteHost(cmd);       break;
      case IRIS_CMD_CUSTOM:       ExecuteCustom(cmd);     break;
      default: {_error("cmd type[0x%x]", cmd->type());  printf("TODO: determine why name (%s) is set, but type isn't\n",cmd->type_name());};
    }
    cmd->set_time_end(timer_->Now());
    if (hook_command_post_) hook_command_post_(cmd);
  }
  if (is_async() && task->user()) AddCallback(task);
  task->set_time_end(timer_->Now());
  TaskPost(task);
  if (hook_task_post_) hook_task_post_(task);
//  if (++stream_index >= nqueues_) stream_index = 0;
  if (!task->system()) _trace("task[%lu:%s] complete dev[%d][%s] time[%lf] end:[%lf]", task->uid(), task->name(), devno(), name(), task->time(), task->time_end());
  if (task->cmd_kernel() != NULL) ProactiveTransfers(task, task->cmd_kernel());
  if (!is_async() || !task->user()) task->Complete();
  busy_ = false;
}

int Device::AddCallback(Task* task) {
  int stream_index = 0;
  stream_index = GetStream(task); //task->uid() % nqueues_; 
  return RegisterCallback(stream_index, (CallBackType)Device::Callback, task, iris_stream_non_blocking);
}

void Device::Callback(void *stream, int status, void* data) {
  Task* task = (Task*) data;
  _printf(" stream_ptr:%p task:%p:%s:%lu status:%d", stream, task, task->name(), task->uid(), status);
  task->Complete();
  _printf(" Completed task stream_ptr:%p task:%p:%s:%lu status:%d", stream, task, task->name(), task->uid(), status);
}

void Device::ProactiveTransfers(Task *task, Command *cmd)
{
  if (!Platform::GetPlatform()->get_enable_proactive()) return;
  if (cmd->kernel() == NULL) return;
  _printf("Doing proactive transfers for task:%s:%lu", task->name(), task->uid());
  int ndevs = Platform::GetPlatform()->ndevs();
  if (dev_2_child_task_ == NULL)
      dev_2_child_task_ = (Task**)malloc(ndevs*sizeof(Task*));
  map<BaseMem *, int> in_mems = cmd->kernel()->in_mems();
  // Extract all input memory objects and unique device transfers
  for (auto & inmem : in_mems) {
      DataMem *mem = (DataMem *)inmem.first;
      int karg_index = inmem.second;
      memset(dev_2_child_task_, 0x0, sizeof(Task*)*ndevs);
      for (int i=0; i<task->nchilds(); i++) {
          Task *child = task->Child(i);
          int dev = child->recommended_dev();
          KernelArg *karg = NULL;
          if (dev != -1 && child->cmd_kernel() != NULL && 
                  (karg = cmd->kernel()->get_in_mem_karg(mem))!=NULL) {
              dev_2_child_task_[dev] = child;
              karg->proactive_enabled();
          }
      }
      //TODO: Explore Hierarchical (Tree style) data transfers in future for asyn
      for (int i=0; i<ndevs; i++) {
          if (dev_2_child_task_[i] != NULL) {
              Device *dev = Platform::GetPlatform()->device(i);
              if (mem->GetMemHandlerType() == IRIS_DMEM) {
                 dev->ExecuteMemInDMemIn(dev_2_child_task_[i], cmd, mem);
              }
              else if (mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
                 dev->ExecuteMemInDMemRegionIn(dev_2_child_task_[i], cmd, (DataMemRegion *)mem);
              }
          }
      }
  }
}

void Device::ExecuteInit(Command* cmd) {
  timer_->Start(IRIS_TIMER_INIT);
  cmd->set_time_start(timer_->Now());
  native_kernel_not_exists_ = false;
  if (SupportJIT()) {
    char* tmpdir = NULL;
    char* src = NULL;
    char* bin = NULL;
    Platform::GetPlatform()->EnvironmentGet("TMPDIR", &tmpdir, NULL);
    Platform::GetPlatform()->GetFilePath(kernel_src(), &src, NULL);
    Platform::GetPlatform()->GetFilePath(kernel_bin(), &bin, NULL);
    bool stat_src = Utils::Exist(src);
    bool stat_bin = Utils::Exist(bin);
    errid_ = IRIS_SUCCESS;
    if (!stat_src && !stat_bin) {
      _error("NO KERNEL SRC[%s] NO KERNEL BIN[%s]", src, bin);
      native_kernel_not_exists_ = true;
    } else if (!stat_src && stat_bin) {
      strncpy(kernel_path_, bin, strlen(bin));
    } else if (stat_src && !stat_bin) {
      Platform::GetPlatform()->EnvironmentGet(kernel_bin(), &bin, NULL);
      sprintf(kernel_path_, "%s/%s-%d", tmpdir, bin, devno_);
      errid_ = Compile(src);
    } else {
      long mtime_src = Utils::Mtime(src);
      long mtime_bin = Utils::Mtime(bin);
      if (mtime_src > mtime_bin) {
        Platform::GetPlatform()->EnvironmentGet(kernel_bin(), &bin, NULL);
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
  if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddKernel(cmd);
  enable_ = true;
}

void Device::ExecuteKernel(Command* cmd) {
  timer_->Start(IRIS_TIMER_KERNEL);
  cmd->set_time_start(timer_->Now());
  Kernel* kernel = ExecuteSelectorKernel(cmd);
  int dim = cmd->dim();
  size_t* off = cmd->off();
  size_t* gws = cmd->gws();
  //size_t gws0 = gws[0];
  size_t* lws = cmd->lws();
  //bool reduction = false;
  iris_poly_mem* polymems = cmd->polymems();
  int npolymems = cmd->npolymems();
  int max_idx = 0;
  int mem_idx = 0;
  //double ltime_start = timer_->GetCurrentTime();
  if (!kernel->vendor_specific_kernel_check_flag(devno_))
      CheckVendorSpecificKernel(kernel);
  KernelLaunchInit(kernel);
  //double ltime = timer_->GetCurrentTime() - ltime_start;
  KernelArg* args = cmd->kernel_args();
  int *params_map = cmd->get_params_map();
  int arg_idx = 0;
  //double atime_start = timer_->GetCurrentTime();
  //double set_mem_time = 0.0f;
  for (int idx = 0; idx < cmd->kernel_nargs(); idx++) {
    if (idx > max_idx) max_idx = idx;
    KernelArg* arg = args + idx;
    BaseMem* bmem = (Mem*)arg->mem;
    if (params_map != NULL && 
        (params_map[idx] & iris_all) == 0 && 
        !(params_map[idx] & type_) ) continue;
    if (bmem && bmem->GetMemHandlerType() == IRIS_MEM) {
      Mem *mem = (Mem *)bmem;
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
        KernelSetMem(kernel, arg_idx, idx, mem, arg->off);
        KernelSetArg(kernel, arg_idx+1, idx + 1, lws[0] * mem->type_size(), NULL);
        //reduction = true;
        if (idx + 1 > max_idx) max_idx = idx + 1;
        idx++;
        arg_idx+=2;
      } else { KernelSetMem(kernel, arg_idx, idx, mem, arg->off); arg_idx+=1; }
      mem_idx++;
    } else if (bmem) {
        //double set_mem_time_start = timer_->GetCurrentTime();
        KernelSetMem(kernel, arg_idx, idx, (DataMem *)bmem, arg->off); arg_idx+=1; 
        //set_mem_time += timer_->GetCurrentTime() - set_mem_time_start;
        mem_idx++;
    } else { KernelSetArg(kernel, arg_idx, idx, arg->size, arg->value); arg_idx+=1; }
  }
  //double atime = timer_->GetCurrentTime() - atime_start;
#if 0
  if (reduction) {
    _trace("max_idx+1[%d] gws[%lu]", max_idx + 1, gws0);
    KernelSetArg(kernel, max_idx + 1, sizeof(size_t), &gws0);
  }
#endif
  //double ktime_start = timer_->GetCurrentTime();
  bool enabled = true;
  if (cmd->task() != NULL && (
              cmd->task()->is_kernel_launch_disabled() ||
              Platform::GetPlatform()->is_kernel_launch_disabled()))
      enabled = false;
  if (enabled) {
      _printf("Launching kernel:%s:%lu task:%s:%lu", kernel->name(), kernel->uid(), cmd->task()->name(), cmd->task()->uid());
      errid_ = KernelLaunch(kernel, dim, off, gws, lws[0] > 0 ? lws : NULL);
  }
  //double ktime = timer_->GetCurrentTime() - ktime_start;
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_KERNEL);
  cmd->SetTime(time);
  if (is_async()) 
      RecordEvent(cmd->kernel()->GetCompletionEventPtr(), cmd->task()->recommended_stream());
  //printf("Task:%s time:%f ktime:%f init:%f atime:%f setmemtime:%f\n", cmd->task()->name(), time, ktime, ltime, atime, set_mem_time);
  cmd->kernel()->history()->AddKernel(cmd, this, time);
  if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddKernel(cmd);
}

void Device::ExecuteMalloc(Command* cmd) {
  cmd->set_time_start(timer_->Now());
  Mem* mem = cmd->mem();
  void* arch = mem->arch(this);
  cmd->set_time_end(timer_->Now());
  if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddH2D(cmd);
  _trace("dev[%d] malloc[%p]", devno_, arch);
}

void Device::GetPossibleDevices(int devno, int *nddevs, int &d2d_dev, int &cpu_dev, int &non_cpu_dev)
{
    d2d_dev = -1;
    cpu_dev = -1;
    non_cpu_dev = -1;
    for(int i=0; nddevs[i] != -1; i++) {
        Device *target_dev = Platform::GetPlatform()->device(nddevs[i]);
        if (d2d_dev == -1 && type() == target_dev->type() &&
                isD2DEnabled() && target_dev->isD2DEnabled()) {
            d2d_dev = nddevs[i];
        }
        if (cpu_dev == -1 && type() != iris_cpu && target_dev->type() == iris_cpu) {
            cpu_dev = nddevs[i];
        }
        else if (non_cpu_dev == -1 && type() == iris_cpu && target_dev->type() != iris_cpu) {
            non_cpu_dev = nddevs[i];
        }
    }   
}

void Device::ExecuteMemResetInput(Task *task, Command* cmd) {
    BaseMem* bmem = (BaseMem *)cmd->mem();
    if (bmem->GetMemHandlerType() != IRIS_DMEM &&
        bmem->GetMemHandlerType() != IRIS_DMEM_REGION) {
        _error("Reset input is called for unsupported memory handler task:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
        return;
    }
    if (bmem->GetMemHandlerType() == IRIS_DMEM) {
    DataMem* mem = (DataMem *)cmd->mem();
    mem->dev_lock(devno_);
    ResetMemory(mem, cmd->reset_value());
    mem->set_host_dirty();
    mem->set_dirty_except(devno_);
    mem->dev_unlock(devno_);
    }
    else if (bmem->GetMemHandlerType() == IRIS_DMEM_REGION) {
    DataMemRegion* mem = (DataMemRegion *)cmd->mem();
    mem->dev_lock(devno_);
    ResetMemory(mem, cmd->reset_value());
    mem->set_host_dirty();
    mem->set_dirty_except(devno_);
    mem->dev_unlock(devno_);
    }
}

void Device::ExecuteMemIn(Task *task, Command* cmd) {
    if (cmd == NULL || cmd->kernel() == NULL) return;
    int *params_map = cmd->get_params_map();
    //int nargs = cmd->kernel_nargs();
    Kernel *kernel = cmd->kernel();
    vector<int> data_mems_in_order = kernel->data_mems_in_order();
    vector<BaseMem *> & all_data_mems_in = kernel->all_data_mems_in();
    if (kernel->is_profile_data_transfers()) {
        kernel->ClearMemInProfile();
    }
    if (kernel->data_mems_in_order().size() > 0) {
        for(int idx : kernel->data_mems_in_order()) {
            if (params_map != NULL && 
                    (params_map[idx] & iris_all) == 0 && 
                    !(params_map[idx] & type_) ) continue;
            KernelArg *karg = kernel->karg(idx);
            // If the kernel argument data transfer is proactively handled, ignore here
            if (karg->proactive) { 
                // Clear it for next time task submit
                // TODO: Revisit this after asynchronous implementation
                karg->proactive_disabled();
                continue;
            }
            if ((size_t)idx < all_data_mems_in.size()) {
                if (all_data_mems_in[idx]->GetMemHandlerType() == IRIS_DMEM) {
                    DataMem *mem = (DataMem*)all_data_mems_in[idx];
                    ExecuteMemInDMemIn(task, cmd, mem);
                }
                else if (all_data_mems_in[idx]->GetMemHandlerType() == IRIS_DMEM_REGION) {
                    DataMemRegion *mem = (DataMemRegion*)all_data_mems_in[idx];
                    ExecuteMemInDMemRegionIn(task, cmd, mem);
                }
                else {
                    _error("Couldn't find idx:%d in data_mems_in_ or data_mem_regions_in_ for task:%s:%ld", idx, task->name(), task->uid());
                }
            }
            else {
                _error("Couldn't find idx:%d<size:%ld in data_mems_in_ or data_mem_regions_in_ for task:%s:%ld", idx, all_data_mems_in.size(), task->name(), task->uid());
            }
        }
    }
    else {
        for(pair<int, DataMem *> it : kernel->data_mems_in()) {
            int idx = it.first;
            DataMem *mem = it.second;
            if (params_map != NULL && 
                    (params_map[idx] & iris_all) == 0 && 
                    !(params_map[idx] & type_) ) continue;
            ExecuteMemInDMemIn(task, cmd, mem);
        }
        for(pair<int, DataMemRegion *> it : kernel->data_mem_regions_in()) {
            int idx = it.first;
            DataMemRegion *mem = it.second;
            if (params_map != NULL && 
                    (params_map[idx] & iris_all) == 0 && 
                    !(params_map[idx] & type_) ) continue;
            ExecuteMemInDMemRegionIn(task, cmd, mem);
        }
    }
}

template <typename DMemType>
void Device::WaitForDataAvailability(int devno, Task *task, DMemType *mem)
{
    if (is_async()) {
        int stream = mem->GetWriteStream(devno);
        if (stream != task->recommended_stream()) {
            Device *dev= Platform::GetPlatform()->device(devno);
            // Even if the parent task and current task are running on same device, it may be using different streams
            for (void * event: mem->GetWaitEvents(devno_)) {
                _printf(" from dev:[%d] event to dev:[%d][%s] task:%s:%lu task_stream:%d mem:%lu mem_stream:%d\n", devno_, devno, dev->name(), task->name(), task->uid(), task->recommended_stream(), mem->uid(), stream);
                WaitForEvent(event, stream, iris_event_disable_timing);
            }
        }
    }
}
template <typename DMemType>
void Device::InvokeDMemInDataTransfer(Task *task, Command *cmd, DMemType *mem)
{
    int nddevs[IRIS_MAX_NDEVS+1];
    Kernel *kernel = cmd->kernel();
    size_t *ptr_off = mem->local_off();
    size_t *gws = mem->host_size();
    size_t *lws = mem->dev_size();
    size_t elem_size = mem->elem_size();
    int dim = mem->dim();
    size_t size = mem->size();
    mem->dev_lock(devno_);
    int cpu_dev = -1;
    int non_cpu_dev = -1;
    int d2d_dev = -1;
    GetPossibleDevices(devno_, mem->get_non_dirty_devices(nddevs), 
            d2d_dev, cpu_dev, non_cpu_dev);
    bool h2d_enabled = false;
    bool d2d_enabled = false;
    bool d2o_enabled = false;
    bool o2d_enabled = false;
    bool d2h_h2d_enabled = false;
    double h2dtime = 0.0f;
    double d2htime = 0.0f;
    double d2dtime = 0.0f;
    double d2otime = 0.0f;
    double o2dtime = 0.0f;
    // Check if it is still dirty
    if (!Platform::GetPlatform()->is_d2d_disabled() && d2d_dev >= 0) { 
        // May be transfer directly from peer device is best 
        // Do D2D communication
        // Keep host data dirty as it is
        _printf("explore D2D dev[%d][%s] task[%ld:%s] mem[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), task->recommended_stream());
        _trace("explore D2D dev[%d][%s] task[%ld:%s] mem[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), task->recommended_stream());
        Device *src_dev = Platform::GetPlatform()->device(d2d_dev);
        void* dst_arch = mem->arch(this);
        void* src_arch = mem->arch(src_dev);
        double start = timer_->Now();

        WaitForDataAvailability(src_dev->devno(), task, mem);
        MemD2D(task, mem, dst_arch, src_arch, mem->size());
        double end = timer_->Now();
        if (is_async()) 
            mem->RecordEvent(devno(), task->recommended_stream());
        if (kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2d, (uint32_t) d2d_dev, (uint32_t) devno_, start, end});
        }
        d2dtime = end - start;
        d2d_enabled = true;
    }
    else if (!Platform::GetPlatform()->is_d2d_disabled() && cpu_dev >= 0) {
        // You didn't find data in peer device, 
        // but you found it in neighbouring CPU (OpenMP) device.
        // Fetch it through H2D
        _trace("explore Device(OpenMP)2Device (O2D) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        Device *src_dev = Platform::GetPlatform()->device(cpu_dev);
        void* src_arch = mem->arch(src_dev);
        size_t off[3] = { 0 };
        size_t host_sizes[3] = { mem->size() };
        size_t dev_sizes[3] = { mem->size() };
        // Though we use H2D command for transfer, 
        // it is still a device to device transfer
        // You do not need offsets as they correspond to host pointer
        double start = timer_->Now();
        WaitForDataAvailability(src_dev->devno(), task, mem);
        MemH2D(task, mem, off, host_sizes, dev_sizes, 1, 1, mem->size(), src_arch, "OpenMP2DEV ");
        double end = timer_->Now();
        if (is_async()) 
            mem->RecordEvent(devno(), task->recommended_stream());
        o2dtime = end - start;
        if (kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_o2d, (uint32_t) cpu_dev, (uint32_t) devno_, start, end});
        }
        o2d_enabled = true;
    }
    else if (!Platform::GetPlatform()->is_d2d_disabled() && this->type() == iris_cpu && non_cpu_dev >= 0) {
        // You found data in non-CPU/OpenMP device, but this device is CPU/OpenMP
        // Use target D2H transfer 
        //_trace("explore Device2Device(OpenMP) (D2O) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        Device *src_dev = Platform::GetPlatform()->device(non_cpu_dev);
        void* src_arch = mem->arch(this);
        size_t off[3] = { 0 };
        size_t host_sizes[3] = { mem->size() };
        size_t dev_sizes[3] = { mem->size() };
        // Though we use H2D command for transfer, 
        // it is still a device to device transfer
        // You do not need offsets as they correspond to host pointer
        bool context_shift = src_dev->IsContextChangeRequired();
        _trace("explore Device2Device(OpenMP) (D2O) dev[%d][%s] task[%ld:%s] mem[%lu] cs:%d", devno_, name_, task->uid(), task->name(), mem->uid(), context_shift);
        if (src_dev->is_async()) {
            src_dev->EventSychronize(cmd->kernel()->GetCompletionEvent());
        }
        double start = timer_->Now();
        src_dev->MemD2H(task, mem, off, host_sizes, dev_sizes, 1, 1, mem->size(), src_arch, "DEV2OpenMP ");
        double end = timer_->Now();
        if (is_async()) 
            mem->RecordEvent(src_dev->devno(), mem->GetWriteStream(src_dev->devno()));
        if (context_shift) ResetContext();
        if (kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2o, (uint32_t) non_cpu_dev, (uint32_t) devno_, start, end});
        }
        d2otime = end - start;
        d2o_enabled = true;
    }
    else if (!mem->is_host_dirty()) {
        // If host is not dirty, it is best to transfer from host
        // None of the devices having valid copy or D2D is not possible
        _trace("explore Host2Device (H2D) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        void* host = mem->host_memory(); // It should work even if host_ptr is null
        double start = timer_->Now();
        errid_ = MemH2D(task, mem, ptr_off, 
                gws, lws, elem_size, dim, size, host);
        double end = timer_->Now();
        _printf("explore Host2Device (H2D) dev[%d][%s] task[%ld:%s] mem[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), task->recommended_stream());
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        if (is_async()) 
            mem->RecordEvent(devno(), task->recommended_stream());
        if (kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_h2d, (uint32_t) -1, (uint32_t) devno_, start, end});
        }
        h2dtime = end - start;
        h2d_enabled = true;
    }
    else {
        // Host doesn't have fresh copy and peer2peer d2d is not possible
        // Fresh copy should be in some other device memory
        // do D2H and follewed by H2D
        //_trace("explore D2H->H2D dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        void* host = mem->host_memory(); // It should work even if host_ptr is null
        Device *src_dev = Platform::GetPlatform()->device(nddevs[0]);
        // D2H should be issued from target src (remote) device
        bool context_shift = src_dev->IsContextChangeRequired();
        _trace("explore D2H->H2D dev[%d][%s] task[%ld:%s] mem[%lu] cs:%d", devno_, name_, task->uid(), task->name(), mem->uid(), context_shift);
        double start = timer_->Now();
        double d2h_start = start;
        errid_ = src_dev->MemD2H(task, mem, ptr_off, 
                gws, lws, elem_size, dim, size, host, "D2H->H2D(1) ");
        double end = timer_->Now();
        if (src_dev->is_async()) 
            mem->RecordEvent(src_dev->devno(), mem->GetWriteStream(src_dev->devno()));
        d2htime = end - start;
        if (context_shift) ResetContext();
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        // H2D should be issued from this current device
        if (is_async() && src_dev->is_async()) { 
            // Assumption: No two tasks mapped to same device have D2H-H2D policies for same memory object. Because, the asynchronous transfers skip the transfer for second task. 
            src_dev->RegisterCallback(mem->GetWriteStream(src_dev->devno()), 
                    (CallBackType)EventExchange::Fire, mem->GetEventExchange(devno()), 
                    iris_stream_non_blocking);
            RegisterCallback(task->recommended_stream(), 
                    EventExchange::Wait, mem->GetEventExchange(devno()),
                    iris_stream_default);
        }
        else if (src_dev->is_async()) {
            src_dev->EventSychronize(cmd->kernel()->GetCompletionEvent());
        }
        else if (is_async()) {
            // Do nothing
        }
        start = timer_->Now();
        errid_ = MemH2D(task, mem, ptr_off, 
                gws, lws, elem_size, dim, size, host, "D2H->H2D(2) ");
        end = timer_->Now();
        if (is_async()) 
            mem->RecordEvent(devno(), task->recommended_stream());
        h2dtime = end - start;
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        mem->clear_host_dirty();
        d2h_h2d_enabled = true;
        if (kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2h_h2d, (uint32_t) nddevs[0], (uint32_t) devno_, d2h_start, end});
        }
    }
    mem->clear_dev_dirty(devno_ );
    mem->dev_unlock(devno_);
    if (h2d_enabled) cmd->kernel()->history()->AddH2D(cmd, this, h2dtime, size);
    if (d2d_enabled) cmd->kernel()->history()->AddD2D(cmd, this, d2dtime, size);
    if (d2o_enabled) cmd->kernel()->history()->AddD2O(cmd, this, d2otime, size);
    if (o2d_enabled) cmd->kernel()->history()->AddO2D(cmd, this, o2dtime, size);
    if (d2h_h2d_enabled) {
        cmd->kernel()->history()->AddD2H_H2D(cmd, this, d2htime+h2dtime, size);
        //cmd->kernel()->history()->AddD2H(cmd, this, d2htime);
        //cmd->kernel()->history()->AddH2D(cmd, this, h2dtime);
    }
}
void Device::ExecuteMemInDMemRegionIn(Task *task, Command* cmd, DataMemRegion *mem) {
    if (mem->is_dev_dirty(devno_)) {
        _trace("Initiating DMEM_REGION data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid(), mem->get_dmem()->uid());
        InvokeDMemInDataTransfer<DataMemRegion>(task, cmd, mem);
    }
    else{
        _trace("Skipped DMEM_REGION data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu] dptr[%p]", devno_, name_, task->uid(), task->name(), mem->uid(), mem->get_dmem()->uid(), mem->arch(devno_));
        WaitForDataAvailability<DataMemRegion>(devno_, task, mem);
    }
}
void Device::ExecuteMemInDMemIn(Task *task, Command* cmd, DataMem *mem) {
    if (mem->is_regions_enabled()) {
        int n_regions = mem->get_n_regions();
        for (int i=0; i<n_regions; i++) {
            DataMemRegion *rmem = (DataMemRegion *)mem->get_region(i);
            if (rmem->is_dev_dirty(devno_)) {
                _trace("Initiating DMEM_REGION region(%d) data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu]", i, devno_, name_, task->uid(), task->name(), rmem->uid(), rmem->get_dmem()->uid());
                InvokeDMemInDataTransfer<DataMemRegion>(task, cmd, rmem);
            }
            else {
                _trace("Skipped DMEM_REGION region(%d) H2D data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu] dptr[%p]", i, devno_, name_, task->uid(), task->name(), rmem->uid(), rmem->get_dmem()->uid(), rmem->arch(devno_));
                WaitForDataAvailability<DataMemRegion>(devno_, task, rmem);
            }
        }
#if 0
        size_t *off = mem->off();
        size_t *host_sizes = mem->host_size();
        size_t *dev_sizes = mem->dev_size();
        size_t elem_size = mem->elem_size();
        int dim = mem->dim();
        size_t size = mem->size();
        float *host = (float *) malloc(size);
        errid_ = MemD2H(task, mem, off, 
                host_sizes, dev_sizes, elem_size, dim, size, host);
        printf("Regions Input: %ld:%s ", task->uid(), task->name());
        for(int i=0; i<dev_sizes[1]; i++) {
            int ai = off[1] + i;
            for(int j=0; j<dev_sizes[0]; j++) {
                int aj = off[0] + j;
                printf("%10.1lf ", host[ai*host_sizes[1]+aj]);
            }
        }
        printf("\n");
#endif
    }
    else if (mem->is_dev_dirty(devno_)) {
        InvokeDMemInDataTransfer<DataMem>(task, cmd, mem);
    }
    else{
        _trace("Skipped DMEM H2D data transfer dev[%d:%s] task[%ld:%s] dmem[%lu] dptr[%p]", devno_, name_, task->uid(), task->name(), mem->uid(), mem->arch(devno_));
        WaitForDataAvailability<DataMem>(devno_, task, mem);
    }
}
void Device::ExecuteMemOut(Task *task, Command* cmd) {
    if (cmd == NULL || cmd->kernel() == NULL) return;
    int *params_map = cmd->get_params_map();
    //int nargs = cmd->kernel_nargs();
    for(pair<int, DataMem *> it : cmd->kernel()->data_mems_out()) {
        int idx = it.first;
        DataMem *mem = it.second;
        if (params_map != NULL && 
                (params_map[idx] & iris_all) == 0 && 
                !(params_map[idx] & type_) ) continue;
        mem->set_dirty_except(devno_);
        mem->set_host_dirty();
        mem->ClearWaitEvents(devno());
        mem->SetWriteStream(devno(), task->recommended_stream());
        mem->AddWaitEvent(devno(), cmd->kernel()->GetCompletionEvent());
    }
    for(pair<int, DataMemRegion *> it : cmd->kernel()->data_mem_regions_out()) {
        int idx = it.first;
        DataMemRegion *mem = it.second;
        if (params_map != NULL && 
                (params_map[idx] & iris_all) == 0 && 
                !(params_map[idx] & type_) ) continue;
        mem->set_dirty_except(devno_);
        mem->set_host_dirty();
        mem->ClearWaitEvents(devno());
        mem->SetWriteStream(devno(), task->recommended_stream());
        mem->AddWaitEvent(devno(), cmd->kernel()->GetCompletionEvent());
    }
}
void Device::ExecuteMemFlushOut(Command* cmd) {
    int nddevs[IRIS_MAX_NDEVS+1];
    BaseMem* bmem = (BaseMem *)cmd->mem();
    if (bmem->GetMemHandlerType() != IRIS_DMEM) {
        _error("Flush out is called for unssuported memory handler task:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
        return;
    }
    DataMem* mem = (DataMem *)cmd->mem();
    if (mem->is_host_dirty()) {
        size_t *ptr_off = mem->off();
        size_t *gws = mem->host_size();
        size_t *lws = mem->dev_size();
        size_t elem_size = mem->elem_size();
        int dim = mem->dim();
        size_t size = mem->size();
        void* host = mem->host_memory(); // It should work even if host_ptr is null
        double start = timer_->Now();
        Task *task = cmd->task();
        Device *src_dev = this;
        if (mem->is_dev_dirty(devno_)) {
            int cpu_dev = -1;
            int non_cpu_dev = -1;
            int d2d_dev = -1;
            GetPossibleDevices(devno_, mem->get_non_dirty_devices(nddevs), 
                    d2d_dev, cpu_dev, non_cpu_dev);
            src_dev = Platform::GetPlatform()->device(nddevs[0]);
            // D2H should be issued from target src (remote) device
            bool context_shift = src_dev->IsContextChangeRequired();
            errid_ = src_dev->MemD2H(task, mem, ptr_off, 
                    gws, lws, elem_size, dim, size, host, "MemFlushOut ");
            if (context_shift) ResetContext();
        }
        else {
            errid_ = MemD2H(task, mem, ptr_off, 
                    gws, lws, elem_size, dim, size, host, "MemFlushOut ");
        }
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        mem->clear_host_dirty();
        double end = timer_->Now();
        double d2htime = end - start;
        Command* cmd_kernel = cmd->task()->cmd_kernel();
        if (cmd_kernel) cmd_kernel->kernel()->history()->AddD2H(cmd_kernel, src_dev, d2htime, size);
        else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, d2htime, size);
        if (task->is_profile_data_transfers()) {
            task->AddOutDataObjectProfile({(uint32_t) task->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2h_h2d, (uint32_t) devno_, (uint32_t) -1, start, end});
        }
    }
    else {
        _trace("MemFlushout is skipped as host already having valid data for task:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
    }
}
void Device::ExecuteH2BroadCast(Command *cmd) {
    int ndevs = Platform::GetPlatform()->ndevs();
    for(int i=0; i<ndevs; i++) {
        Device *src_dev = Platform::GetPlatform()->device(i);
        ExecuteH2D(cmd, src_dev);
    }
}
void Device::ExecuteD2D(Command* cmd, Device *dev) {
    if (dev == NULL) dev = this;
    Mem* mem = cmd->mem();
    size_t off = cmd->off(0);
    size_t *ptr_off = cmd->off();
    size_t *gws = cmd->gws();
    size_t *lws = cmd->lws();
    size_t elem_size = cmd->elem_size();
    int dim = cmd->dim();
    size_t size = cmd->size();
    bool exclusive = cmd->exclusive();
    void* host = cmd->host();
    if (exclusive) mem->SetOwner(off, size, this);
    else mem->AddOwner(off, size, this);
    Device *src_dev = Platform::GetPlatform()->device(cmd->src_dev());
    double start = timer_->Now();
    cmd->set_time_start(start);
    if (src_dev->type() == type()) {
        // Invoke D2D
        void* dst_arch = mem->arch(this);
        void* src_arch = mem->arch(src_dev);
        MemD2D(cmd->task(), mem, dst_arch, src_arch, mem->size());
    }
    else {
        void* host = mem->host_inter(); // It should work even if host_ptr is null
        // D2H should be issued from target src (remote) device
        bool context_shift = src_dev->IsContextChangeRequired();
        errid_ = src_dev->MemD2H(cmd->task(), mem, ptr_off, 
                gws, lws, elem_size, dim, size, host, "D2H->H2D(1) ");
        if (context_shift) ResetContext();
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        // H2D should be issued from this current device
        errid_ = MemH2D(cmd->task(), mem, ptr_off, 
                gws, lws, elem_size, dim, size, host, "D2H->H2D(2) ");
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
    }
    double end = timer_->Now();
    cmd->set_time_end(end);
    cmd->SetTime(end-start);
    Kernel *kernel = Platform::GetPlatform()->null_kernel();
    Command *cmd_kernel = cmd->task()->cmd_kernel();
    if (cmd_kernel != NULL) 
        kernel = cmd_kernel->kernel();
    if (src_dev->type() == type()) {
        kernel->history()->AddD2D(cmd, this, end, size);
    }
    else {
        kernel->history()->AddD2H_H2D(cmd, this, end, size);
    }
}
void Device::ExecuteH2D(Command* cmd, Device *dev) {
  if (dev == NULL) dev = this;
  BaseMem* dmem = (BaseMem *)cmd->mem();
  //we're using datamem so there is no need to execute this memory transfer
  if (dmem->GetMemHandlerType() == IRIS_DMEM) return;
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t *ptr_off = cmd->off();
  size_t *gws = cmd->gws();
  size_t *lws = cmd->lws();
  size_t elem_size = cmd->elem_size();
  int dim = cmd->dim();
  size_t size = cmd->size();
  bool exclusive = cmd->exclusive();
  void* host = cmd->host();
  if (exclusive) mem->SetOwner(off, size, this);
  else mem->AddOwner(off, size, this);
  timer_->Start(IRIS_TIMER_H2D);
  cmd->set_time_start(timer_->Now());
  errid_ = dev->MemH2D(cmd->task(), mem, ptr_off, gws, lws, elem_size, dim, size, host);
  if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_H2D);
  cmd->SetTime(time);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) {
      if  (cmd->is_internal_memory_transfer())
          cmd_kernel->kernel()->history()->AddD2H_H2D(cmd, dev, time, size, false);
      else
          cmd_kernel->kernel()->history()->AddH2D(cmd, dev, time, size);
  }
  else {
      Platform::GetPlatform()->null_kernel()->history()->AddH2D(cmd, dev, time, size);
  }
  if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddH2D(cmd);
}

void Device::ExecuteH2DNP(Command* cmd) {
  //Mem* mem = cmd->mem();
  //size_t off = cmd->off(0);
  //size_t size = cmd->size();
//  if (mem->IsOwner(off, size, this)) return;
  return ExecuteH2D(cmd);
}

void Device::ExecuteD2H(Command* cmd) {
  BaseMem* dmem = (BaseMem *)cmd->mem();
  if (dmem && dmem->GetMemHandlerType() == IRIS_DMEM) {
    //we're using datamem so there is no need to execute this memory transfer -- just flush
    ExecuteMemFlushOut(cmd);
    return;
  }
  Mem* mem = cmd->mem();
  //size_t off = cmd->off(0);
  size_t *ptr_off = cmd->off();
  size_t *gws = cmd->gws();
  size_t *lws = cmd->lws();
  size_t elem_size = cmd->elem_size();
  int dim = cmd->dim();
  size_t size = cmd->size();
  void* host = cmd->host();
  int mode = mem->mode();
  int expansion = mem->expansion();
  timer_->Start(IRIS_TIMER_D2H);
  cmd->set_time_start(timer_->Now());
  errid_ = IRIS_SUCCESS;

  if (mode & iris_reduction) {
    errid_ = MemD2H(cmd->task(), mem, ptr_off, gws, lws, elem_size, dim, mem->size() * expansion, mem->host_inter());
    Reduction::GetInstance()->Reduce(mem, host, size);
  } else errid_ = MemD2H(cmd->task(), mem, ptr_off, gws, lws, elem_size, dim, size, host);
  if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  cmd->set_time_end(timer_->Now());
  double time = timer_->Stop(IRIS_TIMER_D2H);
  cmd->SetTime(time);
  Command* cmd_kernel = cmd->task()->cmd_kernel();
  if (cmd_kernel) {
      if (cmd->is_internal_memory_transfer())
          cmd_kernel->kernel()->history()->AddD2H_H2D(cmd, this, time, size);
      else
          cmd_kernel->kernel()->history()->AddD2H(cmd, this, time, size);
  }
  else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, time, size);
  if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddD2H(cmd);
}

void Device::ExecuteMap(Command* cmd) {
  //void* host = cmd->host();
  //size_t size = cmd->size();
}

void Device::ExecuteReleaseMem(Command* cmd) {
  Mem* mem = cmd->mem();
  mem->Release(); 
}

void Device::ExecuteHost(Command* cmd) {
  void* params = cmd->func_params();
  if (params != NULL) {
      iris_host_task func = cmd->func();
      const int dev = devno_;
      _trace("dev[%d][%s] func[%p] params[%p]", devno_, name_, func, params);
      func(params, &dev);
  }
  else {
      // For python interface
      iris_host_python_task func = cmd->py_func();
      const int dev = devno_;
      int64_t params_id = cmd->func_params_id();
      _trace("dev[%d][%s] func[%p] params[%ld]", devno_, name_, func, params_id);
      func(&params_id, &dev);
  }
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
  func(*(cmd->task()->struct_obj()), params, kernel_name);
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
void Device::RecordEvent(void **event, int stream) {
    if (*event == NULL)
        CreateEvent(event, iris_event_disable_timing);
    RecordEvent(*event, stream);
}
int Device::RegisterCallback(int stream, CallBackType callback_fn, void *data, int flags)
{
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    return IRIS_ERROR;
}
void Device::CreateEvent(void **event, int flags) { 
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    //CPUEvent *levent = new CPUEvent();
    //*event = levent;
}
void Device::RecordEvent(void *event, int stream) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    //CPUEvent *levent = (CPUEvent *)event;
    //levent->Record();
}
void Device::WaitForEvent(void *event, int stream, int flags) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    //CPUEvent *levent = (CPUEvent *)event;
    //levent->Wait();
}
void Device::DestroyEvent(void *event) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    //CPUEvent *levent = (CPUEvent *)event;
    //delete levent;
}
void Device::EventSychronize(void *event) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
}

} /* namespace rt */
} /* namespace iris */


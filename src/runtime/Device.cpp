#include <unistd.h>
#include "Device.h"
#include "Debug.h"
#include "Command.h"
#include "History.h"
#include "Kernel.h"
#include "Mem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include "HostInterface.h"
#include "Platform.h"
#include "Reduction.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"
#include "Worker.h"
#include "LoaderDefaultKernel.h"

#define _debug3 _debug2
namespace iris {
namespace rt {

Device::Device(int devno, int platform) {
  devno_ = devno;
  peer_access_ = NULL;
  active_tasks_ = 0;
  root_dev_ = NULL;
  current_queue_ = 0;
  current_copy_queue_ = 0;
  type_ = iris_cpu;
  first_event_cpu_end_time_ = 0.0f;
  first_event_cpu_begin_time_ = 0.0f;
  platform_ = platform;
  platform_obj_ = Platform::GetPlatform();
  busy_ = false;
  enable_ = false;
  async_ = false;
  native_kernel_not_exists_ = false;
  is_d2d_possible_ = false;
  shared_memory_buffers_ = false;
  can_share_host_memory_ = false;
  nqueues_ = platform_obj_->nstreams();
  dev_2_child_task_ = NULL; 
  memset(vendor_, 0, sizeof(vendor_));
  memset(name_, 0, sizeof(name_));
  memset(version_, 0, sizeof(version_));
  kernel_path_ = "";
  timer_ = new Timer();
  hook_task_pre_ = NULL;
  hook_task_post_ = NULL;
  hook_command_pre_ = NULL;
  hook_command_post_ = NULL;
  julia_if_ = NULL;
  worker_ = NULL;
  ld_default_ = NULL;
  stream_policy_ = STREAM_POLICY_DEFAULT;
  //stream_policy_ = STREAM_POLICY_SAME_FOR_TASK;
  //stream_policy_ = STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL;
  n_copy_engines_ = platform_obj_->ncopy_streams();
}

Device::~Device() {
  while(active_tasks_ > 0) {
    sleep(1);
  }
  if (ld_default_ != NULL) delete ld_default_;
  _event_prof_debug("Device:%d deleted\n", devno());
  FreeDestroyEvents();
  if (peer_access_ != NULL) delete [] peer_access_;
  if (julia_if_ != NULL) delete julia_if_;
  delete timer_;
}

void Device::CallMemReset(BaseMem *mem, size_t size, ResetData & reset_data, void *stream)
{
    int elem_type = mem->element_type();
    if (elem_type == iris_unknown) return;
    if (ld_default_ == NULL) return;
    //ResetData & reset_data = mem->reset_data();
    int reset_type = reset_data.reset_type_;
    void *arch = mem->arch(devno());
    if (reset_type == iris_reset_assign) {
#define RESET_SEQ(IT, T, M)  case IT: ld_default_->iris_reset_ ## M(static_cast<T*>(arch), reset_data.value_.M, size, stream); break;
        switch(elem_type) {
            RESET_SEQ(iris_uint8,  uint8_t,  u8);
            RESET_SEQ(iris_uint16, uint16_t, u16);
            RESET_SEQ(iris_uint32, uint32_t, u32);
            RESET_SEQ(iris_uint64, uint64_t, u64);
            RESET_SEQ(iris_int8,   int8_t,   i8);
            RESET_SEQ(iris_int16,  int16_t,  i16);
            RESET_SEQ(iris_int32,  int32_t,  i32);
            RESET_SEQ(iris_int64,  int64_t,  i64);
            RESET_SEQ(iris_float,  float,    f32);
            RESET_SEQ(iris_double, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset assign\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_arith_seq) {
#define ARITH_SEQ(IT, T, M)  case IT: ld_default_->iris_arithmetic_seq_ ## M(static_cast<T*>(arch), reset_data.start_.M, reset_data.step_.M, size, stream); break;
        switch(elem_type) {
            ARITH_SEQ(iris_uint8,  uint8_t,  u8);
            ARITH_SEQ(iris_uint16, uint16_t, u16);
            ARITH_SEQ(iris_uint32, uint32_t, u32);
            ARITH_SEQ(iris_uint64, uint64_t, u64);
            ARITH_SEQ(iris_int8,   int8_t,   i8);
            ARITH_SEQ(iris_int16,  int16_t,  i16);
            ARITH_SEQ(iris_int32,  int32_t,  i32);
            ARITH_SEQ(iris_int64,  int64_t,  i64);
            ARITH_SEQ(iris_float,  float,    f32);
            ARITH_SEQ(iris_double, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset arithmatic sequence\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_geom_seq) {
#define GEOM_SEQ(IT, T, M)  case IT: ld_default_->iris_geometric_seq_ ## M(static_cast<T*>(arch), reset_data.start_.M, reset_data.step_.M, size, stream); break;
        switch(elem_type) {
            GEOM_SEQ(iris_uint8,  uint8_t,  u8);
            GEOM_SEQ(iris_uint16, uint16_t, u16);
            GEOM_SEQ(iris_uint32, uint32_t, u32);
            GEOM_SEQ(iris_uint64, uint64_t, u64);
            GEOM_SEQ(iris_int8,   int8_t,   i8);
            GEOM_SEQ(iris_int16,  int16_t,  i16);
            GEOM_SEQ(iris_int32,  int32_t,  i32);
            GEOM_SEQ(iris_int64,  int64_t,  i64);
            GEOM_SEQ(iris_float,  float,    f32);
            GEOM_SEQ(iris_double, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset geometric sequence\n", elem_type, mem->uid()); break;
        }
    }
#define RANDOM_SEQ(IT, RTYPE, T, M)  case IT: ld_default_->iris_random_ ## RTYPE ## _seq_ ## M(static_cast<T*>(arch), reset_data.seed_, reset_data.p1_.M, reset_data.p2_.M, size, stream); break;
    else if (reset_type == iris_reset_random_uniform_seq) {
        switch(elem_type) {
            RANDOM_SEQ(iris_uint8,  uniform, uint8_t,  u8);
            RANDOM_SEQ(iris_uint16, uniform, uint16_t, u16);
            RANDOM_SEQ(iris_uint32, uniform, uint32_t, u32);
            RANDOM_SEQ(iris_uint64, uniform, uint64_t, u64);
            RANDOM_SEQ(iris_int8,   uniform, int8_t,   i8);
            RANDOM_SEQ(iris_int16,  uniform, int16_t,  i16);
            RANDOM_SEQ(iris_int32,  uniform, int32_t,  i32);
            RANDOM_SEQ(iris_int64,  uniform, int64_t,  i64);
            RANDOM_SEQ(iris_float,  uniform, float,    f32);
            RANDOM_SEQ(iris_double, uniform, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset uniform sequence\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_random_normal_seq) {
        switch(elem_type) {
            RANDOM_SEQ(iris_uint8,  normal, uint8_t,  u8);
            RANDOM_SEQ(iris_uint16, normal, uint16_t, u16);
            RANDOM_SEQ(iris_uint32, normal, uint32_t, u32);
            RANDOM_SEQ(iris_uint64, normal, uint64_t, u64);
            RANDOM_SEQ(iris_int8,   normal, int8_t,   i8);
            RANDOM_SEQ(iris_int16,  normal, int16_t,  i16);
            RANDOM_SEQ(iris_int32,  normal, int32_t,  i32);
            RANDOM_SEQ(iris_int64,  normal, int64_t,  i64);
            RANDOM_SEQ(iris_float,  normal, float,    f32);
            RANDOM_SEQ(iris_double, normal, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset normal sequence\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_random_log_normal_seq) {
        switch(elem_type) {
            RANDOM_SEQ(iris_uint8,  log_normal, uint8_t,  u8);
            RANDOM_SEQ(iris_uint16, log_normal, uint16_t, u16);
            RANDOM_SEQ(iris_uint32, log_normal, uint32_t, u32);
            RANDOM_SEQ(iris_uint64, log_normal, uint64_t, u64);
            RANDOM_SEQ(iris_int8,   log_normal, int8_t,   i8);
            RANDOM_SEQ(iris_int16,  log_normal, int16_t,  i16);
            RANDOM_SEQ(iris_int32,  log_normal, int32_t,  i32);
            RANDOM_SEQ(iris_int64,  log_normal, int64_t,  i64);
            RANDOM_SEQ(iris_float,  log_normal, float,    f32);
            RANDOM_SEQ(iris_double, log_normal, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset log_normal sequence\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_random_uniform_sobol_seq) {
        switch(elem_type) {
            RANDOM_SEQ(iris_uint8,  uniform_sobol, uint8_t,  u8);
            RANDOM_SEQ(iris_uint16, uniform_sobol, uint16_t, u16);
            RANDOM_SEQ(iris_uint32, uniform_sobol, uint32_t, u32);
            RANDOM_SEQ(iris_uint64, uniform_sobol, uint64_t, u64);
            RANDOM_SEQ(iris_int8,   uniform_sobol, int8_t,   i8);
            RANDOM_SEQ(iris_int16,  uniform_sobol, int16_t,  i16);
            RANDOM_SEQ(iris_int32,  uniform_sobol, int32_t,  i32);
            RANDOM_SEQ(iris_int64,  uniform_sobol, int64_t,  i64);
            RANDOM_SEQ(iris_float,  uniform_sobol, float,    f32);
            RANDOM_SEQ(iris_double, uniform_sobol, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset uniform sobol sequence\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_random_normal_sobol_seq) {
        switch(elem_type) {
            RANDOM_SEQ(iris_uint8,  normal_sobol, uint8_t,  u8);
            RANDOM_SEQ(iris_uint16, normal_sobol, uint16_t, u16);
            RANDOM_SEQ(iris_uint32, normal_sobol, uint32_t, u32);
            RANDOM_SEQ(iris_uint64, normal_sobol, uint64_t, u64);
            RANDOM_SEQ(iris_int8,   normal_sobol, int8_t,   i8);
            RANDOM_SEQ(iris_int16,  normal_sobol, int16_t,  i16);
            RANDOM_SEQ(iris_int32,  normal_sobol, int32_t,  i32);
            RANDOM_SEQ(iris_int64,  normal_sobol, int64_t,  i64);
            RANDOM_SEQ(iris_float,  normal_sobol, float,    f32);
            RANDOM_SEQ(iris_double, normal_sobol, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset normal sobol sequence\n", elem_type, mem->uid()); break;
        }
    }
    else if (reset_type == iris_reset_random_log_normal_sobol_seq) {
        switch(elem_type) {
            RANDOM_SEQ(iris_uint8,  log_normal_sobol, uint8_t,  u8);
            RANDOM_SEQ(iris_uint16, log_normal_sobol, uint16_t, u16);
            RANDOM_SEQ(iris_uint32, log_normal_sobol, uint32_t, u32);
            RANDOM_SEQ(iris_uint64, log_normal_sobol, uint64_t, u64);
            RANDOM_SEQ(iris_int8,   log_normal_sobol, int8_t,   i8);
            RANDOM_SEQ(iris_int16,  log_normal_sobol, int16_t,  i16);
            RANDOM_SEQ(iris_int32,  log_normal_sobol, int32_t,  i32);
            RANDOM_SEQ(iris_int64,  log_normal_sobol, int64_t,  i64);
            RANDOM_SEQ(iris_float,  log_normal_sobol, float,    f32);
            RANDOM_SEQ(iris_double, log_normal_sobol, double,   f64);
            default: 
                _error("Invalid element type:%d for mem uid:%lu for reset log normal sobol sequence\n", elem_type, mem->uid()); break;
        }
    }
}

void Device::LoadDefaultKernelLibrary(const char *key, const char *flags)
{
    if (!platform_obj_->is_default_kernels_load()) return;
    char *src = NULL;
    char *iris = NULL;
    char *filename = NULL;
    char *tmpdir = NULL;
    char path[2048];
    char out[1024];
    worker_->platform()->EnvironmentGet("", &iris, NULL, '\0');
    worker_->platform()->EnvironmentGet("INCLUDE_DIR", &src, NULL);
    worker_->platform()->EnvironmentGet(key, &filename, NULL);
    sprintf(path, "%s/%s/%s", iris, src, filename);
    Platform::GetPlatform()->EnvironmentGet("TMPDIR", &tmpdir, NULL);
    sprintf(out, "%s/%s.so", tmpdir, filename);
    int result = Compile(path, out, flags);
    if (result == IRIS_SUCCESS) {
        ld_default_ = new LoaderDefaultKernel(out);
        ld_default_->Load();
    }
    else {
        _warning("Couldn't load default kernel library for dev:(%d, %s) for default kernels in %s\n", devno(), name(), path);
    }
    free(src);
    free(iris);
    free(filename);
    free(tmpdir);
}

void Device::EnableJuliaInterface() {
  julia_if_ = new JuliaHostInterfaceLoader(model());
}
StreamPolicy Device::stream_policy(Task *task) 
{
    StreamPolicy platform_policy = Platform::GetPlatform()->stream_policy();
    StreamPolicy device_policy = stream_policy();
    StreamPolicy task_policy = task->stream_policy();
    StreamPolicy policy = (device_policy != STREAM_POLICY_DEFAULT) ? device_policy : platform_policy;
    policy = (task_policy != STREAM_POLICY_DEFAULT) ? task_policy : policy;
    return policy; 
}
int Device::GetStream(Task *task) { 
    task->stream_lock();
    int s_index = task->recommended_stream();
    if (s_index == -1) {
        int stream;
        StreamPolicy policy = stream_policy(task);
        if (policy == STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL)
            stream = DEFAULT_STREAM_INDEX;
        else if (policy == STREAM_POLICY_SAME_FOR_TASK)
            stream = get_new_stream_queue();
        else
            stream = get_new_stream_queue(n_copy_engines_);
            //stream = n_copy_engines_ + 1;
        task->set_recommended_stream(stream);
    }
    task->stream_unlock();
    return task->recommended_stream();
    //return task->uid() % nqueues_; 
}

int Device::GetStream(Task *task, BaseMem *mem, bool new_stream) { 
    StreamPolicy policy = stream_policy(task);
    if (policy == STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL)
        return DEFAULT_STREAM_INDEX;
    if (policy == STREAM_POLICY_SAME_FOR_TASK) {
        int stream = GetStream(task);
        if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
        mem->set_recommended_stream(devno(), stream);
        return stream;
    }
    int stream = mem->recommended_stream(devno());
#if 1
    if (new_stream || stream == -1) {
        stream = get_new_copy_stream_queue();
        if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
        mem->set_recommended_stream(devno(), stream);
    }
#else
    if (task->cmd_kernel() != NULL &&
            task->cmd_kernel()->kernel() != NULL) {
        int arg_index = task->cmd_kernel()->kernel()->get_mem_karg_index(mem);
        KernelArg *arg = task->cmd_kernel()->kernel_arg(arg_index);
        //ASSERT(arg != NULL && "Kernel argument shouldn't be null");
        //_debug2("Task:%s:%lu mem:%lu task_stream:%d mem_index:%d mem_stream:%d", task->name(), task->uid(), mem->uid(), stream, arg->mem_index, (arg->mem_index % n_copy_engines_)+1);
        stream = (arg->mem_index % n_copy_engines_)+1;
    }
#endif
    return stream;
}

void Device::Execute(Task* task) {
  //TODO: Clear the proile event created here
  _event_debug("Inside device execute and calling free destroy events %lu:%s %lf\n", task->uid(), task->name(), timer_->Now());
  _debug("Inside device execute and calling free destroy events %lu:%s\n", task->uid(), task->name());
  FreeDestroyEvents();
  ReserveActiveTask();
  double execute_start = (timer_->Now()-first_event_cpu_mid_point_time())*1000.0;
  busy_ = true;
  _event_prof_debug("Execute task:%lu:%s dev:%d:%s\n", task->uid(), task->name(), devno(), name());
  if (is_async(task) && task->user()) task->set_recommended_stream(GetStream(task));
  if (hook_task_pre_) hook_task_pre_(task);
  TaskPre(task);
  task->set_time_start(timer_);
  //printf("================== Task:%s =====================\n", task->name());
  _event_debug("task[%lu:%s] started execution on dev[%d][%s] time[%lf] start:[%lf] q[%d]", task->uid(), task->name(), devno(), name(), task->time(), task->time_start(), task->recommended_stream());
  _trace("task[%lu:%s] started execution on dev[%d][%s] time[%lf] start:[%lf] q[%d]", task->uid(), task->name(), devno(), name(), task->time(), task->time_start(), task->recommended_stream());
  //for(Command *cmd : task->reset_mems()) {
      // Handle Memory Reset commands IRIS_CMD_RESET_INPUT first
  //    ExecuteMemResetInput(task, cmd);
  //}
  //if (task->cmd_kernel()) ExecuteMemIn(task, task->cmd_kernel());     
  vector<DataMem *> d2h_dmems;
  for (int i = 0; i < task->ncmds(); i++) {
      Command* cmd = task->cmd(i);
      // If there are tasks with explicit D2H of DMEM objects 
      // which are not part of kernel, these should be tracked 
      // to set host dirty flag and device valid flags
      if (cmd->type() == IRIS_CMD_D2H && (
              cmd->mem()->GetMemHandlerType() == IRIS_DMEM ||
              cmd->mem()->GetMemHandlerType() == IRIS_DMEM_REGION)) {
         d2h_dmems.push_back((DataMem*)cmd->mem());
      }
  }
  HandleHiddenDMemIns(task);
  for (int i = 0; i < task->ncmds(); i++) {
    //printf("Inside device cmd:%d execute and calling free destroy events %lu:%s %lf\n", i, task->uid(), task->name(), timer_->Now());
    Command* cmd = task->cmd(i);
    if (hook_command_pre_) hook_command_pre_(cmd);
    cmd->set_time_start(timer_);
    switch (cmd->type()) {
      case IRIS_CMD_INIT:         ExecuteInit(cmd);       break;
      case IRIS_CMD_KERNEL:       {
                                      ExecuteMemIn(task, cmd);
                                      ExecuteKernel(cmd);     
                                      ExecuteMemOut(task, task->cmd_kernel());
                                      for(DataMem *dmem : d2h_dmems) {
                                          dmem->set_dirty_except(devno_);
                                          dmem->set_host_dirty();
                                          dmem->disable_reset();
                                      }
                                      break;
                                  }
      case IRIS_CMD_DMEM2DMEM_COPY: {
                                      ExecuteDMEM2DMEM(task, cmd);
                                      break;
                                  }
      case IRIS_CMD_MALLOC:       ExecuteMalloc(cmd);     break;
      case IRIS_CMD_H2D:          ExecuteH2D(cmd);        break;
      case IRIS_CMD_H2BROADCAST:  ExecuteH2BroadCast(cmd); break;
      case IRIS_CMD_H2DNP:        ExecuteH2DNP(cmd);      break;
      case IRIS_CMD_D2H:          ExecuteD2H(cmd);        break;
      case IRIS_CMD_MEM_FLUSH:    ExecuteMemFlushOut(cmd);break;
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
      case IRIS_CMD_MEM_FLUSH_TO_SHADOW:    ExecuteMemFlushOutToShadow(cmd);break;
#endif
#endif
      case IRIS_CMD_RESET_INPUT : ExecuteMemResetInput(task, cmd); break;
      case IRIS_CMD_MAP:          ExecuteMap(cmd);        break;
      case IRIS_CMD_RELEASE_MEM:  ExecuteReleaseMem(cmd); break;
      case IRIS_CMD_HOST:         ExecuteHost(cmd);       break;
      case IRIS_CMD_CUSTOM:       ExecuteCustom(cmd);     break;
      default: {_error("cmd type[0x%x]", cmd->type());  printf("TODO: determine why name (%s) is set, but type isn't\n",cmd->type_name());};
    }
    //printf("Inside post device execute and calling free destroy events %lu:%s %lf\n", task->uid(), task->name(), timer_->Now());
    cmd->set_time_end(timer_);
    if (hook_command_post_) hook_command_post_(cmd);
  }
  HandleHiddenDMemOuts(task);
  task->update_status(IRIS_SUBMITTED);
  if (platform_obj_->is_event_profile_enabled()) {
      double execute_end = (timer_->Now()-first_event_cpu_mid_point_time())*1000.0;
      task->CreateProfileEvent(task, devno(), PROFILE_INIT, this, (float)execute_start, (float)execute_end);
  }
  if (is_async(task) && task->user()) AddCallback(task);
  task->set_time_end(timer_);
  _debug2("Task %s:%lu refcnt:%d\n", task->name(), task->uid(), task->ref_cnt());
  TaskPost(task);
  if (hook_task_post_) hook_task_post_(task);
  if (!task->system()) _trace("task[%lu:%s] complete dev[%d][%s] time[%lf] end:[%lf]", task->uid(), task->name(), devno(), name(), task->time(), task->time_end());
  _debug2("Task %s:%lu refcnt:%d\n", task->name(), task->uid(), task->ref_cnt());
  if (task->cmd_kernel() != NULL) ProactiveTransfers(task, task->cmd_kernel());
  if (!is_async(task) || !task->user()) { FreeActiveTask(); task->Complete(); }
  busy_ = false;
}

bool Device::IsFree()
{
    //cout <<"Active tasks: "<<active_tasks_<<endl;
    if (active_tasks_ == 0) return true;
    return false;
}
int Device::AddCallback(Task* task) {
  int stream_index = task->last_cmd_stream();
  if (stream_index == -1) 
      stream_index = GetStream(task); 
  Device *dev = task->last_cmd_device();
  _event_debug("Waiting call back complete for task:%lu:%s on stream:%d task_stream:%d last_cmd_stream:%d dev:%p last_cmd_dev:%p\n", task->uid(), task->name(), stream_index, GetStream(task), task->last_cmd_stream(), this, dev);
  if (dev != NULL && dev != this) 
      return dev->RegisterCallback(stream_index, (CallBackType)Device::Callback, task, iris_stream_non_blocking);
  else
      return RegisterCallback(stream_index, (CallBackType)Device::Callback, task, iris_stream_non_blocking);
}

void Device::Callback(void *stream, int status, void* data) {
  Task* task = (Task*) data;
  //printf("----Function is callbacked stream:%p status:%d data:%p\n", stream, status, data);
  unsigned long uid = task->uid();
  string tname = task->name();
  task->dev()->FreeActiveTask();
  _event_prof_debug(" ------ Just before completed task stream_ptr:%p task:%p:%s:%lu status:%d data:%p\n", stream, task, tname.c_str(), uid, status, data);
  _event_debug(" ------ Just before completed task stream_ptr:%p task:%p:%s:%lu status:%d data:%p\n", stream, task, tname.c_str(), uid, status, data);
  task->Complete();
  _event_prof_debug(" ------ Completed and Safe return from callback task stream_ptr:%p task:%p:%s:%lu status:%d data:%p\n", stream, task, tname.c_str(), uid, status, data);
}
void Device::ResolveDeviceWrite(Task *task, BaseMem *mem, Device *input_dev, bool instant_wait, BaseMem *src_mem)
{
    if (src_mem == NULL) src_mem = mem;
    int input_devno = input_dev->devno();
    void *input_event = src_mem->GetWriteDeviceEvent(input_devno);
    int input_stream = src_mem->GetWriteStream(input_devno);
    if (input_stream != -1) {
        if (instant_wait) {
            mem->HardDeviceWriteEventSynchronize(input_dev, input_event);
            ResetContext();
            _event_debug(" HardDeviceWriteEventSynchronize dev:[%d][%s] src_dev:[%d][%s] Wait for event:%p input_stream:%d", devno(), name(), input_dev->devno(), input_dev->name(), input_event, input_stream); 
        }
        else {
            int mem_stream = GetStream(task, mem);
            WaitForEvent(input_event, mem_stream, iris_event_wait_default);
            _event_debug(" Wait Event dev:[%d][%s] src_dev:[%d][%s] mem:%lu Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_dev->devno(), input_dev->name(), mem->uid(), input_event, mem_stream, input_stream); 
        }
    }
}
/*
void Device::ResolveHostWrite(Task *task, BaseMem *mem, bool instant_wait) {
    int input_devno = mem->GetHostWriteDevice();
    if (input_devno != -1) {
        void *input_event = mem->GetHostCompletionEvent();
        int input_stream = mem->GetHostWriteStream();
        if (input_stream != -1) {
            if (instant_wait) {
                Device *input_dev = Platform::GetPlatform()->device(input_devno);
                input_dev->EventSynchronize(input_event);
                _event_debug(" WaitForEvent ASYNC_H2D_RESOLVE_SYNC dev:[%d][%s] src_dev:%d Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_devno, input_event, mem_stream, input_stream); 
            }
            else {
                int mem_stream = input_dev->GetStream(task, mem);
                //TODO: Reverify here
                input_dev->WaitForEvent(input_event, mem_stream, iris_event_wait_default);
            }
        }
    }
}
*/
template <AsyncResolveType resolve_type>
void Device::ResolveInputWriteDependency(Task *task, BaseMem *mem, bool async, Device *select_src_dev, BaseMem *src_mem)
{
    if (src_mem == NULL) src_mem = mem;
    if(!async) return;
    if (resolve_type == ASYNC_D2D_RESOLVE) {
        if (!async && 
                select_src_dev != NULL && select_src_dev->is_async(false) && 
                src_mem->GetWriteStream(select_src_dev->devno()) != -1) {
            // Src is async but dest is not
            ResolveDeviceWrite(task, mem, select_src_dev, true, src_mem);
            return;
        }
        else if (!async) return;
        // Src and destination should be same type of device
        ResolveDeviceWrite(task, mem, select_src_dev, false, src_mem);
    }
    else if (resolve_type == ASYNC_DEV_INPUT_RESOLVE) {
        if (!async && 
                select_src_dev != NULL && select_src_dev->is_async(false) && 
                src_mem->GetWriteStream(select_src_dev->devno()) != -1) {
            // Src is async but dest is not
            ResolveDeviceWrite(task, mem, select_src_dev, true, src_mem);
            return;
        }
#ifdef ENABLE_SAME_TYPE_GPU_OPTIMIZATION
        Device *input_dev =  select_src_dev;
        ASSERT(select_src_dev != NULL);
        int input_devno = input_dev->devno();
        if (async && input_devno != -1 && input_dev->model() == model() && 
                input_dev->type() == type()) {
            ResolveDeviceWrite(task, mem, select_src_dev, false, src_mem);
            return;
        }
#endif 
        if (async) {
            ResolveDeviceWrite(task, mem, select_src_dev, true, src_mem);
        }
    }
    else if (resolve_type == ASYNC_SAME_DEVICE_DEPENDENCY) {
        if (!async && 
                select_src_dev != NULL && select_src_dev->is_async(false) && 
                src_mem->GetWriteStream(select_src_dev->devno()) != -1) {
            // Src is async but dest is not
            ResolveDeviceWrite(task, mem, select_src_dev, true, src_mem);
            return;
        }
        if (!async) return;
        Device *input_dev =  select_src_dev;
        int mem_stream = input_dev->GetStream(task, src_mem); 
        ASSERT(select_src_dev != NULL);
        int input_devno = input_dev->devno();
        int input_stream = src_mem->GetWriteStream(input_devno);
        if (input_stream != -1 && input_stream != mem_stream) {
            void *input_event = src_mem->GetWriteDeviceEvent(input_devno);
            input_dev->ResetContext();
            input_dev->WaitForEvent(input_event, mem_stream, iris_event_wait_default);
            ResetContext();
            _event_debug(" Wait ASYNC_SAME_DEVICE_DEPENDENCY dev:[%d][%s] src_dev:[%d][%s] Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_dev->devno(), input_dev->name(), input_event, mem_stream, input_stream); 
        }
    }
    /*
    else if (resolve_type == ASYNC_H2D_RESOLVE_SYNC) {
        int input_devno = mem->GetHostWriteDevice();
        if (input_devno != -1) {
            void *input_event = mem->GetHostCompletionEvent();
            int input_stream = mem->GetHostWriteStream();
            Device *input_dev = (input_devno != -1) ? 
                Platform::GetPlatform()->device(input_devno) : NULL;
            mem->HardHostWriteEventSynchronize(input_dev, input_event);
            ResetContext();
            _event_debug(" HardHostWriteEventSynchronize ASYNC_H2D_RESOLVE_SYNC dev:[%d][%s] src_dev:%d Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_devno, input_event, mem_stream, input_stream); 
        }
    }*/
    else if (resolve_type == ASYNC_UNKNOWN_H2D_RESOLVE) {
        int input_devno = src_mem->GetHostWriteDevice();
        int input_stream = src_mem->GetHostWriteStream();
        int mem_stream = GetStream(task, mem);
        if (input_stream  != -1) {
            void *input_event = src_mem->GetDeviceSpecificHostCompletionEvent(input_devno);
            Device *input_dev = (input_devno != -1) ? 
                Platform::GetPlatform()->device(input_devno) : NULL;
#ifdef ENABLE_SAME_TYPE_GPU_OPTIMIZATION
            if (input_devno != -1 && input_dev->model() == model() && 
                    input_dev->type() == type() && 
                    (model() == iris_cuda || model() == iris_hip)) {
                WaitForEvent(input_event, mem_stream, iris_event_wait_default);
                _event_debug(" WaitForEvent ASYNC_UNKNOWN_H2D_RESOLVE dev:[%d][%s] src_dev:[%d][%s] mem:%lu Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_devno, input_dev->name(), mem->uid(), input_event, mem_stream, input_stream); 
            }
            else //if (input_devno != -1) 
#endif 
            {
                // Threre could be scenario where another explicit D2H is already completed. 
                if (input_dev == NULL) return;
                if (input_event == NULL) return;
                ASSERT(input_dev != NULL);
                ASSERT(input_event != NULL);
                _event_debug(" EventSynchronize ASYNC_UNKNOWN_H2D_RESOLVE H2D mem:%lu dev:[%d][%s] src_dev:[%d][%s] Wait for event:%p mem_stream:%d input_stream:%d", mem->uid(), devno(), name(), input_devno, input_dev->name(), input_event, mem_stream, input_stream); 
                DeviceEventExchange(task, mem, input_event, input_stream, input_dev);
            }
        }
    }
    else if (resolve_type == ASYNC_KNOWN_H2D_RESOLVE) {
        Device *input_dev = select_src_dev;
        int input_devno = input_dev->devno();
        void *input_event = src_mem->GetDeviceSpecificHostCompletionEvent(input_devno);
        int input_stream = input_dev->GetStream(task, src_mem); 
        int mem_stream = GetStream(task, mem); 
        if (input_stream  != -1) {
            Device *input_dev = (input_devno != -1) ? 
                Platform::GetPlatform()->device(input_devno) : NULL;
#ifdef ENABLE_SAME_TYPE_GPU_OPTIMIZATION
            if (input_devno != -1 && input_dev->model() == model() && 
                    input_dev->type() == type() && 
                    (model() == iris_cuda || model() == iris_hip)) {
                WaitForEvent(input_event, mem_stream, iris_event_wait_default);
                _event_debug(" WaitForEvent ASYNC_KNOWN_H2D_RESOLVE dev:[%d][%s] src_dev:[%d][%s] mem:%lu Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_devno, input_dev->name(), mem->uid(), input_event, mem_stream, input_stream); 
            }
            else //if (input_devno != -1) 
#endif 
            {
                _event_debug(" EventSynchronize ASYNC_KNOWN_H2D_RESOLVE H2D mem:%lu dev:[%d][%s] src_dev:[%d][%s] Wait for event:%p mem_stream:%d input_stream:%d", mem->uid(), devno(), name(), input_devno, input_dev->name(), input_event, mem_stream, input_stream); 
                if (input_dev == NULL) return;
                if (input_event == NULL) return;
                ASSERT(input_dev != NULL);
                ASSERT(input_event != NULL);
                DeviceEventExchange(task, mem, input_event, input_stream, input_dev);
            }
        }
    }
}
void Device::DeviceEventExchange(Task *task, BaseMem *mem, void *input_event, int input_stream, Device *input_dev)
{
    int mem_stream = GetStream(task, mem);
    int input_devno = input_dev->devno();
#if 0 //This is not needed?
    //src_dev->ResetContext();
    src_dev->RegisterCallback(src_mem_stream, 
            (CallBackType)BaseEventExchange::Fire, 
            //This should be current device shared event exchange object
            exchange, 
            iris_stream_non_blocking);
#endif
#ifdef DIRECT_H2D_SYNC
    _event_debug("Device synchronizing for mem:%lu task:%lu:%s dev:%d:%s event:%p", mem->uid(), task->uid(), task->name(), input_dev->devno(), input_dev->name(), input_event);
    //TODO: Is it always D2H -> H2D or can it be D2D -> D2H / H2D -> D2D / H2D -> D2H ?
    mem->HardHostWriteEventSynchronize(input_dev, input_event);
    ResetContext();
    _event_debug("Device completed synchronizing for mem:%lu task:%lu:%s dev:%d:%s event:%p", mem->uid(), task->uid(), task->name(), input_dev->devno(), input_dev->name(), input_event);
#else
    void *dest_event = NULL;
    BaseEventExchange *exchange = mem->GetEventExchange(devno());
    exchange->set_mem(mem->uid(), input_stream, input_devno, mem_stream, devno(), input_dev, this, input_event, dest_event);
    RegisterCallback(mem_stream, 
            BaseEventExchange::Wait, 
            exchange,
            iris_stream_default);
    ResetContext();
    _event_debug("Creating callback exchange for synchronizing between mem:%lu task:%lu:%s srcdev:%d:%s event:%p, dev:%d:%s dest_event:%p", mem->uid(), task->uid(), task->name(), input_dev->devno(), input_dev->name(), input_event, devno(), name(), exchange->dest_event());
#endif

}
template <AsyncResolveType resolve_type>
void Device::ResolveOutputWriteDependency(Task *task, BaseMem *mem, bool async, Device *select_src_dev)
{
    if(!async) return;
    int mem_stream = GetStream(task, mem);
    if (resolve_type == ASYNC_D2H_SYNC) {
        Device *input_dev = select_src_dev;
        int input_devno = input_dev->devno();
        int input_stream = input_dev->GetStream(task, mem);
        void *input_event = mem->GetDeviceSpecificHostCompletionEvent(input_devno);
        if (input_stream != -1) {
            mem->HardHostWriteEventSynchronize(input_dev, input_event);
            ResetContext();
            _event_debug(" HardHostWriteEventSynchronize ASYNC_D2H_SYNC dev:[%d][%s] src_dev:[%d][%s] Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_dev->devno(), input_dev->name(), input_event, mem_stream, input_stream); 
        }
    }
    else if (resolve_type == ASYNC_D2O_SYNC) {
        Device *input_dev = select_src_dev;
        int input_devno = input_dev->devno();
        int input_stream = input_dev->GetStream(task, mem);
        void *input_event = mem->GetWriteDeviceEvent(input_devno);
        if (input_stream != -1) {
            mem->HardDeviceWriteEventSynchronize(input_dev, input_event);
            ResetContext();
            _event_debug(" HardDeviceWriteEventSynchronize ASYNC_D2O_SYNC dev:[%d][%s] src_dev:[%d][%s] Wait for event:%p mem_stream:%d input_stream:%d", devno(), name(), input_dev->devno(), input_dev->name(), input_event, mem_stream, input_stream); 
        }
    }
}
void Device::SynchronizeInputToMemory(Task *task, BaseMem *mem)
{
    int mem_stream = GetStream(task, mem);
    int write_dev = mem->GetWriteDevice();
    if (write_dev != -1) {
        int written_stream  = mem->GetWriteStream(write_dev);
        if (written_stream != -1) {
            Device *src_dev = Platform::GetPlatform()->device(write_dev);
            void *event = mem->GetWriteDeviceEvent(write_dev);
            //The upcoming D2H depends on previous complete event
            //printf("Event:%p mem_stream:%d write_dev:%d dev:%d\n", event, mem_stream, write_dev, devno_);
            // Even if device model is same (OpenCL), their types could be different
            if (src_dev->model() != model() || src_dev->type() != type()) {
                mem->HardDeviceWriteEventSynchronize(src_dev, event);
                ResetContext();
            }
            else {
                WaitForEvent(event, mem_stream, iris_event_wait_default);
                _event_debug(" WaitForEvent dev:[%d][%s] src_dev:%d Wait for event:%p mem_stream:%d written_stream:%d", devno(), name(), src_dev->devno(), event, mem_stream, written_stream); 
            }
        }
    }
}
void Device::ResolveH2DEndEvents(Task *task, BaseMem *mem, bool async)
{
    int mem_stream = GetStream(task, mem);
    if (async && platform_obj_->is_event_profile_enabled()) {
        ProfileEvent & prof_event = task->LastProfileEvent();
        _event_debug("prof_event ptr:%p\n", prof_event.start_event_ptr());
        prof_event.RecordEndEvent(); 
    }
    if (async) {
        assert(mem_stream != -1);
        EVENT_DEBUG(void *event = )mem->RecordEvent(devno(), mem_stream, true);
        //EVENT_DEBUG(mem->HardDeviceWriteEventSynchronize(this, event);
        _event_debug("h2d: RecordEvent adding event (H2D) dev[%d][%s] task[%ld:%s] mem:%lu q[%d] event:%p\n", devno_, name_, task->uid(), task->name(), mem->uid(), mem_stream, event);
    }
}
void Device::ResolveH2DStartEvents(Task *task, BaseMem *mem, bool async, BaseMem *src_mem)
{
    int mem_stream = GetStream(task, mem);
    ResolveInputWriteDependency<ASYNC_UNKNOWN_H2D_RESOLVE>(task, mem, async, NULL, src_mem);
    ResetContext();
    if (async && platform_obj_->is_event_profile_enabled()) {
        ProfileEvent & prof_event = task->CreateProfileEvent(mem, -1, PROFILE_H2D, this, mem_stream);
        prof_event.RecordStartEvent(); 
        _event_debug("prof_event ptr:%p %p task:%lu:%s mem:%lu\n", prof_event.start_event_ptr(), prof_event.start_event(), task->uid(), task->name(), mem->uid());
    }
}
void Device::ProactiveTransfers(Task *task, Command *cmd)
{
  if (!Platform::GetPlatform()->get_enable_proactive()) return;
  if (cmd->kernel() == NULL) return;
  _debug2("Doing proactive transfers for task:%s:%lu", task->name(), task->uid());
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
  cmd->set_time_start(timer_);
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
      _warning("NO KERNEL SRC[%s] NO KERNEL BIN[%s]", src, bin);
      native_kernel_not_exists_ = true;
    } else if (!stat_src && stat_bin) {
      kernel_path_ = std::string(bin);
      //strncpy(kernel_path_, bin, strlen(bin)+1);
    } else if (stat_src && !stat_bin) {
      Platform::GetPlatform()->EnvironmentGet(kernel_bin(), &bin, NULL);
      //sprintf(kernel_path_, "%s/%s-%d", tmpdir, bin, devno_);
      kernel_path_ = string(tmpdir) + "/" + string(bin) + "-" + std::to_string(devno_);
      errid_ = Compile(src);
    } else {
      long mtime_src = Utils::Mtime(src);
      long mtime_bin = Utils::Mtime(bin);
      if (mtime_src > mtime_bin) {
        Platform::GetPlatform()->EnvironmentGet(kernel_bin(), &bin, NULL);
        kernel_path_ = string(tmpdir) + "/" + string(bin) + "-" + std::to_string(devno_);
        //sprintf(kernel_path_, "%s/%s-%d", tmpdir, bin, devno_);
        errid_ = Compile(src);
      } else
         kernel_path_ = string(bin); 
          //strncpy(kernel_path_, bin, strlen(bin)+1);
    }
    if (errid_ == IRIS_ERROR) _error("iret[%d]", errid_);
  }
  errid_ = Init();
  if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
  size_t init_size = 16; // Should be multiple of 4
  //send some memory for the device (important for spinning up AMD devices)
  Mem* mem = new Mem(init_size, Platform::GetPlatform());
  mem->SetOwner(this);
  void* src_arch = mem->host_inter(); //mem->arch(this);
  size_t off[3] = { 0 };
  size_t host_sizes[3] = { mem->size() };
  size_t dev_sizes[3] = { mem->size() };
  bool async = async_;
  async_ = false;
  MemH2D(cmd->task(), mem, off, host_sizes, dev_sizes, 4, 1, mem->size(), src_arch, "Init H2D");
  async_ = async;
  delete mem;
  cmd->set_time_end(timer_);
  double time = timer_->Stop(IRIS_TIMER_INIT);
  cmd->SetTime(time);
  if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddKernel(cmd);
  enable_ = true;
}

void Device::ExecuteKernel(Command* cmd) {
  timer_->Start(IRIS_TIMER_KERNEL);
  cmd->set_time_start(timer_);
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
  KernelLaunchInit(cmd, kernel);
  //double ltime = timer_->GetCurrentTime() - ltime_start;
  KernelArg* args = cmd->kernel_args();
  int *params_map = cmd->get_params_map();
  int arg_idx = 0;
  int stream_index = 0;
  Task *task = cmd->task();
  bool async_launch = is_async(cmd->task());
  if (async_launch) {
      WaitForTaskInputAvailability(devno(), cmd->task(), cmd);
      async_launch = true;
      stream_index = GetStream(cmd->task()); //task->uid() % nqueues_; 
  }
  if (async_launch && platform_obj_->is_malloc_async()) {
      ResetContext();
      for(auto && z : cmd->kernel()->out_mems()) {
          BaseMem *mem = z.first;
          int idx = z.second;
          if (params_map != NULL && 
                  (params_map[idx] & iris_all) == 0 && 
                  !(params_map[idx] & type_) ) continue;
          if (mem->GetMemHandlerType() == IRIS_DMEM ||
                  mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
              DataMem *dmem = (DataMem *)mem;
              if (dmem->get_source_mem() != NULL) dmem = dmem->get_source_mem();
              if (dmem->get_arch(devno()) == NULL) {
                  dmem->set_recommended_stream(devno(), stream_index);
                  _event_debug(" Set stream for write mem of task:%lu:%s dev:[%d][%s] mem:%lu task_stream:%d", task->uid(), task->name(), devno(), name(), mem->uid(), stream_index);
                  // It will create a memory in device
                  dmem->arch(this);
              }
          }
      }
  }
  //double atime_start = timer_->GetCurrentTime();
  //double set_mem_time = 0.0f;
  for (int idx = 0; idx < cmd->kernel_nargs(); idx++) {
    if (idx > max_idx) max_idx = idx;
    KernelArg* arg = args + idx;
    BaseMem* bmem = (BaseMem*)arg->mem;
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
        KernelSetMem(kernel, arg_idx, idx, bmem, arg->off); arg_idx+=1; 
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
  StreamPolicy policy = stream_policy(task);
  // Though STREAM_POLICY_SAME_FOR_TASK doesn't require for every input to check, it is required for the cases where the D2O (CUDA to OpenMP) device data transfers are enabled.
  task->set_last_cmd_stream(stream_index);
  bool enabled = true;
  if (task != NULL && (
              task->is_kernel_launch_disabled() ||
              Platform::GetPlatform()->is_kernel_launch_disabled()))
      enabled = false;
  if (enabled) {
      _debug2("Launching kernel:%s:%lu task:%s:%lu stream:%d", kernel->name(), kernel->uid(), task->name(), task->uid(), stream_index);
      if (async_launch && platform_obj_->is_event_profile_enabled()) {
          ProfileEvent &prof_event = task->CreateProfileEvent(task, devno(), PROFILE_KERNEL, this, stream_index);
          prof_event.RecordStartEvent(); 
      }
      errid_ = KernelLaunch(kernel, dim, off, gws, lws[0] > 0 ? lws : NULL);
      if (async_launch && platform_obj_->is_event_profile_enabled()) {
          ProfileEvent &prof_event = task->LastProfileEvent();
          prof_event.RecordEndEvent(); 
      }
      _debug2("Completed kernel:%s:%lu task:%s:%lu", kernel->name(), kernel->uid(), task->name(), task->uid());
  }
  //double ktime = timer_->GetCurrentTime() - ktime_start;
  cmd->set_time_end(timer_);
  double time = timer_->Stop(IRIS_TIMER_KERNEL);
  cmd->SetTime(time);
  if (async_launch) {
      void **event = cmd->kernel()->GetCompletionEventPtr(true);
      RecordEvent(event, task->recommended_stream());
      _event_debug("Task RecordEvent recording event for task:%lu %s stream:%d event:%p\n", task->uid(), task->name(), task->recommended_stream(), *event);
      _event_prof_debug("Task RecordEvent recording event for task:%lu %s stream:%d event:%p\n", task->uid(), task->name(), task->recommended_stream(), *event);
      //if (task->uid() == 14)
      //EventSynchronize(cmd->kernel()->GetCompletionEvent());
  }
  //printf("Task:%s time:%f ktime:%f init:%f atime:%f setmemtime:%f\n", task->name(), time, ktime, ltime, atime, set_mem_time);
  cmd->kernel()->history()->AddKernel(cmd, this, time);
  if (!async_launch && Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddKernel(cmd);
}

void Device::ExecuteMalloc(Command* cmd) {
  bool async_launch = is_async(cmd->task());
  cmd->set_time_start(timer_);
  Mem* mem = cmd->mem();
  void* arch = mem->arch(this);
  cmd->set_time_end(timer_);
  if (!async_launch && Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddH2D(cmd);
  _trace("dev[%d] malloc[%p]", devno_, arch);
}

void Device::GetPossibleDevices(BaseMem *mem, int devno, int *nddevs, int &d2d_dev, int &cpu_dev, int &non_cpu_dev, bool async)
{
    d2d_dev = -1;
    cpu_dev = -1;
    non_cpu_dev = -1;
    for(int i=0; nddevs[i] != -1; i++) {
        Device *target_dev = Platform::GetPlatform()->device(nddevs[i]);
        if (d2d_dev == -1 && type() == target_dev->type() &&
                isD2DEnabled() && target_dev->isD2DEnabled() && IsD2DPossible(target_dev)) {
            if (async && platform_obj_->is_malloc_async() && IsAddrValidForD2D(mem, mem->get_arch(target_dev->devno())))
            //if (async && platform_obj_->is_malloc_async())
                d2d_dev = nddevs[i];
            else if (!async || !platform_obj_->is_malloc_async())
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
    if (bmem->GetMemHandlerType() == IRIS_DMEM || 
            bmem->GetMemHandlerType() == IRIS_DMEM_REGION) {
        int mem_stream = GetStream(task); 
        DataMem* mem = (DataMem *)cmd->mem();
        if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
        mem->dev_lock(devno_);
        ResetMemory(task, cmd, mem);
        mem->set_host_dirty();
        mem->set_dirty_except(devno_);
        mem->dev_unlock(devno_);
        if (is_async(task)) {
            mem->clear_d2h_events();
            mem->clear_streams();
            EVENT_DEBUG(void *k_event = ) mem->RecordEvent(devno(), mem_stream, true); 
            //mem->SetWriteDevice(devno());
            //mem->SetWriteDeviceEvent(devno(), mem->GetCompletionEvent(devno()));
            //mem->SetWriteStream(devno(), mem_stream);
            //It should create new entry of event instead of using existing one
            _event_debug("Reset RecordEvent mem set stream   task:[%lu][%s] output dmem:%lu stream:%d, dev:%d event:%p\n", task->uid(), task->name(), mem->uid(), mem_stream, devno(), k_event);
        }
    }
}

void Device::ExecuteDMEM2DMEM(Task *task, Command *cmd) {
    BaseMem *src_mem = cmd->mem();
    BaseMem *dst_mem = cmd->dst_mem();
    if (src_mem->GetMemHandlerType() == IRIS_DMEM && 
            dst_mem->GetMemHandlerType() == IRIS_DMEM) {
        DataMem *dsrc_mem = (DataMem*)src_mem;
        DataMem *ddst_mem = (DataMem*)dst_mem;
        if (dsrc_mem->get_source_mem() != NULL) dsrc_mem = dsrc_mem->get_source_mem();
        if (ddst_mem->get_source_mem() != NULL) ddst_mem = ddst_mem->get_source_mem();
        InvokeDMemInDataTransfer<DataMem>(task, cmd, ddst_mem, NULL, dsrc_mem);
    }
}
void Device::HandleHiddenDMemIns(Task *task) 
{
    Command* cmd_kernel = task->cmd_kernel();
    if (cmd_kernel == NULL) return;
    // There should be atleast one kernel
    for(DataMem *dmem : task->hidden_dmem_in()) {
        if (dmem->GetMemHandlerType() == IRIS_DMEM) {
            ExecuteMemInDMemIn(task, cmd_kernel, dmem);
        }
        else if (dmem->GetMemHandlerType() == IRIS_DMEM_REGION) {
            DataMemRegion *rdmem = (DataMemRegion*)dmem;
            ExecuteMemInDMemRegionIn(task, cmd_kernel, rdmem);
        }
    }
}
void Device::HandleHiddenDMemOuts(Task *task) 
{
    Command* cmd_kernel = task->cmd_kernel();
    if (cmd_kernel == NULL) return;
    // There should be atleast one kernel
    for(DataMem *dmem : task->hidden_dmem_out()) {
        if (dmem->GetMemHandlerType() == IRIS_DMEM ||
            dmem->GetMemHandlerType() == IRIS_DMEM_REGION) {
            dmem->set_dirty_except(devno_);
            dmem->set_host_dirty();
            dmem->disable_reset();
        }
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
    // If order of DMEM data transfers is provided
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

void Device::WaitForTaskInputAvailability(int devno, Task *task, Command *cmd)
{
    for(auto && z : cmd->kernel()->in_mems()) {
        BaseMem *mem = z.first;
        int idx = z.second;
        //if (mem->GetMemHandlerType() == IRIS_MEM) continue;
        BaseMem *dmem = (BaseMem *)mem;
        int task_stream = GetStream(task);
        int dmem_stream = dmem->GetWriteStream(devno);
        void *event = dmem->GetWriteDeviceEvent(devno);
        // If DMEM input is not yet happened due to reset flag enabled
        if (dmem_stream != task_stream && event != NULL) {
            _event_prof_debug("Wait for event\n");
            WaitForEvent(event, task_stream, iris_event_wait_default);
            _event_prof_debug(" WaitForEvent task:%s:%lu mem:%lu Waiting for event:%p to be fired devno:%d task_stream:%d waiting for dmem_stream:%d\n", task->name(), task->uid(), dmem->uid(), event, devno, task_stream, dmem_stream);
        }
    }
}
template <typename DMemType>
void Device::WaitForDataAvailability(int ldevno, Task *task, DMemType *mem, int read_stream)
{
    int stream = mem->GetWriteStream(ldevno);
    if (read_stream == -1)
        read_stream = GetStream(task, mem);
    // TODO: Check. Why we are testing stream and read_stream. They could be same. Need a better logic. 
    _event_debug("      WFDA: dev:%d stream:%d, read_stream:%d ldevno:%d devno:%d mem:%lu", devno(), stream, read_stream, ldevno, devno_, mem->uid());
    if ((stream != -1) && ((stream != read_stream) || (ldevno != devno_))) {
        Device *ldev= Platform::GetPlatform()->device(ldevno);
        //Device *dev= Platform::GetPlatform()->device(dev);
        // Even if the parent task and current task are running on same device, it may be using different streams
        for (void * event: mem->GetWaitEvents(ldevno)) {
            _event_debug(" WaitForEvent from dev:[%d][%s] event:%p to ldev:[%d][%s] task:%s:%lu task_stream:%d mem:%lu mem_stream:%d", devno_, name(), event, ldevno, ldev->name(), task->name(), task->uid(), read_stream, mem->uid(), stream);
            WaitForEvent(event, read_stream, iris_event_wait_default);
        }
    }
}
template <typename DMemType>
void Device::InvokeDMemInDataTransfer(Task *task, Command *cmd, DMemType *mem, BaseMem *parent, DMemType *src_mem)
{
    int nddevs[IRIS_MAX_NDEVS+1];
    Kernel *kernel = cmd->kernel();
    size_t *ptr_off = mem->local_off();
    size_t *gws = mem->host_size();
    size_t *lws = mem->dev_size();
    size_t elem_size = mem->elem_size();
    int dim = mem->dim();
    size_t size = mem->size();
    //bool is_src_mem_different = false;
    if (src_mem != NULL) {
        //ASSERT(src_mem->dim() == mem->dim());This condition cannot be satisfied now
        ASSERT(src_mem->size() == mem->size());
        ASSERT(src_mem->elem_size() == mem->elem_size());
        //is_src_mem_different = true;
        mem->set_host_dirty(true);
        mem->set_dirty_all(true);
    }
    else {
        src_mem = mem;
    }
    cmd->set_devno(devno_);
    void *cmd_host = cmd->host();
    if (cmd_host != NULL && src_mem->host_ptr() != cmd_host) {
        src_mem->set_dirty_all(true);
        src_mem->set_host_dirty(false);
    }
    mem->dev_lock(devno_);
    int cpu_dev = -1;
    int non_cpu_dev = -1;
    int d2d_dev = -1;
    bool async = is_async(task);
    GetPossibleDevices(src_mem, devno_, src_mem->get_non_dirty_devices(nddevs), 
            d2d_dev, cpu_dev, non_cpu_dev, async);
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
    //_debug2(" task:%s:%lu mem:%lu mem_stream:%d", task->name(), task->uid(), mem->uid(), mem_stream);
    // Check if it is still dirty
    if (!Platform::GetPlatform()->is_d2d_disabled() && d2d_dev >= 0) { 
        int mem_stream = GetStream(task, mem, true);
        // May be transfer directly from peer device is best 
        // Do D2D communication
        // Keep host data dirty as it is
        _event_debug("explore D2D dev[%d][%s] task[%ld:%s] mem[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), mem_stream);
        _trace("explore D2D dev[%d][%s] task[%ld:%s] mem[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), mem_stream);
        Device *src_dev = Platform::GetPlatform()->device(d2d_dev);

        ResolveInputWriteDependency<ASYNC_D2D_RESOLVE>(task, mem, async, src_dev, src_mem);

        if (async && platform_obj_->is_event_profile_enabled()) {
            ProfileEvent & prof_event = task->CreateProfileEvent(mem, src_dev->devno(), PROFILE_D2D, this, mem_stream);
            prof_event.RecordStartEvent(); 
        }
        double start = timer_->Now();
        void* src_arch = src_mem->arch(src_dev);
        ResetContext();
        void* dst_arch = mem->arch(this);

        _event_debug("D2D: src:%d dest:%d src_arch:%p dst_arch:%p mem_stream:%d\n\n", src_dev->devno(), this->devno(), src_arch, dst_arch, mem_stream);
        // Now do D2D
        if (!platform_obj_->is_data_transfers_disabled())
            MemD2D(task, src_dev, mem, dst_arch, src_arch, mem->size());
        double end = timer_->Now();
        // If device is not asynchronous, you don't need to record event in CUDA/HIP device
        if (async && platform_obj_->is_event_profile_enabled()) {
                ProfileEvent & prof_event = task->LastProfileEvent();
                prof_event.RecordEndEvent(); 
        }
        if (async) {
            mem->RecordEvent(devno(), mem_stream, true);
        }
        if (!async && kernel != NULL && kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2d, (uint32_t) d2d_dev, (uint32_t) devno_, start, end});
        }
        d2dtime = end - start;
        d2d_enabled = true;
        if (!async && Platform::GetPlatform()->is_scheduling_history_enabled()){
            string cmd_name = "Internal-D2D(" + string(cmd->task()->name()) + ")-from-" + to_string(src_dev->devno()) + "-to-" + to_string(this->devno());
            Platform::GetPlatform()->scheduling_history()->Add(cmd, cmd_name, "MemD2D", start,end);
        }
        if (parent != NULL && async) {
            void *event = mem->GetCompletionEvent(devno());
            int parent_mem_stream = GetStream(task, parent);
            _event_debug(" WaitForEvent parent mem:%lu  dmem:%lu mem_stream:%d, event:%p dev[%d][%s] event:%p",
                   parent->uid(), mem->uid(), parent_mem_stream, event, devno(), name(), event); 
            WaitForEvent(event, parent_mem_stream, iris_event_wait_default);
        }
    }
    else if (!Platform::GetPlatform()->is_d2d_disabled() && cpu_dev >= 0 &&
            Platform::GetPlatform()->device(cpu_dev)->model() != iris_opencl) {
        // Handling O2D data transfer
        // You didn't find data in peer device, 
        // but you found it in neighbouring CPU (OpenMP) device.
        // Fetch it through H2D
        _event_debug("explore Device(OpenMP)2Device (O2D) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        Device *src_dev = Platform::GetPlatform()->device(cpu_dev);
        size_t off[3] = { 0 };
        size_t host_sizes[3] = { mem->size() };
        size_t dev_sizes[3] = { mem->size() };
        // Though we use H2D command for transfer, 
        // it is still a device to device transfer
        // You do not need offsets as they correspond to host pointer
        int mem_stream = GetStream(task, mem, true);

        ResolveInputWriteDependency<ASYNC_DEV_INPUT_RESOLVE>(task, mem, async, src_dev, src_mem);

        if (async && platform_obj_->is_event_profile_enabled()) {
            ProfileEvent & prof_event = task->CreateProfileEvent(mem, src_dev->devno(), PROFILE_O2D, this, mem_stream);
            prof_event.RecordStartEvent(); 
        }

        void* src_arch = src_mem->arch(src_dev);
        double start = timer_->Now();
        if (!platform_obj_->is_data_transfers_disabled())
            MemH2D(task, mem, off, host_sizes, dev_sizes, 1, 1, mem->size(), src_arch, "OpenMP2DEV ");
        double end = timer_->Now();
        if (async && platform_obj_->is_event_profile_enabled()) {
            ProfileEvent & prof_event = task->LastProfileEvent();
            prof_event.RecordEndEvent(); 
        }
        if (async) {
            mem->RecordEvent(devno(), mem_stream, true);
        }
        o2dtime = end - start;
        if (!async && kernel != NULL && kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_o2d, (uint32_t) cpu_dev, (uint32_t) devno_, start, end});
        }
        o2d_enabled = true;
        if (!async && Platform::GetPlatform()->is_scheduling_history_enabled()){
            string cmd_name = "Internal-O2D(" + string(cmd->task()->name()) + ")-from-" + to_string(src_dev->devno()) + "-to-" + to_string(this->devno());
            Platform::GetPlatform()->scheduling_history()->Add(cmd, cmd_name, "MemO2D", start,end);
        }
        if (parent != NULL && async) {
            void *event = mem->GetCompletionEvent(devno());
            int parent_mem_stream = GetStream(task, parent);
            _event_debug(" WaitForEvent O2D after dmem:%lu mem_stream:%d, event:%p dev[%d][%s]",
                   mem->uid(), mem_stream, event, devno(), name()); 
            WaitForEvent(event, parent_mem_stream, iris_event_wait_default);
        }
    }
    else if (!Platform::GetPlatform()->is_d2d_disabled() && this->model() != iris_opencl && this->type() == iris_cpu && non_cpu_dev >= 0) {
        //D2O Data transfer 
        // You found data in non-CPU/OpenMP device, but this device is CPU/OpenMP
        // Use target D2H transfer 
        //_trace("explore Device2Device(OpenMP) (D2O) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        int mem_stream=0;
        Device *src_dev = Platform::GetPlatform()->device(non_cpu_dev);
        size_t off[3] = { 0 };
        size_t host_sizes[3] = { mem->size() };
        size_t dev_sizes[3] = { mem->size() };
        // Though we use H2D command for transfer, 
        // it is still a device to device transfer
        // You do not need offsets as they correspond to host pointer
        _event_debug("explore Device2Device(OpenMP) (D2O) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        //Create a mem stream on other device
        int src_mem_stream = src_dev->GetStream(task, mem, true);

        src_dev->ResetContext();
        src_dev->ResolveInputWriteDependency<ASYNC_DEV_INPUT_RESOLVE>(task, mem, async, src_dev, src_mem);

        double start = timer_->Now();
        if (async && platform_obj_->is_event_profile_enabled()) {
            ProfileEvent & prof_event = task->CreateProfileEvent(mem, devno(), PROFILE_D2O, src_dev, src_mem_stream);
            prof_event.RecordStartEvent(); 
        }
        void* dst_host = mem->arch(this);
        if (!platform_obj_->is_data_transfers_disabled())
            src_dev->MemD2H(task, src_mem, off, host_sizes, dev_sizes, 1, 1, mem->size(), dst_host, "DEV2OpenMP ");
        double end = timer_->Now();

        if (async && platform_obj_->is_event_profile_enabled()) {
            ProfileEvent & prof_event = task->LastProfileEvent();
            prof_event.RecordEndEvent(); 
        }
        ResetContext(); //This is must. Otherwise, it may still point to src_dev context
        ResolveOutputWriteDependency<ASYNC_D2O_SYNC>(task, mem, async, src_dev);
        if (!async && kernel != NULL && kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2o, (uint32_t) non_cpu_dev, (uint32_t) devno_, start, end});
        }
        d2otime = end - start;
        d2o_enabled = true;
        if (!async && Platform::GetPlatform()->is_scheduling_history_enabled()){
            string cmd_name = "Internal-D2O(" + string(cmd->task()->name()) + ")-from-" + to_string(src_dev->devno()) + "-to-" + to_string(this->devno());
            Platform::GetPlatform()->scheduling_history()->Add(cmd, cmd_name, "MemD2O", start,end);
        }
        if (parent != NULL && async) {
            void *event = mem->GetCompletionEvent(devno());
            int parent_mem_stream = GetStream(task, parent);
            _event_debug(" WaitForEvent parent mem:%lu dmem:%lu mem_stream:%d, event:%p dev[%d][%s]",
                   parent->uid(), mem->uid(), parent_mem_stream, event, devno(), name()); 
            WaitForEvent(event, parent_mem_stream, iris_event_wait_default);
        }
    }
    else if (!src_mem->is_host_dirty()) {
        int mem_stream = GetStream(task, mem, true);
        // H2D (Host 2 Device) transfer
        // If host is not dirty, it is best to transfer from host
        // None of the devices having valid copy or D2D is not possible
        _event_debug("explore Host2Device (H2D) dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        void* host = src_mem->host_memory(); // It should work even if host_ptr is null
        if (src_mem->tmp_host_ptr() != NULL) {
            host = src_mem->tmp_host_ptr();
        }
        if (cmd_host != NULL && host != cmd_host) {
            host = cmd_host;
        }

        ResolveH2DStartEvents(task, mem, async, src_mem);
        mem->arch(this);
        double start = timer_->Now();
        if (!platform_obj_->is_data_transfers_disabled())
            errid_ = MemH2D(task, mem, ptr_off, gws, lws, elem_size, dim, size, host);
        double end = timer_->Now();
        if (cmd_host != NULL && mem->host_ptr() != cmd_host) {
            mem->set_host_dirty(true); // Because we fetched data from some other customized H2D host address
        }

        _event_debug("explore Host2Device (H2D) dev[%d][%s] task[%ld:%s] mem[%lu] q[%d]", devno_, name_, task->uid(), task->name(), mem->uid(), mem_stream);
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);

        ResolveH2DEndEvents(task, mem, async);

        if (!async && kernel != NULL && kernel->is_profile_data_transfers()) {
            kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_h2d, (uint32_t) -1, (uint32_t) devno_, start, end});
        }
        h2dtime = end - start;
        h2d_enabled = true;
        if (!async && Platform::GetPlatform()->is_scheduling_history_enabled()){
            string cmd_name = "Internal-H2D(" + string(cmd->task()->name()) + ")-to-" + to_string(this->devno());
            Platform::GetPlatform()->scheduling_history()->Add(cmd, cmd_name, "MemH2D", start,end);
        }
        if (parent != NULL && async) {
            void *event = mem->GetCompletionEvent(devno());
            int parent_mem_stream = GetStream(task, parent);
            _event_debug("WaitForEvent parent mem:%lu dmem:%lu mem_stream:%d, event:%p dev[%d][%s]",
                   parent->uid(), mem->uid(), parent_mem_stream, event, devno(), name()); 
            WaitForEvent(event, parent_mem_stream, iris_event_wait_default);
        }
        _event_debug("Completed H2D");
    }
    else {
        // D2H->H2D case
        // Host doesn't have fresh copy and peer2peer d2d is not possible
        // Fresh copy should be in some other device memory
        // do D2H and follewed by H2D
        //_trace("explore D2H->H2D dev[%d][%s] task[%ld:%s] mem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid());
        int select_src_dev = nddevs[0];
        for(int i=1; nddevs[i] != -1 && select_src_dev != -1; i++) {
            Device *target_dev = Platform::GetPlatform()->device(nddevs[i]);
            if (type() == target_dev->type())
                select_src_dev = nddevs[i]; 
        }
        if (select_src_dev != -1) { //If it is -1, there is no valid data in any place
            int mem_stream = GetStream(task, mem, true); 
#ifndef HALT_UNTIL
            mem_stream = 0; mem->set_recommended_stream(devno(), mem_stream);
#endif
            void* host = mem->host_memory(); // It should work even if host_ptr is null
            Device *src_dev = Platform::GetPlatform()->device(select_src_dev);
            // D2H should be issued from target src (remote) device
            _event_debug("explore D2H->H2D dev[%d][%s] -> dev[%d][%s] task[%ld:%s] mem[%lu]", src_dev->devno(), src_dev->name(), devno(), name(), task->uid(), task->name(), mem->uid());
            int src_mem_stream = -1;
            //#ifndef DIRECT_H2D_SYNC
            //        src_mem_stream = 1; mem->set_recommended_stream(src_dev->devno(), src_mem_stream);
            //#endif
            bool src_async = src_dev->is_async(false);
            //TODO: Think here

            double d2h_start = 0;
            if (src_async) mem->HostWriteLock(src_dev->devno());
            int host_write_dev = mem->GetHostWriteDevice();
            if ((host_write_dev != src_dev->devno())) {
                //((mem->GetHostWriteStream() != src_mem_stream)) 
                src_mem_stream = src_dev->GetStream(task, mem, true); 
                ResolveInputWriteDependency<ASYNC_SAME_DEVICE_DEPENDENCY>(task, mem, src_async, src_dev, src_mem);
                d2h_start = timer_->Now();
                src_dev->ResetContext();
                if (async && src_async && platform_obj_->is_event_profile_enabled()) {
                    ProfileEvent & prof_event = task->CreateProfileEvent(mem, -1, PROFILE_D2HH2D_D2H, src_dev, src_mem_stream);
                    prof_event.RecordStartEvent(); 
                }
                _event_debug("In mem:[%lu] src_mem_stream:%d src_dev:%d dev:%d host_write_dev:%d", mem->uid(), src_mem_stream, src_dev->devno(), devno(), host_write_dev);

                // Do Device to Host Transfer
                if (!platform_obj_->is_data_transfers_disabled())
                    errid_ = src_dev->MemD2H(task, src_mem, ptr_off, 
                            gws, lws, elem_size, dim, size, host, "D2H->H2D(1) ");

                if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
                d2htime = timer_->Now() - d2h_start;
                if (async && src_async && platform_obj_->is_event_profile_enabled()) {
                    ProfileEvent & prof_event = task->LastProfileEvent();
                    prof_event.RecordEndEvent(); 
                }
                if (async && src_async) { 
                    // Source generated data using asynchronous device
                    mem->HostRecordEvent(src_dev->devno(), src_mem_stream);
                    _event_debug("After host write_event:%d stream:%d", mem->GetHostWriteDevice(), mem->GetHostWriteStream());
                }
            }
            else {
                _event_debug("Reuse of mem:[%lu] src_mem_stream:%d src_dev:%d dev:%d\n", mem->uid(), src_mem_stream, src_dev->devno(), devno());
                src_mem_stream = mem->GetHostWriteStream();
            }
            if (src_async) mem->HostWriteUnLock(src_dev->devno());

            ResetContext();
            ResolveInputWriteDependency<ASYNC_KNOWN_H2D_RESOLVE>(task, mem, async, src_dev);
            _event_debug("   MemD2H -> MemH2D registered callbacks completed callbacks");

            // H2D should be issued from this current device
            if (async && platform_obj_->is_event_profile_enabled()) {
                ProfileEvent & prof_event = task->CreateProfileEvent(mem, -1, PROFILE_D2HH2D_H2D, this, mem_stream);
                prof_event.RecordStartEvent(); 
            }
            double start = timer_->Now();
            mem->arch(this);
            if (!platform_obj_->is_data_transfers_disabled())
                errid_ = MemH2D(task, mem, ptr_off, 
                        gws, lws, elem_size, dim, size, host, "D2H->H2D(2) ");
            double end = timer_->Now();
            _event_debug("   MemD2H -> MemH2D done");
            if (async && platform_obj_->is_event_profile_enabled()) {
                ProfileEvent & prof_event = task->LastProfileEvent();
                prof_event.RecordEndEvent(); 
            }
            if (async) {
                mem->RecordEvent(devno(), mem_stream, true);
                //mem->HardDeviceWriteEventSynchronize(this, event); 
            }
            h2dtime = end - start;
            if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
            mem->clear_host_dirty();
            d2h_h2d_enabled = true;
            if (!async && kernel != NULL && kernel->is_profile_data_transfers()) {
                kernel->AddInDataObjectProfile({(uint32_t) cmd->task()->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2h_h2d, (uint32_t) select_src_dev, (uint32_t) devno_, d2h_start, end});
            }
            if (!async && Platform::GetPlatform()->is_scheduling_history_enabled()){
                string cmd_name = "Internal-D2H-H2D(" + string(cmd->task()->name()) + ")-from-" + to_string(src_dev->devno()) + "-to-" + to_string(this->devno());
                Platform::GetPlatform()->scheduling_history()->Add(cmd, cmd_name, "MemD2H_H2D", d2h_start,end);
            }
            if (parent != NULL && async) {
                void *event = mem->GetCompletionEvent(devno());
                int parent_mem_stream = GetStream(task, parent);
                _event_debug("WaitForEvent parent mem:%lu  dmem:%lu mem_stream:%d, event:%p dev[%d][%s]\n",
                        parent->uid(), mem->uid(), parent_mem_stream, event, devno(), name()); 
                WaitForEvent(event, parent_mem_stream, iris_event_wait_default);
            }
        }
    }
    mem->clear_dev_dirty(devno_ );
    mem->dev_unlock(devno_);
    if (h2d_enabled && kernel) kernel->history()->AddH2D(cmd, this, h2dtime, size);
    if (d2d_enabled && kernel) kernel->history()->AddD2D(cmd, this, d2dtime, size);
    if (d2o_enabled && kernel) kernel->history()->AddD2O(cmd, this, d2otime, size);
    if (o2d_enabled && kernel) kernel->history()->AddO2D(cmd, this, o2dtime, size);
    if (d2h_h2d_enabled && kernel) {
        kernel->history()->AddD2H_H2D(cmd, this, d2htime+h2dtime, size);
        //cmd->kernel()->history()->AddD2H(cmd, this, d2htime);
        //cmd->kernel()->history()->AddH2D(cmd, this, h2dtime);
    //if (Platform::GetPlatform()->is_scheduling_history_enabled()) Platform::GetPlatform()->scheduling_history()->AddD2H_H2D(cmd);
    }
}

void Device::ExecuteMemInDMemRegionIn(Task *task, Command* cmd, DataMemRegion *mem) {
    if (mem->is_dev_dirty(devno_)) {
        _trace("Initiating DMEM_REGION data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu]", devno_, name_, task->uid(), task->name(), mem->uid(), mem->get_dmem()->uid());
        InvokeDMemInDataTransfer<DataMemRegion>(task, cmd, mem);
    }
    else{
        _trace("Skipped DMEM_REGION data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu] dptr[%p]", devno_, name_, task->uid(), task->name(), mem->uid(), mem->get_dmem()->uid(), mem->get_arch(devno_));
        //if (is_async(task)) WaitForDataAvailability<DataMemRegion>(devno_, task, mem);
    }
}
void Device::ExecuteMemInDMemIn(Task *task, Command* cmd, DataMem *mem) {
    if (mem->is_regions_enabled()) {
        // If regisions are enabled
        int n_regions = mem->get_n_regions();
        int mem_stream = -1;
        bool async = is_async(task);
        if (async) {
            mem_stream = GetStream(task, mem, true);
        }
        for (int i=0; i<n_regions; i++) {
            DataMemRegion *rmem = (DataMemRegion *)mem->get_region(i);
            if (rmem->is_dev_dirty(devno_)) {
                _event_debug("Initiating DMEM_REGION region(%d) data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu]", i, devno_, name_, task->uid(), task->name(), rmem->uid(), rmem->get_dmem()->uid());
                InvokeDMemInDataTransfer<DataMemRegion>(task, cmd, rmem, mem);
            }
            else {
                _event_debug("Skipped DMEM_REGION region(%d) H2D data transfer dev[%d:%s] task[%ld:%s] dmem_reg[%lu] dmem[%lu] dptr[%p]", i, devno_, name_, task->uid(), task->name(), rmem->uid(), rmem->get_dmem()->uid(), rmem->get_arch(devno_));
                if (async) {
                    int written_stream = rmem->GetWriteStream(devno());
                    void *event = rmem->GetWriteDeviceEvent(devno());
                    if (mem_stream != written_stream) {
                        WaitForEvent(event, mem_stream, iris_event_wait_default);
                        _event_debug(" WaitForEvent parent mem:%lu(non-dirty)  dmem:%lu written_stream:%d mem_stream:%d, event:%p dev[%d][%s]\n",
                               mem->uid(), rmem->uid(), written_stream, mem_stream, event, devno(), name()); 
                    }
                }
                //if (is_async(task)) WaitForDataAvailability<DataMemRegion>(devno_, task, rmem);
            }
        }
        if (async) {
            mem->RecordEvent(devno(), mem_stream, true);
            //mem->SetWriteStream(devno(), mem_stream);
            //mem->SetWriteDevice(devno());
            //mem->SetWriteDeviceEvent(devno(), mem->GetCompletionEvent(devno()));
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
        // Regions are not enabled, but data for device is not valid. 
        // Hence, fetch it either from neightbors or host
        if (mem->child().size() > 0) {
            void *host = mem->host_memory();
            void *tmp_host_ptr = mem->tmp_host_memory();
            memcpy(tmp_host_ptr, host, mem->size());
            for(const auto & child : mem->child()) {
                DataMem *child_mem = child.first;
                size_t offset = child.second;
                void **arch_ptr = (void **)((char *)tmp_host_ptr + offset);
                if (child_mem->get_source_mem() != NULL)
                    child_mem = child_mem->get_source_mem();
                InvokeDMemInDataTransfer<DataMem>(task, cmd, child_mem);
                *arch_ptr = child_mem->arch(devno());
                //printf("parsing child mem:%lu size:%lu parent_tmp_host_ptr:%p parent_host:%p offset:%lu child_dev_arch:%p child_arch_ptr:%p child_host_arch_ptr:%p\n", child_mem->uid(), child_mem->size(), tmp_host_ptr, host, offset, *arch_ptr, arch_ptr, host+offset);
            }
            if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
            InvokeDMemInDataTransfer<DataMem>(task, cmd, mem);
            //printf("----- mem:%lu size:%lu arch:%p host:%p tmp_host_ptr:%p\n", mem->uid(), mem->size(), mem->arch(devno()), host, tmp_host_ptr);
        }
        else {
            if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
            InvokeDMemInDataTransfer<DataMem>(task, cmd, mem);
        }
    }
    else {
        _trace("Skipped DMEM H2D data transfer dev[%d:%s] task[%ld:%s] dmem[%lu] dptr[%p]", devno_, name_, task->uid(), task->name(), mem->uid(), mem->get_arch(devno_));
        //if (is_async(task)) WaitForDataAvailability<DataMem>(devno_, task, mem);
    }
}
void Device::ExecuteMemOut(Task *task, Command* cmd) {
    if (cmd == NULL || cmd->kernel() == NULL) return;
    int *params_map = cmd->get_params_map();
    //int nargs = cmd->kernel_nargs();
    for(auto && z : cmd->kernel()->out_mems()) {
        BaseMem *mem = z.first;
        int idx = z.second;
        if (params_map != NULL && 
                (params_map[idx] & iris_all) == 0 && 
                !(params_map[idx] & type_) ) continue;
        if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
        if (mem->GetMemHandlerType() == IRIS_DMEM ||
                mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
            DataMem *dmem = (DataMem *)mem;
            //printf("ExecuteMemOut mem:%lu dev:%d set_dirty_except:%d host_dirty:True\n", dmem->uid(), devno_, devno_);
            dmem->set_dirty_except(devno_);
            dmem->set_host_dirty();
            dmem->disable_reset();
        }
        if (is_async(task)) {
            int mem_stream = GetStream(task);
            mem->clear_d2h_events();
            mem->clear_streams();
            mem->SetWriteStream(devno(), mem_stream);
            mem->SetWriteDevice(devno());
            void *k_event = cmd->kernel()->GetCompletionEvent();
            mem->SetWriteDeviceEvent(devno(), k_event);
            //mem->RecordEvent(devno(), mem_stream, true); //It should create new entry of event instead of using existing one
            _event_debug("Rewrite back RecordEvent mem set stream   task:[%lu][%s] output dmem:%lu stream:%d, dev:%d event:%p mem_ptr:%p\n", task->uid(), task->name(), mem->uid(), mem_stream, devno(), k_event, mem->get_arch(devno()));
        }
    }
}
void Device::ExecuteMemFlushOut(Command* cmd) {
    int nddevs[IRIS_MAX_NDEVS+1];
    BaseMem* bmem = (BaseMem *)cmd->mem();
    if (bmem->GetMemHandlerType() != IRIS_DMEM) {
        _error("Flush out is called for unsupported memory handler task:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
        return;
    }
    DataMem* mem = (DataMem *)cmd->mem();
    if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
    if (mem->is_host_dirty()) {
        size_t *ptr_off = mem->off();
        size_t *gws = mem->host_size();
        size_t *lws = mem->dev_size();
        size_t elem_size = mem->elem_size();
        int dim = mem->dim();
        size_t size = mem->size();
        //printf("Pointer offset %d, gws %d lws %d elem_size %d dim %d size %d\n", 
            //*ptr_off, *gws, *lws, elem_size, dim, size);
        void* host = mem->host_memory(); // It should work even if host_ptr is null
        bool update_host_flag = true;
        if (cmd->host() != NULL) {
            update_host_flag = false;
            host = cmd->host();
        }
        double start = timer_->Now();
        Task *task = cmd->task();
        bool async = is_async(task);
        Device *src_dev = this;
        if (mem->is_dev_dirty(devno_)) {
            int cpu_dev = -1;
            int non_cpu_dev = -1;
            int d2d_dev = -1;
            GetPossibleDevices(mem, devno_, mem->get_non_dirty_devices(nddevs), 
                    d2d_dev, cpu_dev, non_cpu_dev, async);
            _event_debug(" Flushing out mem:%lu dev:[%d][%s] src_dev:%d:%s\n", mem->uid(), devno(), name(), src_dev->devno(), src_dev->name());
            src_dev = Platform::GetPlatform()->device(nddevs[0]);
            // D2H should be issued from target src (remote) device
            int src_mem_stream = src_dev->GetStream(task, mem, true);
            bool src_async = src_dev->is_async(false);

            if (src_async) mem->HostWriteLock(src_dev->devno());
            if ((mem->GetHostWriteDevice() != src_dev->devno())) {
                ResolveInputWriteDependency<ASYNC_DEV_INPUT_RESOLVE>(task, mem, async, src_dev);

                if (async && src_async && platform_obj_->is_event_profile_enabled()) {
                    ProfileEvent & prof_event = task->CreateProfileEvent(mem, -1, PROFILE_D2H, src_dev, src_mem_stream);
                    prof_event.RecordStartEvent(); 
                }
                src_dev->ResetContext();
                if (!platform_obj_->is_data_transfers_disabled())
                    errid_ = src_dev->MemD2H(task, mem, ptr_off, 
                            gws, lws, elem_size, dim, size, host, "MemFlushOut ");
                if (async && src_async && platform_obj_->is_event_profile_enabled()) {
                    ProfileEvent & prof_event = task->LastProfileEvent();
                    prof_event.RecordEndEvent(); 
                }
                if (async && src_async) mem->HostRecordEvent(src_dev->devno(), src_mem_stream);
            }
            else {
                src_mem_stream = mem->GetHostWriteStream();
            }
            if (src_async) mem->HostWriteUnLock(src_dev->devno());
            //TODO: Shouldn't task call back from the source device
            if (async && src_async) {
                _event_debug("Flush mem:%lu dev:[%d][%s] host recorded src_dev:%d:%s task:%lu:%s and can wait for event:%p mem_stream:%d\n", mem->uid(), devno(), name(), src_dev->devno(), src_dev->name(), task->uid(), task->name(), mem->GetHostCompletionEvent(), src_mem_stream);
#if 0
                void *event = NULL;
                src_dev->CreateEvent(&event, iris_event_disable_timing);
                src_dev->RecordEvent(&event, src_mem_stream);
                src_dev->EventSynchronize(event);
                src_dev->DestroyEvent(event);
#endif
                task->set_last_cmd_stream(src_mem_stream);
                task->set_last_cmd_device(src_dev);
            }
            ResetContext();
        }
        else {
            int mem_stream = -1;
            if (async) mem->HostWriteLock(devno());
            int host_write_dev = mem->GetHostWriteDevice();
            if ((host_write_dev != devno())) {
                //((mem->GetHostWriteStream() != src_mem_stream)) 
                // This should be same device and has valid copy
                mem_stream = GetStream(task, mem, true);
                _event_debug(" Flushing out mem:%lu dev:[%d][%s] HostWrite:%d", mem->uid(), devno(), name(), host_write_dev);

                ResolveInputWriteDependency<ASYNC_DEV_INPUT_RESOLVE>(task, mem, async, this);
                ResetContext();
                if (async && platform_obj_->is_event_profile_enabled()) {
                    ProfileEvent & prof_event = task->CreateProfileEvent(mem, -1, PROFILE_D2H, this, mem_stream);
                    prof_event.RecordStartEvent(); 
                }
                if (!platform_obj_->is_data_transfers_disabled())
                    errid_ = MemD2H(task, mem, ptr_off, 
                            gws, lws, elem_size, dim, size, host, "MemFlushOut ");
                if (async && platform_obj_->is_event_profile_enabled()) {
                    ProfileEvent & prof_event = task->LastProfileEvent();
                    prof_event.RecordEndEvent(); 
                }
                if (async) {
                    mem->HostRecordEvent(devno(), mem_stream);
                }
                _event_debug("After flush host write_event:%d stream:%d", mem->GetHostWriteDevice(), mem->GetHostWriteStream());
            }
            else {
                mem_stream = mem->GetHostWriteStream();
            }
            if (async) mem->HostWriteUnLock(devno());
            if (async) {
                //mem->HostRecordEvent(devno(), mem_stream);
                //mem->SetHostWriteDevice(devno());
                //mem->SetHostWriteStream(mem_stream);
                task->set_last_cmd_stream(mem_stream);
                task->set_last_cmd_device(this);
                _event_debug("Flush recorded event dev:%d mem:%lu task:%lu:%s wait for event:%p mem_stream:%d\n", devno(), mem->uid(), task->uid(), task->name(), mem->GetHostCompletionEvent(), mem_stream);
            }
        }
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        if (update_host_flag)
            mem->clear_host_dirty();
        double end = timer_->Now();
        double d2htime = end - start;
        Command* cmd_kernel = cmd->task()->cmd_kernel();
        if (cmd_kernel) cmd_kernel->kernel()->history()->AddD2H(cmd_kernel, src_dev, d2htime, size);
        else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, d2htime, size);
        if (task->is_profile_data_transfers()) {
            task->AddOutDataObjectProfile({(uint32_t) task->uid(), (uint32_t) mem->uid(), (uint32_t) iris_dt_d2h_h2d, (uint32_t) devno_, (uint32_t) -1, start, end});
        }
        if (!async && Platform::GetPlatform()->is_scheduling_history_enabled()){
          //TODO: clean up
          string cmd_name = "Internal-D2H(" + string(cmd->task()->name()) + ")-from-" + to_string(src_dev->devno());// + "-to-" + string(this->name());
          cmd->task()->set_dev(src_dev);
          Platform::GetPlatform()->scheduling_history()->Add(cmd, cmd_name, "MemFlushOut", start,end);
        }

    }
    else {
        _trace("MemFlushout is skipped as host already having valid data for task:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
    }
}
#ifdef AUTO_PAR
#ifdef AUTO_SHADOW
void Device::ExecuteMemFlushOutToShadow(Command* cmd) {
    int nddevs[IRIS_MAX_NDEVS+1];
    BaseMem* bmem = (BaseMem *)cmd->mem();
    if (bmem->GetMemHandlerType() != IRIS_DMEM) {
        _error("Shadow Flush out is called for unssuported memory handler task:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
        return;
    }
    DataMem* mem = (DataMem *)cmd->mem();
    if (mem->get_source_mem() != NULL) mem = mem->get_source_mem();
    //if (mem->is_host_dirty()) {
    size_t *ptr_off, *gws, *lws, elem_size, size;
    int dim;
    if(mem->get_is_shadow() == false) {
        ptr_off = mem->get_current_dmem_shadow()->off();
        gws = mem->get_current_dmem_shadow()->host_size();
        lws = mem->get_current_dmem_shadow()->dev_size();
        elem_size = mem->get_current_dmem_shadow()->elem_size();
        dim = mem->get_current_dmem_shadow()->dim();
        size = mem->get_current_dmem_shadow()->size();
    }
    else {
        ptr_off = mem->get_main_dmem()->off();
        gws = mem->get_main_dmem()->host_size();
        lws = mem->get_main_dmem()->dev_size();
        elem_size = mem->get_main_dmem()->elem_size();
        dim = mem->get_main_dmem()->dim();
        size = mem->get_main_dmem()->size();
    }

    //printf("Pointer offset %d, gws %d lws %d elem_size %d dim %d size %d\n", 
            //*ptr_off, *gws, *lws, elem_size, dim, size);
    //void* host = mem->host_memory(); // It should work even if host_ptr is null
    void* host;
    if(mem->get_is_shadow() == false)
        host = mem->get_current_dmem_shadow()->host_memory(); // get the host of shadow 
    else 
        host = mem->get_main_dmem()->host_memory(); // get the host of main 
    //void* host = mem->get_host_shadow_ptr(); // It is getting the shadow host pointer 
    
    /*std::cout << "Before " <<  cmd->task()->name()<< " : ";
    for(int i = 0; i < 32; i=i+8){
        std::cout << *((double*) (host + i)) << " "; 
    }
    std::cout << std::endl; */
   
    double start = timer_->Now();
    Task *task = cmd->task();
    Device *src_dev = this;
    bool async = is_async(task);
    if (mem->is_dev_dirty(devno_)) {
        int cpu_dev = -1;
        int non_cpu_dev = -1;
        int d2d_dev = -1;
        GetPossibleDevices(mem, devno_, mem->get_non_dirty_devices(nddevs), 
                    d2d_dev, cpu_dev, non_cpu_dev, async);
        src_dev = Platform::GetPlatform()->device(nddevs[0]);
        // D2H should be issued from target src (remote) device
        if (!platform_obj_->is_data_transfers_disabled())
            errid_ = src_dev->MemD2H(task, mem, 
                    ptr_off, gws, lws, elem_size, dim, size, host, "MemShadowFlushOut ");
                //ptr_off, gws, lws, elem_size, dim, size, host, "MemShadowFlushOut ");
        ResetContext();
    }
    else {
        if (!platform_obj_->is_data_transfers_disabled())
            errid_ = MemD2H(task, mem, 
                    ptr_off, gws, lws, elem_size, dim, size, host, "MemShadowFlushOut ");
                //ptr_off, gws, lws, elem_size, dim, size, host, "MemShadowFlushOut ");
    }
    if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
    //mem->get_current_dmem_shadow()->clear_host_shadow_dirty();
    if(mem->get_is_shadow() == false)
        mem->get_current_dmem_shadow()->clear_host_dirty();
    else 
        mem->get_main_dmem()->clear_host_dirty();
    //std::cout << "shadow flush host pointer " << host << std::endl;
    //mem->clear_host_shadow_dirty();
    //// Now need to set the shadow dmem to clear the host dirty
    //need to create map both in Task and AutoDAG to recover the shadow object associated with this
    double d2htime = timer_->Now() - start;
    Command* cmd_kernel = cmd->task()->cmd_kernel();
    if (cmd_kernel) cmd_kernel->kernel()->history()->AddD2H(cmd_kernel, src_dev, d2htime, size);
    else Platform::GetPlatform()->null_kernel()->history()->AddD2H(cmd, this, d2htime, size);
    /*}
    else {
        std::cout << " I am Here -----------------"  << std::endl;
        memcpy(mem->get_current_dmem_shadow()->host_memory(), mem->host_memory(), mem->size());
        _trace("MemShadowFlushout is copying host to host of shadow:%ld:%s\n", cmd->task()->uid(), cmd->task()->name());
    }*/
    /*if(mem->get_is_shadow() == false)
        mem->get_current_dmem_shadow()->clear_host_dirty();
    else 
        mem->get_main_dmem()->clear_host_dirty();*/
    //void* p = mem->get_current_dmem_shadow()->host_memory(); 
    //void* q = mem->host_memory(); 
    /*std::cout << "After " <<  cmd->task()->name()<< " : ";
    for(int i = 0; i < 32; i=i+8){
        std::cout << *((double*) (host + i)) << " "; 
    }
    std::cout << std::endl; */
}
#endif
#endif
void Device::ExecuteH2BroadCast(Command *cmd) {
    int ndevs = Platform::GetPlatform()->ndevs();
    for(int i=0; i<ndevs; i++) {
        Device *src_dev = Platform::GetPlatform()->device(i);
        ExecuteH2D(cmd, src_dev);
    }
    ResetContext();
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
    cmd->set_time_start(timer_);
    double start = timer_->Now();
    if (src_dev->type() == type()) {
        // Invoke D2D
        void* dst_arch = mem->arch(this);
        void* src_arch = mem->arch(src_dev);
        MemD2D(cmd->task(), src_dev, mem, dst_arch, src_arch, mem->size());
    }
    else {
        void* host = mem->host_inter(); // It should work even if host_ptr is null
        // D2H should be issued from target src (remote) device
        errid_ = src_dev->MemD2H(cmd->task(), mem, ptr_off, 
                gws, lws, elem_size, dim, size, host, "D2H->H2D(1) ");
        ResetContext();
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
        // H2D should be issued from this current device
        errid_ = MemH2D(cmd->task(), mem, ptr_off, 
                gws, lws, elem_size, dim, size, host, "D2H->H2D(2) ");
        if (errid_ != IRIS_SUCCESS) _error("iret[%d]", errid_);
    }
    cmd->set_time_end(timer_);
    double end = timer_->Now();
    cmd->SetTime(end-start);
    Kernel *kernel = Platform::GetPlatform()->null_kernel();
    Command *cmd_kernel = cmd->task()->cmd_kernel();
    if (cmd_kernel != NULL) 
        kernel = cmd_kernel->kernel();
    if (src_dev->type() == type()) {
        kernel->history()->AddD2D(cmd, this, cmd->time_end(), size);
    }
    else {
        kernel->history()->AddD2H_H2D(cmd, this, cmd->time_end(), size);
    }
}
void Device::ExecuteH2D(Command* cmd, Device *dev) {
  if (dev == NULL) dev = this;
  void* host = cmd->host();
  BaseMem* dmem = (BaseMem *)cmd->mem();
  //we're using datamem so there is no need to execute this memory transfer
  if (dmem->GetMemHandlerType() == IRIS_DMEM) {
    ExecuteMemInDMemIn(cmd->task(), cmd, (DataMem *)dmem);
    return;
  }
  if (dmem->GetMemHandlerType() == IRIS_DMEM_REGION) {
    ExecuteMemInDMemRegionIn(cmd->task(), cmd, (DataMemRegion *)dmem);
    return;
  }
  Mem* mem = cmd->mem();
  size_t off = cmd->off(0);
  size_t *ptr_off = cmd->off();
  size_t *gws = cmd->gws();
  size_t *lws = cmd->lws();
  size_t elem_size = cmd->elem_size();
  int dim = cmd->dim();
  size_t size = cmd->size();
  bool exclusive = cmd->exclusive();

  if (exclusive) mem->SetOwner(off, size, this);
  else mem->AddOwner(off, size, this);
  timer_->Start(IRIS_TIMER_H2D);
  Task *task = cmd->task();
  bool async = is_async(task);
  //ResolveH2DStartEvents(task, mem, async);
  int mem_stream = -1;
  if (async) {
      mem_stream = dev->GetStream(task, mem, true);
      //_event_debug(" before dev:[%d][%s] mem:%ld task:%lu:%s stream:%d", devno(), name(), mem->uid(), task->uid(), task->name(), mem_stream);
      SynchronizeInputToMemory(task, mem);
  }
  //_event_debug(" second dev:[%d][%s] mem:%ld task:%lu:%s stream:%d", devno(), name(), mem->uid(), task->uid(), task->name(), mem_stream);
  cmd->set_time_start(timer_);
  errid_ = dev->MemH2D(cmd->task(), mem, ptr_off, gws, lws, elem_size, dim, size, host);
  if (errid_ != IRIS_SUCCESS) _error("iret[%d] dev[%d][%s]", errid_, dev->devno(), dev->name());
  cmd->set_time_end(timer_);
  //ResolveH2DEndEvents(task, mem, async);
  if (async) {
    void *event = NULL;
    //_event_debug(" dev:[%d][%s] mem:%ld task:%lu:%s stream:%d", devno(), name(), mem->uid(), task->uid(), task->name(), mem_stream);
    dev->CreateEvent(&event, iris_event_disable_timing);
    dev->RecordEvent(&event, mem_stream);
    dev->EventSynchronize(event);
    dev->DestroyEvent(event);
  }
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
  if (dmem && (dmem->GetMemHandlerType() == IRIS_DMEM ||
              dmem->GetMemHandlerType() == IRIS_DMEM_REGION)) {
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
  cmd->set_time_start(timer_);
  errid_ = IRIS_SUCCESS;

  Task *task = cmd->task();
  bool async = is_async(task);
  int mem_stream = -1;
  if (async) {
      mem_stream = GetStream(task, mem, true);
      SynchronizeInputToMemory(task, mem);
  }
  if (mode & iris_reduction) {
    errid_ = MemD2H(cmd->task(), mem, ptr_off, gws, lws, elem_size, dim, mem->size() * expansion, mem->host_inter());
    Reduction::GetInstance()->Reduce(mem, host, size);
  } else errid_ = MemD2H(cmd->task(), mem, ptr_off, gws, lws, elem_size, dim, size, host);
  if (errid_ != IRIS_SUCCESS) _error("iret[%d] dev[%d][%s]", errid_, devno(), name());
  //printf("D2H mem_stream:%d\n", mem_stream);
  if (async) {
    void *event = NULL;
    CreateEvent(&event, iris_event_disable_timing);
    RecordEvent(&event, mem_stream);
    EventSynchronize(event);
    DestroyEvent(event);
  }
  cmd->set_time_end(timer_);
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
  string kernel_name = string(kernel->name());
  func(*(cmd->task()->struct_obj()), params, (char *)kernel_name.c_str());
  Kernel *selected_kernel = Platform::GetPlatform()->GetKernel(kernel_name.c_str());
  selected_kernel->set_task(kernel->task());
  return selected_kernel;
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
int Device::RegisterCallback(int stream, CallBackType callback_fn, void *data, int flags)
{
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    Utils::PrintStackTrace();
    return IRIS_ERROR;
}
void Device::CreateEvent(void **event, int flags) { 
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    Utils::PrintStackTrace();
    //CPUEvent *levent = new CPUEvent();
    //*event = levent;
}
void Device::RecordEvent(void **event, int stream, int event_creation_flag) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    Utils::PrintStackTrace();
    //CPUEvent *levent = (CPUEvent *)event;
    //levent->Record();
}
void Device::WaitForEvent(void *event, int stream, int flags) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    Utils::PrintStackTrace();
    //CPUEvent *levent = (CPUEvent *)event;
    //levent->Wait();
}
void Device::DestroyEvent(BaseMem *mem, void *event) {
    _event_debug(" Mem:%ld event:%p destroy now", mem->uid(), event);
    DestroyEvent(event);
}
void Device::DestroyEvent(void *event) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    Utils::PrintStackTrace();
    //CPUEvent *levent = (CPUEvent *)event;
    //delete levent;
}
void Device::EventSynchronize(void *event) {
    _error("Device:%d:%s Invalid function call!", devno_, name()); 
    worker_->platform()->IncrementErrorCount();
    Utils::PrintStackTrace();
}

} /* namespace rt */
} /* namespace iris */


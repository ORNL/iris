#include <string>
#include "Task.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Kernel.h"
#include "Mem.h"
#include "DataMem.h"
#include "Pool.h"
#include "Scheduler.h"
#include "Timer.h"
#include "Worker.h"
#include "AutoDAG.h"

namespace iris {
namespace rt {

void ProfileEvent::Clean() {
    //printf("prof_event trying to destroy :%p %p\n", &start_event_, start_event_);
    if (start_event_ != NULL && event_dev_ != NULL) 
        event_dev_->EnableDestroyEvent(start_event_);
    if (end_event_ != NULL && event_dev_ != NULL) 
        event_dev_->EnableDestroyEvent(end_event_);
    //printf("completed prof_event trying to destroy :%p %p\n", &start_event_, start_event_);
}
float ProfileEvent::GetStartTime() {
    if (!event_fetch_flag_) return start_time_; 
    float etime =0.0;
    if (start_event_ != NULL && event_dev_ != NULL) 
        etime = event_dev_->GetEventTime(start_event_, stream_);
    return etime;
}
float ProfileEvent::GetEndTime() {
    if (!event_fetch_flag_) return end_time_; 
    float etime =0.0;
    if (end_event_ != NULL && event_dev_ != NULL) 
        etime = event_dev_->GetEventTime(end_event_, stream_);
    return etime;

}
void ProfileEvent::RecordStartEvent() {
    event_dev_->RecordEvent(&start_event_, stream_, iris_event_default);
    event_dev_->TrackDestroyEvent(start_event_);
    _event_prof_debug("Recorded id:%lu type:%d Start event:%p stream:%d", id_, type_, start_event_, stream_);
}
void ProfileEvent::RecordEndEvent() {
    event_dev_->RecordEvent(&end_event_, stream_, iris_event_default);
    event_dev_->TrackDestroyEvent(end_event_);
    _event_prof_debug("Recorded id:%lu type:%d End event:%p stream:%d", id_, type_, end_event_, stream_);
}
Task::Task(Platform* platform, int type, const char* name, int max_cmds) {
  j_policy_ = "";
  j_policy_flag_ = false;
  cmds_.reserve(16);
  //printf("Creating task:%lu:%s ptr:%p\n", uid(), name, this);
  is_kernel_launch_disabled_ = false;
  enable_julia_if_ = false;
  julia_kernel_type_ = iris_julia_native;
  type_ = type;
  recommended_stream_ = -1;
  recommended_dev_ = -1;
  max_cmds_ = (max_cmds < IRIS_TASK_MAX_CMDS) ? IRIS_TASK_MAX_CMDS : max_cmds;
  ncmds_ = 0;
  disable_consistency_ = false;
  //cmds_ = new Command *[max_cmds_];
  cmd_kernel_ = NULL;
  cmd_last_ = NULL;
  stream_policy_ = STREAM_POLICY_DEFAULT;
  platform_ = platform;
  if (platform) scheduler_ = platform_->scheduler();
  dev_ = NULL;
  parent_ = 0;
  parent_exist_ = false;
  subtasks_complete_ = 0;
  sync_ = false;
  time_ = 0.0;
  time_start_ = 0.0;
  time_end_ = 0.0;
  ns_time_submit_ = 0;
  ns_time_start_ = 0;
  ns_time_end_ = 0;
  last_cmd_stream_ = -1;
  last_cmd_device_ = NULL;
  user_ = false;
  system_ = false;
  internal_memory_transfer_ = false;
  arch_ = NULL;
  meta_data_[0] = meta_data_[1] = meta_data_[2] = meta_data_[3] = -1;
  //parent
  depends_uids_ = NULL;
  depends_max_ = 1024;
  ndepends_ = 0;
  // childs
  childs_uids_ = NULL;
  childs_max_ = 1024;
  nchilds_ = 0;

  given_name_ = name != NULL;
  profile_data_transfers_ = false;
  name_="";
  if (name) name_ = std::string(name);
  status_ = IRIS_NONE;

  pthread_mutex_init(&stream_mutex_, NULL);
  pthread_mutex_init(&mutex_pending_, NULL);
  pthread_mutex_init(&mutex_executable_, NULL);
  //pthread_mutex_init(&mutex_complete_, NULL);
  pthread_mutex_init(&mutex_subtasks_, NULL);
  //pthread_cond_init(&complete_cond_, NULL);
  brs_policy_ = iris_default;
#ifdef AUTO_PAR
#ifdef AUTO_FLUSH
  graph_ = NULL;
#endif
#ifdef AUTO_SHADOW
  shadow_dep_added_ = false;
#endif
#endif
  df_scheduling_ = false;
  set_object_track(Platform::GetPlatform()->task_track_ptr());
  platform_->task_track().TrackObject(this, uid());
  async_execution_ = platform_->is_async(); // Same as platform by default
  _debug2("Task created %lu %s %p ref_cnt:%d\n", uid(), name_.c_str(), this, ref_cnt());
  _trace("Task created %lu %s %p ref_cnt:%d", uid(), name_.c_str(), this, ref_cnt());
}

Task::~Task() {
  //printf("Releasedd task:%lu:%s released ptr:%p ref_cnt:%d\n", uid(), name(), this, ref_cnt());
  //Platform::GetPlatform()->task_track().UntrackObject(this, uid());
  _trace("Task deleted %lu %s %p ref_cnt:%d", uid(), name(), this, ref_cnt());
  //printf("Trying to delete profile events\n");
  for (ProfileEvent & p : profile_events_) {
    p.Clean();
  }
  //profile_events_.clear();
  for (int i = 0; i < ncmds_; i++) delete cmds_[i];
  cmds_.clear();
  //delete [] cmds_;
  if (depends_uids_) delete [] depends_uids_;
  pthread_mutex_destroy(&stream_mutex_);
  pthread_mutex_destroy(&mutex_pending_);
  pthread_mutex_destroy(&mutex_executable_);
  //pthread_mutex_destroy(&mutex_complete_);
  pthread_mutex_destroy(&mutex_subtasks_);
  //pthread_cond_destroy(&complete_cond_);
  subtasks_.clear();
  if (childs_uids_ != NULL) delete [] childs_uids_;
  _trace("released task:%lu:%s released ref_cnt:%d", uid(), name(), ref_cnt());
  //_printf("released task:%lu:%s released ref_cnt:%d", uid(), name(), ref_cnt());
}
bool Task::IsKernelSupported(Device *dev)
{
    Command *cmd = cmd_kernel();
    if (cmd == NULL) return true;
    Kernel *kernel = cmd->kernel();
    if (kernel == NULL) return true;
    if (kernel->isSupported(dev)) return true;
    return false;
}
double Task::TimeInc(double t) {
  time_ += t;
  return time_;
}

void Task::set_name(const char* name) {
  name_ = name;
  given_name_ = true;
}
void Task::set_name(std::string name) {
  name_ = name;
  given_name_ = true;
}

void Task::set_parent(Task* task) {
  parent_exist_ = true;
  parent_ = task->uid();
}

void Task::set_brs_policy(int brs_policy) {
  if(df_scheduling_) return;
  //printf("task %s old dev %d, new dev %d\n", name(), brs_policy_, brs_policy);
  brs_policy_ = brs_policy;// == iris_default ? platform_->device_default() : brs_policy;
  if(ncmds_ <= 1){//iris_pending policy only works on single command tasks
    if (brs_policy == iris_pending) status_ = IRIS_PENDING;
    if (brs_policy != iris_pending && status_ == IRIS_PENDING) status_ = IRIS_NONE;
  }
  if (!HasSubtasks()) return;
  for (std::vector<Task*>::iterator I = subtasks_.begin(), E = subtasks_.end(); I != E; ++I)
    (*I)->set_brs_policy(brs_policy);
}

const char* Task::brs_policy_string() {
  switch(brs_policy_){
    case iris_ftf: return ("ftf");
    case iris_data: return ("data");
    case iris_default: return ("default");
    case iris_depend: return ("depend");
    case iris_sdq: return ("sdq");
    case iris_pending: return ("pending");
    case iris_profile: return ("profile");
    case iris_random: return ("random");
    case iris_roundrobin: return ("roundrobin");
    case iris_julia_policy: return ("julia_policy");
    default: return("unknown");
  }
}

const char* Task::task_status_string() {
  switch(status_){
    case IRIS_COMPLETE:   return("complete");
    case IRIS_RUNNING:    return("running");
    case IRIS_SUBMITTED:  return("submitted");
    case IRIS_QUEUED:     return("queued");
    case IRIS_NONE:       return("none");
    case IRIS_PENDING:    return("pending");
    default: return("unknown");
  }
}

void Task::set_opt(const char* opt) {
  if (!opt) return;
  if (std::string(opt) == opt_) return;
  opt_ = std::string(opt);
  //memset(opt_, 0, sizeof(opt_));
  //strncpy(opt_, opt, strlen(opt)+1);
}

void Task::AddMemResetCommand(Command* cmd) {
  reset_mems_.push_back(cmd);
}

bool Task::is_task_with_single_flush() {
    if (ncmds_ == 1 && cmds_[0]->type() == IRIS_CMD_MEM_FLUSH) {
        return true;
    }
    return false;
}
int Task::get_device_affinity() {
    if (is_task_with_single_flush()) {
        BaseMem* mem = (BaseMem *)cmds_[0]->mem();
        return mem->get_dev_affinity();
    }
    return -1;
}
void Task::AddCommand(Command* cmd) {
  //if (ncmds_ >= max_cmds_) _error("ncmds[%d]", ncmds_);
  //cmds_[ncmds_++] = cmd;
  cmds_.push_back(cmd); ncmds_++;
  if (cmd->type() == IRIS_CMD_KERNEL) {
    //if (cmd_kernel_) _error("kernel[%s] is already set", cmd->kernel()->name());
    if (!given_name_) {
      name_ = cmd->kernel()->name();
      given_name_ = true;
    }
    if (cmd_kernel_==NULL) cmd_kernel_ = cmd;
  }
  if (!system_ &&
     (cmd->type() == IRIS_CMD_KERNEL || cmd->type() == IRIS_CMD_H2D ||
      cmd->type() == IRIS_CMD_H2DNP  || cmd->type() == IRIS_CMD_D2H ||
      cmd->type() == IRIS_CMD_MEM_FLUSH ))
    cmd_last_ = cmd;
}

void Task::print_incomplete_tasks()
{
  if (depends_uids_ == NULL) return;
  printf("Task Name: %ld:%s\n", uid(), name());
  for (int i = 0; i < ndepends_; i++) {
    Task *dep= depend(i);
    if (dep!= NULL) 
        printf("      Running dependent task: %d:%ld:%ld:%s Status:%s object exists task:%p\n", i, depends_uids_[i], dep->uid(), dep->name(), dep->task_status_string(), dep);
    else
        printf("      Running dependent task: %d:%ld object not exists\n", i, depends_uids_[i]);
  }
}

void Task::ClearCommands() {
  for (int i = 0; i < ncmds_; i++) delete cmds_[i];
  cmds_.clear();
  ncmds_ = 0;
}

bool Task::Dispatchable() {
  //if we, or the tasks we depend on are pending (or not-complete), we can't run.
  _debug2("Checking task:%lu:%s dispatchable depends_uids_:%p ndepends_:%d pending:%d", uid(), name(), depends_uids_, ndepends_, (status_ == IRIS_PENDING));
  //print_incomplete_tasks();
  if (status_ == IRIS_PENDING) return false;
  if (depends_uids_ == NULL) return true;
  bool has_cmds = (ncmds() > 0) ? true: false;
  _event_prof_debug("Checking task:%lu:%s dispatchable depends_uids_:%p ndepends_:%d pending:%d has_cmds:%d", uid(), name(), depends_uids_, ndepends_, (status_ == IRIS_PENDING), has_cmds);
  for (int i = 0; i < ndepends_; i++) {
      Task *dep = platform_->get_task_object(depends_uids_[i]);
      //printf("dep:%p dep async:%d dep_status:%d,%d dep->status():%d IRIS_COMPLETE:%d IRIS_RUNNING:%d\n", dep, dep->is_async(), dep->status() == IRIS_COMPLETE, dep->status() == IRIS_RUNNING, dep->status(), IRIS_COMPLETE, IRIS_RUNNING);
#if 1
      if ( dep != NULL && dep->is_async() && has_cmds && !(dep->status() == IRIS_COMPLETE || dep->status() == IRIS_SUBMITTED)) return false;
      if ( dep != NULL && dep->is_async() && !has_cmds && dep->status() != IRIS_COMPLETE) return false;
      if ( dep != NULL && !dep->is_async() && dep->status() != IRIS_COMPLETE) return false;
#else
      if ( dep != NULL && dep->status() != IRIS_COMPLETE) return false;
#endif
  }
  _debug2("Task task:%lu:%s is ready to run", uid(), name());
  return true;
}

//dispatch all of this tasks pending dependencies on the assigned device.
void Task::DispatchDependencies() {
  //pthread_mutex_lock(&mutex_pending_);
  if (status_ == IRIS_PENDING) set_status_none(); 
  for (int i = 0; i < ndepends_; i++) {
      Task *dep= depend(i);
      if (dep!= NULL) {
          _trace("      Dispatch dependencies task: %d:%ld:%ld:%s Status:%s object exists\n", i, depends_uids_[i], dep->uid(), dep->name(), dep->task_status_string());
          if (dep->status() == IRIS_PENDING) 
              dep->set_status_none();
      }
  }
  //pthread_mutex_unlock(&mutex_pending_);
}

bool Task::Executable() {
  _trace("Task executable check %lu %s %p", uid(), name(), this);
  pthread_mutex_lock(&mutex_executable_);
  if (status_ == IRIS_NONE && subtasks_complete_ == subtasks_.size()) {
    status_ = IRIS_RUNNING;
    pthread_mutex_unlock(&mutex_executable_);
    return true;
  }
  pthread_mutex_unlock(&mutex_executable_);
  return false;
}

void Task::set_status_none() {
  pthread_mutex_lock(&mutex_pending_);
  status_ = IRIS_NONE;
  pthread_mutex_unlock(&mutex_pending_);
}
void Task::set_pending() {
    if (brs_policy_ == iris_pending) {
        pthread_mutex_lock(&mutex_pending_);
        status_ = IRIS_PENDING;
        pthread_mutex_unlock(&mutex_pending_);
    }
}

void Task::Complete() {
  _debug2(" task:%lu:%s is completed ref_cnt:%d", uid(), name(), ref_cnt());
  bool is_user_task = user_;
  bool platform_release_flag = platform_->release_task_flag();
  if (dev_ && type() != IRIS_MARKER) set_devno(dev_->devno());
  status_ = IRIS_COMPLETE;
  std::unique_lock<std::mutex> lock(mutex_complete_cpp_);
  status_ = IRIS_COMPLETE;
  complete_cond_cpp_.notify_all();
  lock.unlock();
  //pthread_mutex_lock(&mutex_complete_);
  //pthread_cond_broadcast(&complete_cond_);
  //pthread_mutex_unlock(&mutex_complete_);
  //printf("calling profile events\n");
  if (user_) platform_->ProfileCompletedTask(this);
  // For task with subtasks, the parent task is not in any worker queue. 
  // However, it has to call the completion of parent task each time.
  // Parent marker task was never go through Worker::Execute.
  if (parent_exist_) parent()->CompleteSub();
  else {
    _debug2(" task:%lu:%s is completed ref_cnt:%d", uid(), name(), ref_cnt());
    if (dev_) dev_->worker()->TaskComplete(this);
    else if (scheduler_) scheduler_->Invoke();
    _debug2(" task:%lu:%s is completed ref_cnt:%d", uid(), name(), ref_cnt());
  }
  //if (!is_user_task) return;
  _debug2(" trying to release task:%lu:%s ref_cnt:%d", uid(), name(), ref_cnt());
  if (platform_release_flag) {
      for (int i = 0; i < ndepends_; i++) {
          Task *dep = platform_->get_task_object(depends_uids_[i]);
          if(dep != NULL && dep->user()) { _debug2(" dep_task:%lu:%s ref_cnt:%d", dep->uid(), dep->name(), dep->ref_cnt()-1); dep->Release(); }
      }
      unsigned long luid = uid(); string lname = name_; _debug2(" task:%lu:%s is completed and trying to release ref_cnt:%d", uid(), name(), ref_cnt());
      int ret_ref_cnt = Release();
      _debug2(" task:%lu:%s is completed and after release ref_cnt:%d", luid, lname.c_str(), ret_ref_cnt);
      //_printf(" Complete: task:%lu:%s ref_cnt:%d", luid, lname.c_str(), ret_ref_cnt);
  }
}

void Task::TryReleaseTask()
{
    for (int i = 0; i < ndepends_; i++) {
        Task *dep = platform_->get_task_object(depends_uids_[i]);
        if(dep != NULL && dep->user()) dep->Release();
    }
    _debug2("task:%lu:%s trying to release here as well ref_cnt:%d", uid(), name(), ref_cnt());
    if (user_) Release();
}

void Task::CompleteSub() {
  Retain();
  pthread_mutex_lock(&mutex_subtasks_);
  if (++subtasks_complete_ == subtasks_.size()) Complete();
  pthread_mutex_unlock(&mutex_subtasks_);
  //_printf(" CompleteSub:: task:%lu:%s is ref_cnt:%d", uid(), name(), ref_cnt());
  Release();
}

void Task::Wait() {
  std::unique_lock<std::mutex> lock(mutex_complete_cpp_);
  complete_cond_cpp_.wait(lock, [this]{ return status_ == IRIS_COMPLETE; });
#if 0
  pthread_mutex_lock(&mutex_complete_);
  _debug2("Waiting for task completion: task:%lu:%s ref_cnt:%d", uid(), name(), ref_cnt());
  if (status_ != IRIS_COMPLETE){
    //either parent (or abandoned) marker tasks can still linger in the waiting queue without ever being assigned a device, should this occur do we assign a device or just abandon the task (by claiming it is complete)?
    pthread_cond_wait(&complete_cond_, &mutex_complete_);
  }

  //if (!platform->is_task_exist(id)) return;
  pthread_mutex_unlock(&mutex_complete_);
  //printf(" task:%lu:%s dependency is clear\n", uid(), name());
#endif
}

void Task::AddSubtask(Task* subtask) {
  subtask->set_parent(this);
  subtask->set_brs_policy(brs_policy_);
  subtasks_.push_back(subtask);
}

bool Task::HasSubtasks() {
  return !subtasks_.empty();
}

// Note: It is possible that the parent task might have been already completed and released. 
// If it is released, it might have been allocated for some other task. 
// Hence, validate the parent task not only from whether object exists or not, but also by
// comparing the actual parent task uid.
void Task::AddDepend(Task* task, unsigned long uid) {
#ifdef AUTO_PAR
#ifdef IGNORE_MANUAL
  if ( task->platform()->get_auto_dag()->get_auto_dep() == false) return;
#endif

#ifdef SANITY_CHECK
  if(std::string(this->name()).find("Graph") == std::string::npos){  
    std::string full_dep = std::string(this->name()) + "->" + std::string(task->name());
    std::cout << full_dep << std::endl;
    if (task->platform()->get_auto_dag()->get_auto_dep() == false){
        task->platform()->get_auto_dag()->add_manual_dep_list(full_dep);
    } else {
        task->platform()->get_auto_dag()->add_auto_dep_list(full_dep);
    }
  }

  task->platform()->get_auto_dag()->extra_dependencies();
  task->platform()->get_auto_dag()->missing_dependencies();
#endif

#endif

  if (depends_uids_ == NULL) {
      depends_uids_ = new unsigned long[depends_max_];
  }
  for (int i = 0; i < ndepends_; i++) if (uid == depends_uids_[i]) return;
  if (ndepends_ == depends_max_ - 1) {
    unsigned long *old_uids = depends_uids_;
    depends_max_ *= 2;
    depends_uids_ = new unsigned long[depends_max_];
    memcpy(depends_uids_, old_uids, ndepends_*sizeof(unsigned long));
    delete [] old_uids;
  } 
  depends_uids_[ndepends_] = uid;
  if(platform_->get_enable_proactive() 
		&& std::string(this->name()).find("Graph") == std::string::npos  
		&& std::string(this->name()).find("flush-out") == std::string::npos ) {
      unsigned long this_uid = this->uid();
      platform_->task_track().CallBackIfObjectExists(uid, 
              [](void *data, void *lhs, void *rhs) {
                Task *parent_task = (Task *)data;
                unsigned long child_uid = *((unsigned long *)rhs);
                parent_task->AddChild(child_uid);
              }, &uid, &this_uid);
  }
  ndepends_++;
}

void Task::AddChild(unsigned long uid) {
  if (childs_uids_ == NULL) {
      childs_uids_ = new unsigned long[childs_max_];
  }
  for (int i = 0; i < nchilds_; i++) if (uid == childs_uids_[i]) return;
  if (nchilds_ == childs_max_ - 1) {
    unsigned long *old_uids = childs_uids_;
    childs_max_ *= 2;
    childs_uids_ = new unsigned long[childs_max_];
    memcpy(childs_uids_, old_uids, nchilds_*sizeof(unsigned long));
    delete [] old_uids;
  }

  childs_uids_[nchilds_] = uid;
  nchilds_++;
  //if (platform_->task_track().IsObjectExists(uid))
  //    task->Retain();
}

// One call to add all the childs by tracking the parents
void Task::AddAllChilds() {
  for (int i = 0; i < ndepends_; i++) 
	platform_->get_task_object(depends_uids_[i])->AddChild(this->uid());
}


#ifdef AUTO_PAR
#ifdef AUTO_FLUSH
void Task::ReplaceDependFlushTask(Task * task) {
    //if (ndepends_ > 1) 
        //_error("Flush task should not have more than one dependency:%ld:%s\n", this->uid(), this->name());

    //platform_->get_task_object(depends_uids_[0])->Release();
    ndepends_ = 0;
    this->AddDepend(task, task->uid());

}
#endif
#endif

/*
void Task::RemoveDepend(Task* task) {
  for (int i = 0; i < ndepends_; i++) 
    if (task == depends_[i]){
      delete depends_[i];
      ndepends_--;
      task->Release();
    }
}
*/




void Task::Submit(int brs_policy, const char* opt, int sync) {
  status_ = IRIS_NONE;
  set_brs_policy(brs_policy);
  //if the submitted task is pending but it is a d2h transfer, then, default to a data movement minimization policy.
  if (brs_policy == iris_pending){
    if (!cmd_last_ || cmd_last_->type_d2h() || cmd_last_->type_memflush()){
      set_brs_policy(iris_data);
    }
  }
  //if we have a non-pending policy, dispatch all pending dependencies.
  if (brs_policy != iris_pending){
    this->DispatchDependencies();
  }
  set_opt(opt);
  sync_ = sync;
  if (cmd_last_) cmd_last_->set_last();
  user_ = true;
  //Retain();
  for (int i=0; i<ndepends_; i++) {
      unsigned long dep_uid = depends_uids_[i];
      _debug2("added dependency task:%lu ", dep_uid);
      platform_->task_track().CallBackIfObjectExists(dep_uid, Task::StaticRetain);
  }
}

Task* Task::Create(Platform* platform, int type, const char* name, int max_cmds) {
  return new Task(platform, type, name, max_cmds);
//  return platform->pool()->GetTask();
}

int Task::Ok(){
  if (dev_) return dev_->ok();
  return IRIS_SUCCESS;
}

int Task::ncmds_kernel() {
  int total = 0;
  for (int i = 0; i < ncmds_; i++) {
    Command* cmd = cmds_[i];
    if (cmd->type_kernel()) total++;
  }
  return total;
}

int Task::ncmds_memcpy() {
  int total = 0;
  for (int i = 0; i < ncmds_; i++) {
    Command* cmd = cmds_[i];
    if (cmd->type_h2d() || cmd->type_h2broadcast() || cmd->type_h2dnp() || cmd->type_d2h()) total++;
  }
  return total;
}

void Task::set_time_start(Timer* d) {
  time_start_ = d->Now();
  ns_time_start_ = d->NowNS();
}

void Task::set_time_end(Timer* d) {
  time_end_ = d->Now();
  ns_time_end_ = d->NowNS();
}

ProfileEvent & Task::CreateProfileEvent(BaseMem *mem, int connect_dev, ProfileRecordType type, Device *dev, int stream) { 
      profile_events_.push_back(ProfileEvent(mem->uid(), connect_dev, type, dev, stream));
      return profile_events_.back();
}

void Task::insert_hidden_dmem(DataMem *dmem, int mode)
{
    if (mode == iris_r) {
        hidden_dmem_in_.push_back(dmem);
    }
    else if (mode == iris_w) {
        hidden_dmem_out_.push_back(dmem);
    }
    else if (mode == iris_rw) {
        hidden_dmem_in_.push_back(dmem);
        hidden_dmem_out_.push_back(dmem);
    }
    else {
        _error("Unknown memory mode dmem:%lu mode:%d\n", dmem->uid(), mode);
    }
}

} /* namespace rt */
} /* namespace iris */

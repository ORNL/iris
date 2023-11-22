#ifndef IRIS_SRC_RT_TASK_H
#define IRIS_SRC_RT_TASK_H

#include "Retainable.h"
#include "Command.h"
#include "Platform.h"
#include "History.h"
#include <pthread.h>
#include <vector>
#include <string>
#include <iostream>

#define IRIS_COMPLETE   0x0
#define IRIS_RUNNING    0x1
#define IRIS_SUBMITTED  0x2
#define IRIS_QUEUED     0x3
#define IRIS_NONE       0x4
#define IRIS_PENDING    0x5

#define IRIS_TASK       0x0
#define IRIS_TASK_PERM  0x1
#define IRIS_MARKER     0x2

namespace iris {
namespace rt {

class Scheduler;
class Graph;

class Task: public Retainable<struct _iris_task, Task> {
public:
  Task(Platform* platform, int type = IRIS_TASK, const char* name = NULL);
  virtual ~Task();

  void AddCommand(Command* cmd);
  void AddMemResetCommand(Command* cmd);
  void ClearCommands();

  void AddSubtask(Task* subtask);
  bool HasSubtasks();

  void AddDepend(Task* task, unsigned long uid);
//  void RemoveDepend(Task* task);

  void Submit(int brs_policy, const char* opt, int sync);

  bool Dispatchable();
  bool Executable();
  void Complete();
  void Wait();
  int Ok();

  double TimeInc(double t);

  int type() { return type_; }
  const char* name() { return name_.c_str(); }
  void set_name(std::string name);
  void set_name(const char* name);
  bool given_name(){return given_name_;}
  bool user() { return user_; }
  void set_user(bool flag=true) { user_ = flag; }
  bool disable_consistency() { return disable_consistency_; }
  bool set_disable_consistency(bool flag=true) { return disable_consistency_ = flag; }
  bool system() { return system_; }
  void set_system() { system_ = true; }
  bool marker() { return type_ == IRIS_MARKER; }
  int status() { return status_; }
  Task* parent() { return platform_->get_task_object(parent_); }
  Command* cmd(int i) { return cmds_[i]; }
  Command* cmd_kernel() { return cmd_kernel_; }
  Command* cmd_last() { return cmd_last_; }
  void TryReleaseTask();
  void set_dev(Device* dev) { dev_ = dev; }
  Platform* platform() { return platform_; }
  Device* dev() { return dev_; }
  void set_devno(int devno) { devno_ = devno; }
  int devno() { return devno_; }
  int ncmds() { return ncmds_; }
  int ncmds_kernel();
  int ncmds_memcpy();
  void set_time_start(double d) { time_start_ = d; }
  void set_time_end(double d) { time_end_ = d; }
  double time() { return time_; }
  double time_start() { return time_start_; }
  double time_end() { return time_end_; }
  void set_parent(Task* task);
  void set_brs_policy(int brs_policy);
  void set_profile_data_transfers(bool flag=true) { profile_data_transfers_ = flag; }
  bool is_profile_data_transfers() { return profile_data_transfers_; }
  void AddOutDataObjectProfile(DataObjectProfile hist) { out_dataobject_profiles.push_back(hist); }
  void ClearMemOutProfile() { out_dataobject_profiles.clear(); }
  vector<DataObjectProfile> & out_mem_profiles() { return out_dataobject_profiles; }
  void set_opt(const char* opt);
  const char* get_opt(){return opt_;}
  char* opt() { return opt_; }
  int brs_policy() { return brs_policy_; }
  int recommended_stream() { return recommended_stream_; }
  void stream_lock() {
    pthread_mutex_lock(&stream_mutex_);
  }
  void stream_unlock() {
    pthread_mutex_unlock(&stream_mutex_);
  }
  void set_recommended_stream(int stream) { recommended_stream_ = stream; }
  int recommended_dev() { return recommended_dev_; }
  void set_recommended_dev(int dev) { recommended_dev_ = dev; }
  const char* brs_policy_string();
  const char* task_status_string();
  bool sync() { return sync_; }
  bool is_async() { return async_execution_; }
  void set_async(bool flag) { async_execution_ = flag; }
  std::vector<Task*>* subtasks() { return &subtasks_; }
  Task* subtask(int i) { return subtasks_[i]; }
  bool is_subtask() { return parent_exist_; }
  bool is_kernel_launch_disabled() { return is_kernel_launch_disabled_; }
  void set_kernel_launch_disabled(bool flag=true) { is_kernel_launch_disabled_ = flag; }
  int ndepends() { return ndepends_; }
  void set_ndepends(int n) { ndepends_ = n; }
  //Task** depends() { return depends_; }
  unsigned long* depends() { return depends_uids_; }
  Task* depend(int i) { return platform_->get_task_object(depends_uids_[i]); }
  void* arch() { return arch_; }
  void set_stream_policy(StreamPolicy policy) { stream_policy_ = policy; }
  StreamPolicy stream_policy() { return stream_policy_; }
  void set_arch(void* arch) { arch_ = arch; }
  void set_pending();
  void set_status_none();
  bool pending() {return status_==IRIS_PENDING;}
  std::vector<Command *> & reset_mems() { return reset_mems_; }
  void DispatchDependencies();
  bool is_internal_memory_transfer() { return internal_memory_transfer_;}
  void set_internal_memory_transfer() { internal_memory_transfer_ = true;}
  void set_metadata(int index, int data) { meta_data_[index] = data; }
  int metadata(int index) { return meta_data_[index]; }
  void print_incomplete_tasks();

  Task* Child(int i) { return platform_->get_task_object(childs_uids_[i]); }
  int nchilds() { return nchilds_; }
  void AddChild(unsigned long uid);
  void AddAllChilds();
 
#ifdef AUTO_PAR
  std::vector<BaseMem*>* get_write_list() { return &write_list_; }
  std::vector<BaseMem*>* get_read_list() { return &read_list_; }
  void add_to_read_list(BaseMem* mem) { read_list_.push_back(mem); }
  void add_to_write_list(BaseMem* mem) { write_list_.push_back(mem); }
#ifdef AUTO_FLUSH
  Graph* get_graph(){return graph_;}
  void set_graph(Graph* graph){graph_ = graph;}
  //void EraseDepend();
  void ReplaceDependFlushTask(Task * task);
#endif
#ifdef AUTO_SHADOW
  void set_shadow_dep_added(bool shadow_dep_added){ shadow_dep_added_ = shadow_dep_added;}
  bool get_shadow_dep_added(){ return shadow_dep_added_;}
#endif
#endif

private:
  void CompleteSub();

private:
  std::string name_;
  bool given_name_;
  unsigned long parent_;
  bool parent_exist_;
  int ncmds_;
  Command* cmds_[64];
  Command* cmd_kernel_;
  Command* cmd_last_;
  Device* dev_;
  int meta_data_[4];
  int devno_;
  Platform* platform_;
  Scheduler* scheduler_;
  std::vector<Task*> subtasks_;
  std::vector<Command *> reset_mems_;
#ifdef AUTO_PAR
  std::vector<BaseMem*> write_list_;
  std::vector<BaseMem*> read_list_;
#ifdef AUTO_FLUSH
  Graph* graph_;
#endif
#ifdef AUTO_SHADOW
  bool shadow_dep_added_;
#endif
#endif
  size_t subtasks_complete_;
  void* arch_;

  // for keepign track of the parents
  //Task** depends_;
  unsigned long* depends_uids_;
  int depends_max_;
  int ndepends_;
 
  // for keeping track of the childs
  unsigned long* childs_uids_;
  int childs_max_;
  int nchilds_;


  int recommended_stream_;
  int recommended_dev_;
  int brs_policy_;
  char opt_[64];

  int type_;
  int status_;
  bool sync_;
  bool user_;
  bool system_;
  bool disable_consistency_;
  bool internal_memory_transfer_;
  bool is_kernel_launch_disabled_;
  bool profile_data_transfers_;
  bool async_execution_;
  StreamPolicy stream_policy_;

  double time_;
  double time_start_;
  double time_end_;

  pthread_mutex_t stream_mutex_;
  pthread_mutex_t mutex_pending_;
  pthread_mutex_t mutex_executable_;
  pthread_mutex_t mutex_complete_;
  pthread_mutex_t mutex_subtasks_;
  pthread_cond_t complete_cond_;
  vector<DataObjectProfile>       out_dataobject_profiles;

public:
  static Task* Create(Platform* platform, int type, const char* name);
  static Task* Create(Platform* platform, int type, std::string name) {
    return Create(platform, type, name.c_str());
  }
  bool IsKernelSupported(Device *dev);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_TASK_H */

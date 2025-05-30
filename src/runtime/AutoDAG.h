#ifndef IRIS_SRC_RT_AUTODAG_H
#define IRIS_SRC_RT_AUTODAG_H

#include<string>
#include<vector>

namespace iris {
namespace rt {

class Platform;
class Command;
class Task;
class BaseMem;
class DataMem;
class Graph; 
class Kernel; 
class AutoDAG {
public:
  AutoDAG(Platform* platform, bool enable_df_sched);
  ~AutoDAG(){};

  void create_dependency(Command* cmd, Task* task, 
		  int param_info, BaseMem* mem, 
          Kernel* kernel, int idx);

  void create_multi_read_dependency(Task* task, 
		  BaseMem* mem);
  bool get_auto_dep(){ return auto_dep_;}
  void set_auto_dep(){ auto_dep_ = true;}
  void unset_auto_dep(){ auto_dep_ = false;}
  void df_bc_scheduling(Task* task, DataMem* mem);
  void initalize_h2d_task();
  void add_h2d_df_task(Task* task, Kernel* kernel);
#ifdef SANITY_CHECK
  void add_auto_dep_list(std::string new_item){ auto_dep_list_.push_back(new_item);}
  void add_manual_dep_list(std::string new_item){ manual_dep_list_.push_back(new_item);}
  void extra_dependencies();
  void missing_dependencies();
#endif

#ifdef AUTO_FLUSH
  void create_auto_flush(Command* cmd, Task* task, 
		  BaseMem* mem);
  Graph* get_current_graph(){return current_graph_;}
  void set_current_graph(Graph* current_graph){
	  current_graph_ = current_graph;}
#endif
#ifdef AUTO_SHADOW
  void create_auto_shadow(Command* cmd, Task* task, 
		  //BaseMem& mem);
		  BaseMem* mem);
  void create_shadow_flush(Command* cmd, Task* task, 
		  BaseMem* mem);
  int get_number_of_shadow(){ return number_of_shadow_;}
#endif

private:
  Platform* platform_;
  Task* current_task_;
  Kernel* current_kernel_;
  //BaseMem* current_mem_;
  int current_param_info_;
  int current_idx_;
  bool auto_dep_; // to mark whether it's a manual dependency or auto dependency
  char tn[256];
  int num_dev_; // total device 
  int cur_dev_; // current device
  bool enable_df_sched_; // enabling df scheduling
  std::vector<Task *> h2d_task_list_;
  Graph* current_graph_;
  bool enable_auto_flush_;

#ifdef SANITY_CHECK
  std::vector<std::string> manual_dep_list_;
  std::vector<std::string> auto_dep_list_;
#endif

#ifdef AUTO_SHADOW
   //Map<Dmem>  current_graph_;
  int number_of_shadow_;
#endif

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_AUTODAG_H */

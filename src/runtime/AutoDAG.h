#ifndef IRIS_SRC_RT_AUTODAG_H
#define IRIS_SRC_RT_AUTODAG_H

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
  AutoDAG(Platform* platform);
  ~AutoDAG(){};

  void create_dependency(Command* cmd, Task* task, 
		  int param_info, BaseMem* mem, 
          Kernel* kernel, int idx);

  void create_multi_read_dependency(Task* task, 
		  BaseMem* mem);


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

#endif

private:
  Platform* platform_;
  Task* current_task_;
  Kernel* current_kernel_;
  //BaseMem* current_mem_;
  int current_param_info_;
  int current_idx_;
  char tn[256];
#ifdef AUTO_FLUSH
  Graph* current_graph_;
#endif
#ifdef AUTO_SHADOW
   //Map<Dmem>  current_graph_;
#endif


};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_AUTODAG_H */

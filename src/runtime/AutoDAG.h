#ifndef IRIS_SRC_RT_AUTODAG_H
#define IRIS_SRC_RT_AUTODAG_H

namespace iris {
namespace rt {

class Platform;
class Command;
class Task;
class BaseMem;
class Graph; 
class AutoDAG {
public:
  AutoDAG(Platform* platform);
  ~AutoDAG(){};

  void create_dependency(Command* cmd, Task* task, 
		  int param_info, 
		  BaseMem* mem, Task* task_prev);

#ifdef AUTO_FLUSH
  void create_auto_flush(Command* cmd, Task* task, 
		  int param_info, 
		  BaseMem* mem, Task* task_prev);
#endif

private:
  Platform* platform_;
  char tn[256];
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_AUTODAG_H */

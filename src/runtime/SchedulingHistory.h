#ifndef BRISBANE_SRC_RT_SCHEDULING_HISTORY_H
#define BRISBANE_SRC_RT_SCHEDULING_HISTORY_H

#include "Profiler.h"
#include <set>

namespace iris {
namespace rt {

class Command;
class Task;

class SchedulingHistory : public Profiler {
public:
  SchedulingHistory(Platform* platform);
  virtual ~SchedulingHistory();

  void AddKernel(Command* cmd);
  void AddH2D(Command* cmd);
  void AddD2H(Command* cmd);
  void AddTask(Task* task);

private:
  void Add(Command* cmd);
  virtual int CompleteTask(Task* task);
  int CompleteCommand(Command* command);

protected:
  virtual int Main();
  virtual int Exit();
  virtual const char* FileExtension();

};

} /* namespace rt */
} /* namespace iris */


#endif /* IRIS_SRC_RT_SCHEDULING_HISTORY_H */


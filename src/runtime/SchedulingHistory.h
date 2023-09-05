#ifndef BRISBANE_SRC_RT_SCHEDULING_HISTORY_H
#define BRISBANE_SRC_RT_SCHEDULING_HISTORY_H

#include <mutex>
#include <iostream>
#include <fstream>

namespace iris {
namespace rt {

class Platform;
class Command;
class Task;

class SchedulingHistory {
public:
  SchedulingHistory(Platform* platform);
  virtual ~SchedulingHistory();

  void AddKernel(Command* cmd);
  void AddH2D(Command* cmd);
  void AddD2H(Command* cmd);
  void AddD2D(Command* cmd);
  void AddD2H_H2D(Command* cmd);
  void AddTask(Task* task);
  void Add(Command* cmd, std::string name, std::string type,double time_start,double time_end);



private:
  void Add(Command* cmd);
  int CompleteCommand(Command* command);
  int CompleteTask(Task* task);
  const char* policy_str(int policy);
  int CompleteSpecialCommand(Command* command, std::string name, std::string type, double time_start, double time_end);

  std::ofstream myfile;
  std::mutex file_mutex;

};

} /* namespace rt */
} /* namespace iris */


#endif /* IRIS_SRC_RT_SCHEDULING_HISTORY_H */


#ifndef IRIS_SRC_RT_PROFILER_DOT_H
#define IRIS_SRC_RT_PROFILER_DOT_H

#include "Profiler.h"
#include "pthread.h"
#include <set>
#include <map>
#include <vector>
#include <iostream>

namespace iris {
namespace rt {

class ProfilerDOT : public Profiler {
public:
  ProfilerDOT(Platform* platform);
  virtual ~ProfilerDOT();

  virtual int CompleteTask(Task* task);

protected:
  virtual int Main();
  virtual int Exit();
  virtual const char* FileExtension();

private:
  std::set<unsigned long> tasks_exit_;
  pthread_mutex_t   dot_lock_;
  bool no_task_;
#ifdef PER_TASK_COLOR
  std::vector<std::string> list_color;
  std::map<std::string, std::string> map_color;
  int round_robin_counter;
#endif

};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_PROFILER_DOT_H */


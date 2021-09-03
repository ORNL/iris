#ifndef BRISBANE_SRC_RT_PROFILER_DOT_H
#define BRISBANE_SRC_RT_PROFILER_DOT_H

#include "Profiler.h"
#include <set>

namespace brisbane {
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
  bool no_task_;
};

} /* namespace rt */
} /* namespace brisbane */


#endif /*BRISBANE_SRC_RT_PROFILER_DOT_H */


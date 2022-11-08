#ifndef IRIS_SRC_RT_PROFILER_DOT_H
#define IRIS_SRC_RT_PROFILER_DOT_H

#include "Profiler.h"
#include "pthread.h"
#include <set>

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
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_PROFILER_DOT_H */


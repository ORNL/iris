#pragma once
#include "Profiler.h"
#include "pthread.h"

namespace iris {
namespace rt {

class ProfilerEventRecord : public Profiler {
public:
  ProfilerEventRecord(Platform* platform);
  virtual ~ProfilerEventRecord();

  virtual int CompleteTask(Task* task);

protected:
  virtual int Main();
  virtual int Exit();
  virtual const char* FileExtension() { return "events.html"; }

private:
  pthread_mutex_t   chart_lock_;
  double first_task_;
  bool kernel_profile_;
};

} /* namespace rt */
} /* namespace iris */




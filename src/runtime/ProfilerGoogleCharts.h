#ifndef IRIS_SRC_RT_PROFILER_GOOGLE_CHARTS_H
#define IRIS_SRC_RT_PROFILER_GOOGLE_CHARTS_H

#include "Profiler.h"
#include "pthread.h"

namespace iris {
namespace rt {

class ProfilerGoogleCharts : public Profiler {
public:
  ProfilerGoogleCharts(Platform* platform, bool kernel_profile=false);
  virtual ~ProfilerGoogleCharts();

  virtual int CompleteTask(Task* task);

protected:
  virtual int Main();
  virtual int Exit();
  virtual const char* FileExtension() { if (kernel_profile_) return "kernel.html"; else return "html"; }

private:
  pthread_mutex_t   chart_lock_;
  double first_task_;
  bool kernel_profile_;
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_PROFILER_GOOGLE_CHARTS_H */


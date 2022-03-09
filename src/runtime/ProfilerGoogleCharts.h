#ifndef IRIS_SRC_RT_PROFILER_GOOGLE_CHARTS_H
#define IRIS_SRC_RT_PROFILER_GOOGLE_CHARTS_H

#include "Profiler.h"

namespace iris {
namespace rt {

class ProfilerGoogleCharts : public Profiler {
public:
  ProfilerGoogleCharts(Platform* platform);
  virtual ~ProfilerGoogleCharts();

  virtual int CompleteTask(Task* task);

protected:
  virtual int Main();
  virtual int Exit();
  virtual const char* FileExtension() { return "html"; }

private:
  double first_task_;
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_PROFILER_GOOGLE_CHARTS_H */


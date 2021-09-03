#ifndef BRISBANE_SRC_RT_PROFILER_GOOGLE_CHARTS_H
#define BRISBANE_SRC_RT_PROFILER_GOOGLE_CHARTS_H

#include "Profiler.h"

namespace brisbane {
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
} /* namespace brisbane */


#endif /*BRISBANE_SRC_RT_PROFILER_GOOGLE_CHARTS_H */


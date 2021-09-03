#include "ProfilerDOT.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"

#define PROFILER_DOT_HEADER "digraph {\n" \
                            "start[shape=Mdiamond, label=\"main\"]\n"

#define PROFILER_DOT_FOOTER "}\n"

namespace brisbane {
namespace rt {

ProfilerDOT::ProfilerDOT(Platform* platform) : Profiler(platform) {
  no_task_ = true;
  OpenFD();
  Main();
}

ProfilerDOT::~ProfilerDOT() {
  Exit();
}

int ProfilerDOT::Main() {
  Write(PROFILER_DOT_HEADER);
  return BRISBANE_OK;
}

int ProfilerDOT::Exit() {
  char s[64];
  for (std::set<unsigned long>::iterator I = tasks_exit_.begin(), E = tasks_exit_.end(); I != E; ++I) {
    unsigned long tid = *I;
    sprintf(s, "task%lu -> end\n", tid);
    Write(s);
  }
  if (no_task_) Write("start -> end\n");
  sprintf(s, "end[shape=Msquare, label=\"exit\\n%lf\"]\n", platform_->time_app());
  Write(s);
  Write(PROFILER_DOT_FOOTER);
  return BRISBANE_OK;
}

int ProfilerDOT::CompleteTask(Task* task) {
  no_task_ = false;
  unsigned long tid = task->uid();
  Device* dev = task->dev();
  int type = dev->type();
  int policy = task->brs_policy();
  double time = task->time();
  tasks_exit_.insert(tid);
  char s[64];
  sprintf(s, "task%lu[style=filled, fillcolor=%s, label=\"%s (%s)\\n%lf\"]\n", tid,
      type & brisbane_cpu ? "cyan" : type & brisbane_gpu ? "green" : "purple", task->name(), policy_str(policy), time);
  Write(s);

  int ndepends = task->ndepends();
  if (ndepends == 0) {
    sprintf(s, "start -> task%lu\n", tid);
    Write(s);
  } else {
    Task** deps = task->depends();
    for (int i = 0; i < ndepends; i++) {
      unsigned long duid = deps[i]->uid();
      tasks_exit_.erase(duid);
      sprintf(s, "task%lu -> task%lu\n", duid, tid);
      Write(s);
    }
  }
  return BRISBANE_OK;
}

const char* ProfilerDOT::FileExtension() {
  return "dot";
}

} /* namespace rt */
} /* namespace brisbane */


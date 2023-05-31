#include "ProfilerDOT.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"

#define PROFILER_DOT_HEADER "digraph {\n" \
                            "start[shape=Mdiamond, label=\"main\"]\n"

#define PROFILER_DOT_FOOTER "}\n"

namespace iris {
namespace rt {

ProfilerDOT::ProfilerDOT(Platform* platform) : Profiler(platform, "DOT") {
  no_task_ = true;
  OpenFD();
  Main();
  pthread_mutex_init(&dot_lock_, NULL);
}

ProfilerDOT::~ProfilerDOT() {
  Exit();
  pthread_mutex_destroy(&dot_lock_);
}

int ProfilerDOT::Main() {
  Write(PROFILER_DOT_HEADER);
  return IRIS_SUCCESS;
}

int ProfilerDOT::Exit() {
  char s[64];
  pthread_mutex_lock(&dot_lock_);
  for (std::set<unsigned long>::iterator I = tasks_exit_.begin(), E = tasks_exit_.end(); I != E; ++I) {
    unsigned long tid = *I;
    sprintf(s, "task%lu -> end\n", tid);
    Write(s);
  }
  if (no_task_) Write("start -> end\n");
  sprintf(s, "end[shape=Msquare, label=\"exit\\n%lf\"]\n", platform_->time_app());
  Write(s);
  Write(PROFILER_DOT_FOOTER);
  pthread_mutex_unlock(&dot_lock_);
  return IRIS_SUCCESS;
}

int ProfilerDOT::CompleteTask(Task* task) {
  no_task_ = false;
  unsigned long tid = task->uid();
  Device* dev = task->dev();
  int type = iris_cpu; 
  if (dev != NULL) type = dev->type();
  int policy = task->brs_policy();
  double time = task->time();
  pthread_mutex_lock(&dot_lock_);
  if (tasks_exit_.find(tid) == tasks_exit_.end())
      tasks_exit_.insert(tid);
  pthread_mutex_unlock(&dot_lock_);
  char s[1024];
  sprintf(s, "task%lu[style=filled, fillcolor=%s, label=\"%s (%s)\\n%lf\"]\n", tid,
      type & iris_cpu ? "cyan" : type & iris_gpu ? "green" : "purple", task->name(), policy_str(policy), time);
  pthread_mutex_lock(&dot_lock_);
  Write(s);
  pthread_mutex_unlock(&dot_lock_);

  int ndepends = task->ndepends();
  if (ndepends == 0) {
    sprintf(s, "start -> task%lu\n", tid);
    Write(s);
  } else {
    unsigned long *deps = task->depends();
    pthread_mutex_lock(&dot_lock_);
    for (int i = 0; i < ndepends; i++) {
      unsigned long duid = deps[i];
      if (tasks_exit_.find(duid) != tasks_exit_.end())
          tasks_exit_.erase(duid);
      sprintf(s, "task%lu -> task%lu\n", duid, tid);
      Write(s);
    }
    pthread_mutex_unlock(&dot_lock_);
  }
  return IRIS_SUCCESS;
}

const char* ProfilerDOT::FileExtension() {
  return "dot";
}

} /* namespace rt */
} /* namespace iris */


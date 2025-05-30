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

#ifdef PER_TASK_COLOR
  list_color = {"blue","red","cyan","purple","green"};
  round_robin_counter = 0;
#endif
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
#ifndef PER_TASK_COLOR
  sprintf(s, "end[shape=Msquare, label=\"exit\\n%lf\"]\n", platform_->time_app());
#else
  sprintf(s, "end[shape=Msquare, label=exit]\n");
#endif
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
#ifndef PER_TASK_COLOR
  sprintf(s, "task%lu[style=filled, fillcolor=%s, label=\"%s (%s)\\n%lf\"]\n", tid,
      type & iris_cpu ? "cyan" : type & iris_gpu ? "green" : "purple", task->name(), policy_str(policy), time);
  pthread_mutex_lock(&dot_lock_);
  Write(s);
  pthread_mutex_unlock(&dot_lock_);
#else
  std::string current_color = "green";
  std::string short_task_name = task->name();
  if(task->name() != NULL){
    std::string temp_task_name(task->name(),5);
    short_task_name = temp_task_name;
    if( map_color.find(short_task_name) == map_color.end()){ 
        current_color = list_color[round_robin_counter++];
        if (list_color.size() == round_robin_counter) 
            round_robin_counter = 0;
        map_color.insert ( std::pair<std::string,std::string>(short_task_name,current_color)); 
   }else {
        current_color = map_color.find(short_task_name)->second;
    }
    //std::cout << short_task_name << " " << current_color << std::endl; 
  }
  if ( short_task_name != "Graph"){
    sprintf(s, "task%lu[style=filled, fillcolor=%s, label=\"%s\"]\n", tid,
    //sprintf(s, "task%lu[style=filled, fillcolor=%s, label=\"%s (%s)\\n%lf\"]\n", tid,
      current_color.c_str(), short_task_name.c_str());
      //current_color.c_str(), short_task_name.c_str(), policy_str(policy), time);
      //current_color.c_str(), task->name(), policy_str(policy), time);
    pthread_mutex_lock(&dot_lock_);
    Write(s);
    pthread_mutex_unlock(&dot_lock_);
  }
#endif
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
#ifndef PER_TASK_COLOR
      sprintf(s, "task%lu -> task%lu\n", duid, tid);
      Write(s);
#else
      if ( short_task_name != "Graph"){
        sprintf(s, "task%lu -> task%lu\n", duid, tid);
        Write(s);
      }
#endif
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


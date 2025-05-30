#include "SchedulingHistory.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"
#include "Command.h"

#include <time.h>
#include <iostream>
#include <fstream>
#include <locale>
#include <string>

#define SCHEDULING_HISTORY_HEADER "taskname,type,start,end,duration,size,policy,acclname"
#define SCHEDULING_HISTORY_FOOTER ""

namespace iris {
namespace rt {

SchedulingHistory::SchedulingHistory(Platform* platform) {
  char* provided_filepath = getenv("IRIS_HISTORY_FILE");
  std::string provided_filepath_str;
  if (!provided_filepath){
    time_t t = time(NULL);
    char s[64];
    strftime(s, 64, "%Y%m%d-%H%M%S", localtime(&t));
    //provided_filepath = new char[512];
    //sprintf(provided_filepath, "%s-%s-%s.%s", platform->app(), platform->host(), s, ".csv");
    provided_filepath_str = string(platform->app()) + "-" + string(platform->host()) + "-" + string(s) + string(".csv");
  }
  else 
    provided_filepath_str = string(provided_filepath);
  _trace("Schedule history path:%s", provided_filepath_str.c_str());

  myfile.open(provided_filepath_str.c_str(), std::ios::out);
  myfile << SCHEDULING_HISTORY_HEADER << std::endl;
}

SchedulingHistory::~SchedulingHistory() {
  myfile.close();
}

void SchedulingHistory::AddKernel(Command* cmd) {
  //ignore logging kernels at the moment
  //if (cmd->type_init()){
  //  //printf("not logging initialization -- but performed init on device id %d\n",cmd->task()->dev()->devno());
  //  return;
  //}
  Add(cmd);
}

void SchedulingHistory::AddH2D(Command* cmd) {
  Add(cmd);
}

void SchedulingHistory::AddD2H(Command* cmd) {
  Add(cmd);
}

void SchedulingHistory::AddD2D(Command* cmd) {
  Add(cmd);
}

void SchedulingHistory::AddD2H_H2D(Command* cmd) {
  Add(cmd);
}

void SchedulingHistory::AddTask(Task* task) {
  CompleteTask(task);
}

void SchedulingHistory::Add(Command* cmd){
  CompleteCommand(cmd);
}

void SchedulingHistory::Add(Command* cmd, string name, string type,double time_start,double time_end){
  CompleteSpecialCommand(cmd,name,type,time_start,time_end);
}

int SchedulingHistory::CompleteSpecialCommand(Command* command, string name, string type,double time_start,double time_end) {
  const std::lock_guard<std::mutex> lock(file_mutex);
  char s[1024];
  size_t ksize  = command->size();
  myfile << name << ','
         << type << ','
         << time_start << ','
         << time_end << ','
         << (time_end - time_start) << ','
         << ksize << ','
         << policy_str(command->task()->brs_policy()) << ','
         << command->task()->dev()->name() << ' '
         << command->task()->dev()->devno() << std::endl;
  return IRIS_SUCCESS;
}

int SchedulingHistory::CompleteCommand(Command* command) {
  const std::lock_guard<std::mutex> lock(file_mutex);
  if (!command->given_name()) {
    if(command->type_kernel()){//use task name rather than kernel name
      command->set_name(command->task()->name());
    } else {
      command->set_name(command->type_name());
    }
  }

  char s[1024];
  size_t ksize  = command->type_kernel() ? command->ws() : command->size();
  myfile << command->name() << ','
         << command->type_name() << ','
         << command->time_start() << ','
         << command->time_end() << ','
         << command->time_duration() << ','
         << ksize << ','
         << policy_str(command->task()->brs_policy()) << ','
         << command->task()->dev()->name() << ' '
         << command->task()->dev()->devno() << std::endl;
  return IRIS_SUCCESS;
}

int SchedulingHistory::CompleteTask(Task* task) {
  /*
  if (task->cmd_last()->type_h2dnp())
    printf("data memory uses type_h2dnp\n");
  if (task->cmd_last()->type_h2broadcast())
    printf("data memory uses h2broadcast\n");
  if (task->cmd_last()->type_memflush())
    printf("data memory uses memflush\n");
  */
  if (task->type() == IRIS_TASK_PERM){
    //if the task is named prioritize using that in the log
    if (task->given_name()){
      task->cmd_last()->set_name(task->name());
    }
    if(task->cmd_last()->type_kernel()){
      Add(task->cmd_last());
      return IRIS_SUCCESS;
    }
    else if (task->cmd_last()->type_h2d())
      AddH2D(task->cmd_last());
    else if (task->cmd_last()->type_d2h())
      AddD2H(task->cmd_last());
    else
      return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

const char* SchedulingHistory::policy_str(int policy) {
  switch (policy) {
    case iris_default:    return "default";
    case iris_cpu:        return "cpu";
    case iris_nvidia:     return "nvidia";
    case iris_amd:        return "amd";
    case iris_gpu_intel:  return "gpu intel";
    case iris_gpu:        return "gpu";
    case iris_phi:        return "phi";
    case iris_fpga:       return "fpga";
    case iris_dsp:        return "dsp";
    case iris_roundrobin: return "roundrobin";
    case iris_julia_policy: return "julia policy";
    case iris_depend:     return "depend";
    case iris_data:       return "data";
    case iris_profile:    return "profile";
    case iris_random:     return "random";
    case iris_pending:    return "pending";
    case iris_sdq:        return "sdq";
    case iris_custom:     return "custom";
    default: break;
  }
  return policy & iris_ftf ? "ftf" : "?";
}


} /* namespace rt */
} /* namespace iris */


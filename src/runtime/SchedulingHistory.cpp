#include "SchedulingHistory.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"
#include "Command.h"

#include <iostream>
#include <fstream>
//#include <codecvt>
#include <locale>
#include <string>

#define SCHEDULING_HISTORY_HEADER "taskname,type,start,end,duration,size,policy,acclname\n"

#define SCHEDULING_HISTORY_FOOTER ""

namespace iris {
namespace rt {

SchedulingHistory::SchedulingHistory(Platform* platform) : Profiler(platform, "SchedulingHistory") {
  OpenFD();
  Main();
}

SchedulingHistory::~SchedulingHistory() {
  Exit();
}

int SchedulingHistory::Main() {
  Write(SCHEDULING_HISTORY_HEADER);
  return IRIS_SUCCESS;
}

void SchedulingHistory::AddKernel(Command* cmd) {
  //todo set type?
  Add(cmd);
}

void SchedulingHistory::AddH2D(Command* cmd) {
  //check if the buffer has a name, use that rather than the transfer type
  if(cmd->task()->name())
    cmd->set_name(cmd->task()->name());
  Add(cmd);
}

void SchedulingHistory::AddD2H(Command* cmd) {
  if(cmd->task()->name())
    cmd->set_name(cmd->task()->name());
  Add(cmd);
}

void SchedulingHistory::AddTask(Task* task) {
  CompleteTask(task);
}

void SchedulingHistory::Add(Command* cmd){
  //set time, and device id?
  CompleteCommand(cmd);
}

int SchedulingHistory::Exit() {
  Write(SCHEDULING_HISTORY_FOOTER);
  return IRIS_SUCCESS;
}

int SchedulingHistory::CompleteCommand(Command* command) {
  //don't use named kernels (that's propagated to the command name anyway) and omit unnamed h2d/d2h transfers (these are unnamed and would inherit the wrong kernel command name instead)
  if(!command->name()) {
    command->set_name(command->type_name());
  }
  char s[512];
  size_t ksize  = command->type_kernel() ? command->ws() : command->size();//

  sprintf(s, "%s,%s,%f,%f,%f,%zu,%s,%s #%i\n",command->name(),command->type_name(),command->time_start(),command->time_end(),command->time_duration(),ksize,policy_str(command->task()->brs_policy()),command->task()->dev()->name(),command->task()->dev()->devno());
  Write(s);
  return IRIS_SUCCESS;
}

int SchedulingHistory::CompleteTask(Task* task) {
  if (task->type() == IRIS_TASK_PERM){
    //if the task is named prioritize using that in the log
    if (task->given_name()){
      task->cmd_last()->set_name(task->name());
    }

    if(task->cmd_last()->type_kernel()){
      char s[512];
      size_t ksize  = task->cmd_last()->ws();
      sprintf(s, "%s,%s,%f,%f,%f,%zu,%s,%s #%i\n",task->name(),task->cmd_last()->type_name(),task->cmd_last()->time_start(),task->cmd_last()->time_end(),task->cmd_last()->time_duration(),ksize,policy_str(task->brs_policy()),task->dev()->name(),task->dev()->devno());
      Write(s);
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

const char* SchedulingHistory::FileExtension() {
  return "csv";
}

} /* namespace rt */
} /* namespace iris */


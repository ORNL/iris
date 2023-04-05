#include "ProfilerGoogleCharts.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"

#define PROFILER_HEADER1  "<script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>\n" \
                          "<script type='text/javascript'>\n" \
                          "google.charts.load('current', {packages:['timeline']});\n" \
                          "google.charts.setOnLoadCallback(drawChart);\n" \
                          "function drawChart() {\n" \
                          "var container = document.getElementById('iris');\n" \
                          "var chart = new google.visualization.Timeline(container);\n" \
                          "var dataTable = new google.visualization.DataTable();\n"
#define PROFILER_HEADER2  "dataTable.addColumn({ type: 'string', id: 'Device' });\n" \
                          "dataTable.addColumn({ type: 'string', id: 'Task' });\n" \
                          "dataTable.addColumn({ type: 'number', id: 'Start' });\n" \
                          "dataTable.addColumn({ type: 'number', id: 'End' });\n" \
                          "dataTable.addRows([\n"

#define PROFILER_FOOTER   "]);\n" \
                          "var options = {\n" \
                          "timeline: { colorByRowLabel: true },\n" \
                          "avoidOverlappingGridLines: false\n" \
                          "};\n" \
                          "chart.draw(dataTable, options);\n" \
                          "}\n" \
                          "</script>\n" \
                          "<div id='iris' style='height: 100%;'></div>\n"

namespace iris {
namespace rt {

ProfilerGoogleCharts::ProfilerGoogleCharts(Platform* platform, bool kernel_profile) : Profiler(platform, "GoogleChart") {
  kernel_profile_ = kernel_profile;
  first_task_ = 0.0;
  pthread_mutex_init(&chart_lock_, NULL);
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("GOOGLE_CHART", &path, NULL);
  OpenFD(path);
  Main();
}

ProfilerGoogleCharts::~ProfilerGoogleCharts() {
  Exit();
  pthread_mutex_destroy(&chart_lock_);
}

int ProfilerGoogleCharts::Main() {
  pthread_mutex_lock(&chart_lock_);
  Write(PROFILER_HEADER1);
  Write(PROFILER_HEADER2);
  pthread_mutex_unlock(&chart_lock_);
  return IRIS_SUCCESS;
}

int ProfilerGoogleCharts::Exit() {
  pthread_mutex_lock(&chart_lock_);
  Write(PROFILER_FOOTER);
  pthread_mutex_unlock(&chart_lock_);
  return IRIS_SUCCESS;
}

int ProfilerGoogleCharts::CompleteTask(Task* task) {
  //if (first_task_ == 0.0) first_task_ = task->time_start() * 1.e+3;
  //printf("First task time:%f\n", first_task_);
  //unsigned long tid = task->uid();
  Device* dev = task->dev();
  int policy = task->brs_policy();
  if (dev != NULL) {
      char s[1024];
      if (!kernel_profile_) {
          sprintf(s, "[ '%s %d', '%s (%s)', %lf, %lf ],\n", dev->name(), dev->devno(), task->name(), policy_str(policy), (task->time_start() * 1.e+3) - first_task_, (task->time_end() * 1.e+3) - first_task_);
          pthread_mutex_lock(&chart_lock_);
          Write(s);
          pthread_mutex_unlock(&chart_lock_);
      }
      else if (task->cmd_kernel() != NULL){
          sprintf(s, "[ '%s %d', '%s (%s)', %lf, %lf ],\n", dev->name(), dev->devno(), task->name(), policy_str(policy), (task->cmd_kernel()->time_start() * 1.e+3) - first_task_, (task->cmd_kernel()->time_end() * 1.e+3) - first_task_);
          pthread_mutex_lock(&chart_lock_);
          Write(s);
          pthread_mutex_unlock(&chart_lock_);
      }
      //printf("Profiling Task: %s %lf %lf\n", s, task->time_start()*1.e+3, task->time_end()*1.e+3);
  }
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


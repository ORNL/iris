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
                          "var container = document.getElementById('brisbane');\n" \
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
                          "<div id='brisbane' style='height: 100%;'></div>\n"

namespace brisbane {
namespace rt {

ProfilerGoogleCharts::ProfilerGoogleCharts(Platform* platform) : Profiler(platform) {
  first_task_ = 0.0;
  OpenFD();
  Main();
}

ProfilerGoogleCharts::~ProfilerGoogleCharts() {
  Exit();
}

int ProfilerGoogleCharts::Main() {
  Write(PROFILER_HEADER1);
  Write(PROFILER_HEADER2);
  return BRISBANE_OK;
}

int ProfilerGoogleCharts::Exit() {
  Write(PROFILER_FOOTER);
  return BRISBANE_OK;
}

int ProfilerGoogleCharts::CompleteTask(Task* task) {
  if (first_task_ == 0.0) first_task_ = task->time_start() * 1.e+3;
  unsigned long tid = task->uid();
  Device* dev = task->dev();
  int policy = task->brs_policy();
  char s[256];
  sprintf(s, "[ '%s %d', '%s (%s)', %lf, %lf ],\n", dev->name(), dev->devno(), task->name(), policy_str(policy), (task->time_start() * 1.e+3) - first_task_, (task->time_end() * 1.e+3) - first_task_);
  Write(s);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */


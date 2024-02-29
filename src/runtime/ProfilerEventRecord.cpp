#include "ProfilerEventRecord.h"
#include "Debug.h"
#include "Device.h"
#include "Platform.h"
#include "Task.h"

#include <iostream>
#include <string>
#include <regex>

using namespace std;
#define PROFILER_HEADER1  R"delimiter(<script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>\
<script type='text/javascript'> \
google.charts.load('current', {packages:['timeline']}); \
google.charts.setOnLoadCallback(drawChart); \
const ProfileRecordType = \
{ \
    PROFILE_H2D : 0, \
    PROFILE_D2H : 1, \
    PROFILE_D2D : 2, \
    PROFILE_D2H_H2D : 3, \
    PROFILE_KERNEL : 4, \
    PROFILE_O2D : 5, \
    PROFILE_D2O : 6, \
}; \
const ProfileTypeColor = \
{ \
    0: 'red', \
    1: 'cyan', \
    2: 'yellow', \
    3: 'darkmagenta', \
    4: 'green', \
    5: 'coral', \
    6: 'brown', \
}; \
function drawChart() { \
var container = document.getElementById('iris'); \
var chart = new google.visualization.Timeline(container); \
var dataTable = new google.visualization.DataTable(); \
)delimiter"

#define PROFILER_HEADER2  R"delimiter( \
dataTable.addColumn({ type: 'string', id: 'Device' }); \
dataTable.addColumn({ type: 'string', id: 'Task' }); \
dataTable.addColumn({ type: 'number', id: 'Start' }); \
dataTable.addColumn({ type: 'number', id: 'End' }); \
dataTable.addColumn({ type: 'string', role: 'style'}); \
/* Function to load rows from another data structure and add to the data table. */\
function loadRowsAndDrawChart(rows, color_options) { \
    dataTable.addRows(rows); \
    /*var options = { \
      colors: color_options, \
    }; \
    chart.draw(dataTable, options); */\
    chart.draw(dataTable); \
} \
/* Checkbox change event handler */\
function extract_data(newData, color_options, pattern) { \
    /* Add data based on a flag (in this case, just a sample) */ \
    for (let i=0; i<iris_data.length; i++) { \
        if  (iris_data[i][4] === pattern) { \
            var data = iris_data[i].slice(0,4); \
            data.push('color:'+ProfileTypeColor[iris_data[i][4]]); \
            newData.push(data); \
            color_options.push(ProfileTypeColor[iris_data[i][4]]); \
        } \
    } \
} \
function handleCheckboxChange() { \
    console.log(dataTable.getNumberOfRows()); \
    dataTable.removeRows(0, dataTable.getNumberOfRows()); \
    var newData = []; \
    var color_options = []; \
    var kernel_enable = document.getElementById('kernel'); \
    var d2h_enable = document.getElementById('d2h_enable'); \
    var d2d_enable = document.getElementById('d2d_enable'); \
    var h2d_enable = document.getElementById('h2d_enable'); \
    var d2h_h2d_enable = document.getElementById('d2h_h2d_enable'); \
    if (d2h_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2H); \
    }  \
    if (h2d_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_H2D); \
    }  \
    if (d2h_h2d_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2H_H2D); \
    }  \
    if (d2d_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2D); \
    }  \
    if (kernel_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_KERNEL); \
    }  \
    loadRowsAndDrawChart(newData, color_options); \
} \
var iris_data = [ \
)delimiter"

#define PROFILER_FOOTER  R"delimiter( \
]; \
var options = { \
timeline: { colorByRowLabel: true }, \
avoidOverlappingGridLines: false \
}; \
handleCheckboxChange(); \
document.getElementById('kernel').addEventListener('change', handleCheckboxChange); \
document.getElementById('h2d_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('d2h_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('d2d_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('d2h_h2d_enable').addEventListener('change', handleCheckboxChange); \
} \
</script> \
<label for="kernel">Kernel:</label> \
<input type="checkbox" id="kernel" checked> \
<label for="h2d_enable">H2D:</label> \
<input type="checkbox" id="h2d_enable" checked> \
<label for="d2d_enable">D2D:</label> \
<input type="checkbox" id="d2d_enable" checked> \
<label for="d2h_enable">D2H:</label> \
<input type="checkbox" id="d2h_enable" checked> \
<label for="d2h_h2d_enable">D2H-H2D:</label> \
<input type="checkbox" id="d2h_h2d_enable" checked> \
<div id='iris' style='height: 100%;'></div> \
)delimiter"

namespace iris {
namespace rt {

std::string removeTrailingBackslash(const std::string& input) {
    //cout << "Input:" << input << "\n";
    // Regular expression to match a backslash at the end of a line
    std::regex pattern("\\\\\n");

    // Replace each match with an empty string
    return std::regex_replace(input, pattern, "\n");
}

ProfilerEventRecord::ProfilerEventRecord(Platform* platform) : Profiler(platform, "EventChart") {
  first_task_ = 0.0;
  pthread_mutex_init(&chart_lock_, NULL);
  char* path = NULL;
  Platform::GetPlatform()->EnvironmentGet("EVENT_CHART", &path, NULL);
  OpenFD(path);
  Main();
}

ProfilerEventRecord::~ProfilerEventRecord() {
  Exit();
  pthread_mutex_destroy(&chart_lock_);
}

int ProfilerEventRecord::Main() {
  pthread_mutex_lock(&chart_lock_);
  Write(removeTrailingBackslash(PROFILER_HEADER1));
  Write(removeTrailingBackslash(PROFILER_HEADER2));
  pthread_mutex_unlock(&chart_lock_);
  return IRIS_SUCCESS;
}

int ProfilerEventRecord::Exit() {
  pthread_mutex_lock(&chart_lock_);
  Write(removeTrailingBackslash(PROFILER_FOOTER));
  pthread_mutex_unlock(&chart_lock_);
  return IRIS_SUCCESS;
}

int ProfilerEventRecord::CompleteTask(Task* task) {
  //if (first_task_ == 0.0) first_task_ = task->time_start() * 1.e+3;
  //printf("First task time:%f\n", first_task_);
  //unsigned long tid = task->uid();
  Device* dev = task->dev();
  int policy = task->brs_policy();
  vector<ProfileEvent> & pevents = task->profile_events();
  char s[1024];
#if 0
  if (dev != NULL) {
      sprintf(s, "[ '%s %d', '%s (%s)', %lf, %lf, -1 ],\n", dev->name(), dev->devno(), task->name(), policy_str(policy), (task->time_start() * 1.e+3) - first_task_, (task->time_end() * 1.e+3) - first_task_);
      pthread_mutex_lock(&chart_lock_);
      Write(s);
      pthread_mutex_unlock(&chart_lock_);
      printf("Profiling Task: %s %lf %lf\n", s, task->time_start()*1.e+3, task->time_end()*1.e+3);
  }
#endif
#if 1
  //printf("Profiling task:%lu:%s size:%lu\n", task->uid(), task->name(), pevents.size());
  for (ProfileEvent & p : pevents) {
     Device *event_dev = p.event_dev();
     double first_event_cpu_begin_time = event_dev->first_event_cpu_begin_time();
     double first_event_cpu_end_time = event_dev->first_event_cpu_end_time();
     double midpoint = (first_event_cpu_end_time - first_event_cpu_begin_time)/2.0f;
     double start_time = midpoint + p.GetStartTime();
     double end_time = midpoint + p.GetEndTime();
     int stream = p.stream();
     unsigned long uid = p.uid();
     ProfileRecordType type = p.type();
     int connect_dev = p.connect_dev();
     if (type == PROFILE_KERNEL) {
         sprintf(s, "[ '%s %d', '%lu:%s (%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_KERNEL],\n", event_dev->name(), event_dev->devno(), task->uid(), task->name(), policy_str(policy), stream, start_time, end_time);
     }
     else if (type == PROFILE_D2D) {
         sprintf(s, "[ '%s %d', 'D2D: m%lu from (%d) task (%llu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2D],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time);
     }
     else if (type == PROFILE_H2D) {
         sprintf(s, "[ '%s %d', 'H2D: m%lu from (Host) task (%llu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_H2D],\n", event_dev->name(), event_dev->devno(), uid, task->uid(), task->name(), stream, start_time, end_time);
     }
     else if (type == PROFILE_D2O) {
         sprintf(s, "[ '%s %d', 'D2O: m%lu from (%d) task (%llu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2O],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time);
     }
     else if (type == PROFILE_O2D) {
         sprintf(s, "[ '%s %d', 'O2D: m%lu from (%d) task (%llu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_O2D],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time);
     }
     else if (type == PROFILE_D2H) {
         sprintf(s, "[ '%s %d', 'D2H: m%lu to (Host) task (%llu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2H],\n", event_dev->name(), event_dev->devno(), uid, task->uid(), task->name(), stream, start_time, end_time);
     }
     else {
         sprintf(s, "[ '%s %d', 'D2D: m%lu from (%d) task (%llu:%s) stream (%d)', %lf, %lf, %d],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time, (int)p.type());
     }
     pthread_mutex_lock(&chart_lock_);
     Write(s);
     pthread_mutex_unlock(&chart_lock_);
  }
#endif
  //printf("Completed ProfilerEventRecord::CompleteTask\n");
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


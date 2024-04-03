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
    PROFILE_D2HH2D_D2H : 3, \
    PROFILE_D2HH2D_H2D : 4, \
    PROFILE_KERNEL : 5, \
    PROFILE_O2D : 6, \
    PROFILE_D2O : 7, \
    PROFILE_INIT: 8, \
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
    7: 'black', \
    8: 'gray', \
}; \
function sort_compare_fn(a, b) { \
    const v1 = a[0]; \
    const v2 = b[0]; \
    return v1.localeCompare(v2); \
}; \
var options = { \
timeline: { colorByRowLabel: true }, \
avoidOverlappingGridLines: false \
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
    const sorted_rows = rows.sort(sort_compare_fn); \
    dataTable.addRows(sorted_rows); \
    /*var col_options = { \
      colors: color_options, \
    }; \
    chart.draw(dataTable, col_options); */\
    chart.draw(dataTable, options); \
} \
/* Checkbox change event handler */\
function extract_task_data(newData, color_options) { \
    /* Add data based on a flag (in this case, just a sample) */ \
    var task_map = { }; \
    for (let i=0; i<iris_data.length; i++) { \
        var task_uid = iris_data[i][5]; \
        var task_name = iris_data[i][6]; \
        var start_time = iris_data[i][2]; \
        var end_time = iris_data[i][3]; \
        if (task_map[task_uid] === undefined) { \
            task_map[task_uid] = [iris_data[i][0], task_name, -1.0, 0.0, '']; /*color:red*/ \
        } \
        if (task_map[task_uid][2] < 0.0 || task_map[task_uid][2] > start_time) \
            task_map[task_uid][2] = start_time; \
        if (task_map[task_uid][3] < end_time) \
            task_map[task_uid][3] = end_time; \
        if (iris_data[i][4] === ProfileRecordType.PROFILE_KERNEL) \
            task_map[task_uid][0] = iris_data[i][0]; \
    } \
    for (const [key, data] of Object.entries(task_map)) { \
            newData.push(data); \
    } \
} \
function extract_data(newData, color_options, pattern) { \
    /* Add data based on a flag (in this case, just a sample) */ \
    for (let i=0; i<iris_data.length; i++) { \
        if  (iris_data[i][4] === pattern) { \
            var data = iris_data[i].slice(0,4); \
            data.push(''); /*color:'+ProfileTypeColor[iris_data[i][4]]*/ \
            newData.push(data); \
            color_options.push(ProfileTypeColor[iris_data[i][4]]); \
        } \
    } \
} \
function handleCheckboxChange() { \
    dataTable.removeRows(0, dataTable.getNumberOfRows()); \
    var newData = []; \
    var color_options = []; \
    var kernel_enable = document.getElementById('kernel'); \
    var d2h_enable = document.getElementById('d2h_enable'); \
    var d2d_enable = document.getElementById('d2d_enable'); \
    var h2d_enable = document.getElementById('h2d_enable'); \
    var task_enable= document.getElementById('task_enable'); \
    var d2h_h2d_enable = document.getElementById('d2h_h2d_enable'); \
    if (d2h_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2H); \
    }  \
    if (h2d_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_H2D); \
    }  \
    if (task_enable.checked) { \
        extract_task_data(newData, color_options); \
    }  \
    if (d2h_h2d_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2HH2D_D2H); \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2HH2D_H2D); \
    }  \
    if (d2d_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_D2D); \
    }  \
    if (kernel_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_KERNEL); \
    }  \
    if (init_enable.checked) { \
        extract_data(newData, color_options, ProfileRecordType.PROFILE_INIT); \
    }  \
    loadRowsAndDrawChart(newData, color_options); \
} \
var iris_data = [ \
)delimiter"

#define PROFILER_FOOTER  R"delimiter( \
]; \
handleCheckboxChange(); \
document.getElementById('kernel').addEventListener('change', handleCheckboxChange); \
document.getElementById('init_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('h2d_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('d2h_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('d2d_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('d2h_h2d_enable').addEventListener('change', handleCheckboxChange); \
document.getElementById('task_enable').addEventListener('change', handleCheckboxChange); \
} \
</script> \
<label for="init_enable">Commander dispatcher:</label>  <input type="checkbox" id="init_enable"> \
<label for="kernel">Kernel:</label>  <input type="checkbox" id="kernel" checked> \
<label for="h2d_enable">H2D:</label> <input type="checkbox" id="h2d_enable" checked> \
<label for="d2d_enable">D2D:</label> <input type="checkbox" id="d2d_enable" checked> \
<label for="d2h_enable">D2H:</label> <input type="checkbox" id="d2h_enable" checked> \
<label for="d2h_h2d_enable">D2H-H2D:</label> <input type="checkbox" id="d2h_h2d_enable" checked> \
<br> \
<label for="task_enable">Task level details:</label>  <input type="checkbox" id="task_enable"> \
<br> \
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
     double midpoint = event_dev->first_event_cpu_mid_point_time();
     double start_time = midpoint + p.GetStartTime();
     double end_time = midpoint + p.GetEndTime();
     int stream = p.stream();
     unsigned long uid = p.uid();
     ProfileRecordType type = p.type();
     int connect_dev = p.connect_dev();
     if (type == PROFILE_KERNEL) {
         sprintf(s, "[ '%s %d', 'Kernel Task:%lu:%s (%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_KERNEL, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), task->uid(), task->name(), policy_str(policy), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_D2D) {
         sprintf(s, "[ '%s %d', 'D2D: m%lu from (%d) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2D, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_H2D) {
         sprintf(s, "[ '%s %d', 'H2D: m%lu from (Host) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_H2D, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_D2H) {
         sprintf(s, "[ '%s %d', 'D2H: m%lu to (Host) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2H, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_D2HH2D_H2D) {
         sprintf(s, "[ '%s %d', 'D2H-H2D (H2D): m%lu from (Host) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2HH2D_H2D, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_D2HH2D_D2H) {
         sprintf(s, "[ '%s %d', 'D2H-H2D (D2H): m%lu to (Host) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2HH2D_D2H, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_D2O) {
         sprintf(s, "[ '%s %d', 'D2O: m%lu from (%d) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_D2O, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_O2D) {
         sprintf(s, "[ '%s %d', 'O2D: m%lu from (%d) task (%lu:%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_O2D, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time, task->uid(), task->name());
     }
     else if (type == PROFILE_INIT) {
         sprintf(s, "[ '%s %d', 'Command dispatcher Task:%lu:%s (%s) stream (%d)', %lf, %lf, ProfileRecordType.PROFILE_INIT, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), task->uid(), task->name(), policy_str(policy), stream, start_time, end_time, task->uid(), task->name());
     }
     else {
         sprintf(s, "[ '%s %d', 'D2D: m%lu from (%d) task (%lu:%s) stream (%d)', %lf, %lf, %d, %lu, '%s'],\n", event_dev->name(), event_dev->devno(), uid, connect_dev, task->uid(), task->name(), stream, start_time, end_time, (int)p.type(), task->uid(), task->name());
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


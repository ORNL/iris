#ifndef IRIS_SRC_RT_GRAPH_H
#define IRIS_SRC_RT_GRAPH_H

#include "Retainable.h"
#include "Command.h"
#include "Platform.h"
#include <pthread.h>
#include <vector>
#include <memory>
using namespace std;
namespace iris {
namespace rt {
class BaseMem;
class GraphMetadata;
class TaskProfile;
class DataObjectProfile;

#ifdef AUTO_PAR
class AutoDAG;
#endif



class Graph: public Retainable<struct _iris_graph, Graph> {
public:
  Graph(Platform* platform);
  virtual ~Graph();

  void AddTask(Task* task, unsigned long uid);
  void Submit();
  void Complete();
  void Wait();
  void ResetMemories();

  int enable_mem_profiling();
  Platform* platform() { return platform_; }
  std::vector<Task*>* tasks() { return &tasks_; }
  std::vector<Task*> & tasks_list() { return tasks_; }
  std::vector<Task*> formatted_tasks();
  int iris_tasks(iris_task *pv);
  void set_order(int *order);
  int tasks_count() { return tasks_.size(); }
  bool is_retainable() { return retain_tasks_; }
  void enable_retainable(); 
  void disable_retainable();
  static void GraphRelease(void *data);
  shared_ptr<GraphMetadata> get_metadata(int iterations=3);
private:
  Platform* platform_;
  Scheduler* scheduler_;
  std::vector<Task*> tasks_;

  //Task* start_;
  Task* end_;
  iris_task end_brs_task_;

  bool retain_tasks_;
  int status_;
  
  pthread_mutex_t mutex_complete_;
  pthread_cond_t complete_cond_;
  shared_ptr<GraphMetadata> graph_metadata_;
  vector<int> tasks_order_;
public:
  static Graph* Create(Platform* platform);
  Task* end() { return end_; }
};
struct CommData3D
{
    uint32_t from_id;  
    uint32_t to_id;  
    uint32_t mem_id;
    size_t size;
};
class GraphMetadata {
public:
    GraphMetadata(Graph *graph, int iterations=3) : graph_(graph), iterations_(iterations) {
        dep_adj_list_ = NULL;
        dep_adj_matrix_ = NULL;
        comm_task_data_= NULL;
        comm_task_data_size_ = 0;
        comm_task_adj_matrix_ = NULL;
        comp_task_adj_matrix_ = NULL;
        map_task_inputs_outputs();
        json_url_ = NULL;
    }
    void set_iterations(int iterations) { iterations_ = iterations; }
    void map_task_inputs_outputs();
    size_t comm_task_data_size() { return comm_task_data_size_; }
    CommData3D *comm_task_data() { return comm_task_data_; }
    int get_max_parallelism(void);
    bool exists_edge(unsigned long u, unsigned long v, int8_t * dep_matrix, int ntasks);
    void get_dependency_matrix(int8_t *dep_matrix=NULL, bool adj_matrix=true);
    void level_order_traversal(int8_t s, int ntasks, int8_t* dep_matrix);
    void get_3d_comm_data();
    void get_2d_comm_adj_matrix(size_t *comm_task_adj_matrix=NULL);
    void calibrate_compute_cost_adj_matrix(double *comp_task_adj_matrix=NULL, bool only_device_type=false);
    void get_3d_comm_time(double *obj_2_dev_dev_time, int *mem_ids, int iterations, bool pin_memory_flag);
    size_t count_mems() { return mem_index_hash_valid_.size(); }
    //map<unsigned long, vector<unsigned long> > & task_inputs_map() { return task_inputs_map_; }
    //map<unsigned long, vector<unsigned long> > & task_outputs_map() { return task_outputs_map_; }
    map<unsigned long, BaseMem *> & mem_index_hash() { return mem_index_hash_; }
    map<unsigned long, Task *> & task_uid_hash() { return task_uid_hash_; }
    void fetch_task_execution_schedules(int kernel_profile=false);
    void fetch_dataobject_execution_schedules();
    size_t task_schedule_count() { return task_schedule_count_; }
    size_t dataobject_schedule_count() { return dataobject_schedule_count_; }
    DataObjectProfile *dataobject_schedule_data() { return dataobject_schedule_data_; }
    TaskProfile *task_schedule_data() { return task_schedule_data_; }
    const char* json_url() { return json_url_; }
    void set_json_url(const char* path) { json_url_ = path; }

private:
    Graph *graph_;
    int iterations_;
    int max_level_; 
    int max_parallelism_; 
    int8_t *dep_adj_matrix_;
    int8_t *dep_adj_list_;
    CommData3D *comm_task_data_;
    TaskProfile *task_schedule_data_;
    size_t task_schedule_count_;
    DataObjectProfile *dataobject_schedule_data_;
    size_t dataobject_schedule_count_;
    size_t comm_task_data_size_;
    size_t *comm_task_adj_matrix_;
    double *comp_task_adj_matrix_;
    map<unsigned long, unsigned long> mem_flash_out_2_new_id_map_;
    map<unsigned long, unsigned long> mem_flash_out_new_id_2_mid_map_;
    map<unsigned long, set<unsigned long>> mem_flash_out_2_task_map_;
    map<unsigned long, set<unsigned long>> mem_flash_task_2_mem_ids_;
    map<unsigned long, unsigned long> task_uid_2_index_hash_;
    map<unsigned long, unsigned long> task_index_2_uid_hash_;
    map<unsigned long, Task *> task_uid_hash_;
    map<unsigned long, BaseMem *> mem_index_hash_;
    map<unsigned long, BaseMem *> mem_index_hash_valid_;
    map<unsigned long, unsigned long> mem_regions_2_dmem_hash_;
    map<unsigned long, vector<unsigned long>> output_tasks_map_;
    map<unsigned long, vector<unsigned long>> task_inputs_map_;
    map<unsigned long, vector<unsigned long>> task_outputs_map_;
    vector<vector<unsigned long>> levels_dag_;
    const char* json_url_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_GRAPH_H */

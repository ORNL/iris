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
class Graph: public Retainable<struct _iris_graph, Graph> {
public:
  Graph(Platform* platform);
  virtual ~Graph();

  void AddTask(Task* task);
  void Submit();
  void Complete();
  void Wait();

  Platform* platform() { return platform_; }
  std::vector<Task*>* tasks() { return &tasks_; }
  std::vector<Task*> & tasks_list() { return tasks_; }
  std::vector<Task*> formatted_tasks();
  int iris_tasks(iris_task *pv);
  int tasks_count() { return tasks_.size(); }
  bool is_retainable() { return retain_tasks_; }
  void enable_retainable() { retain_tasks_ = true; }
  void disable_retainable() { retain_tasks_ = false; }
  shared_ptr<GraphMetadata> get_metadata(int iterations=3);
private:
  Platform* platform_;
  Scheduler* scheduler_;
  std::vector<Task*> tasks_;

  //Task* start_;
  Task* end_;

  bool retain_tasks_;
  int status_;
  
  pthread_mutex_t mutex_complete_;
  pthread_cond_t complete_cond_;
  shared_ptr<GraphMetadata> graph_metadata_;
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
    }
    void set_iterations(int iterations) { iterations_ = iterations; }
    void map_task_inputs_outputs();
    size_t comm_task_data_size() { return comm_task_data_size_; }
    CommData3D *comm_task_data() { return comm_task_data_; }
    void get_dependency_matrix(int8_t *dep_matrix=NULL, bool adj_matrix=true);
    void get_3d_comm_data();
    void get_2d_comm_adj_matrix(size_t *comm_task_adj_matrix=NULL);
    void calibrate_compute_cost_adj_matrix(double *comp_task_adj_matrix=NULL, bool only_device_type=false);
private:
    Graph *graph_;
    int iterations_;
    int8_t *dep_adj_matrix_;
    int8_t *dep_adj_list_;
    CommData3D *comm_task_data_;
    size_t comm_task_data_size_;
    size_t *comm_task_adj_matrix_;
    double *comp_task_adj_matrix_;
    map<unsigned long, unsigned long> task_uid_2_index_hash_;
    map<unsigned long, Task *> task_uid_hash_;
    map<unsigned long, BaseMem *> mem_index_hash_;
    map<unsigned long, unsigned long> mem_regions_2_dmem_hash_;
    map<unsigned long, vector<unsigned long>> output_tasks_map_;
    map<unsigned long, vector<unsigned long>> task_inputs_map_;
    map<unsigned long, vector<unsigned long>> task_outputs_map_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_GRAPH_H */

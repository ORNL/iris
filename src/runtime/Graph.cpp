#include "Graph.h"
#include "Debug.h"
#include "BaseMem.h"
#include "Mem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include "Scheduler.h"
#include "Platform.h"
#include "Device.h"
#include "Command.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"
#include <vector>
#include <set>
#include <queue>
#include <typeinfo>
#include <algorithm>

#ifdef AUTO_PAR
#include "AutoDAG.h"
#endif
 
#define GET2D_INDEX(NCOL, I, J)  ((I)*(NCOL)+(J))
#define PRUNE_EDGES //Prunes end node incoming edges
using namespace std;
namespace iris {
namespace rt {

Graph::Graph(Platform* platform) {
  platform_ = platform;
  retain_tasks_ = false;
  graph_metadata_ = nullptr;
  scheduler_ = NULL;
  if (platform) scheduler_ = platform_->scheduler();
  status_ = IRIS_NONE;

  end_ = Task::Create(platform_, IRIS_TASK_PERM, "Graph");
  end_brs_task_ = *(end_->struct_obj());
  tasks_.push_back(end_);

  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_cond_init(&complete_cond_, NULL);

#ifdef AUTO_PAR
#ifdef AUTO_FLUSH
  platform_->get_auto_dag()->set_current_graph(this);
#endif
#endif

  set_object_track(platform_->graph_track_ptr());
  platform_->graph_track().TrackObject(this, uid());
}

void Graph::enable_retainable() 
{ 
    retain_tasks_ = true; 
    DisableRelease();
    Retain();
    //printf("Retained graph:%lu flag:%d\n", uid(), is_retainable());
}

void Graph::disable_retainable() 
{ 
    retain_tasks_ = false; 
    EnableRelease();
    Release();
    //printf("Disabled Retained graph:%lu flag:%d\n", uid(), is_retainable());
}

shared_ptr<GraphMetadata> Graph::get_metadata(int iterations)
{
    if (graph_metadata_ == nullptr)
        graph_metadata_ = make_shared<GraphMetadata>(this, iterations);
    return graph_metadata_;
}

Graph::~Graph() {
  //platform_->graph_track().UntrackObject(this, uid());
  pthread_mutex_destroy(&mutex_complete_);
  pthread_cond_destroy(&complete_cond_);
  //if (end_) delete end_;
#if 0
  vector<Task *> & tasks = tasks_list();
  Platform *platform = Platform::GetPlatform();
  for (std::vector<Task*>::iterator I = tasks.begin(), E = tasks.end(); I != E; ++I) {
    Task* task = *I;
    if (!platform->track().IsObjectExists(task)) continue;
    _trace("destruct Task:%lu:%s ref_cnt:%d", task->uid(), task->name(), task->ref_cnt());
    if (!task->IsRelease())
        task->ForceRelease();
  }
#endif
  _trace("graph released");
}
void Graph::GraphRelease(void *data) {
    Graph *graph = (Graph *)data;
    std::vector<Task*>* tasks = graph->tasks();
    for (std::vector<Task*>::iterator I = tasks->begin(), E = tasks->end(); I != E; ++I) {
      Task* task = *I;
      //printf("Graph task:%lu:%s retained ref_cnt:%d\n", task->uid(), task->name(), task->ref_cnt()-1);
      if (!task->IsRelease()) {
          task->EnableRelease();
          _debug2("Graph task:%lu:%s retained ref_cnt:%d", task->uid(), task->name(), task->ref_cnt()-1);
          task->Release();
      }
    }
    if (!graph->IsRelease()) {
        graph->EnableRelease();
        graph->Release();
    }
    else {
        // Graph needs to be released manually. 
        // TODO: In future this should be done automatically
        graph->Release();
    }
}
void Graph::AddTask(Task* task, unsigned long uid) {
  //printf("Graph:%lu Task:%lu:%s Is retainable:%d task->release:%d\n", this->uid(), task->uid(), task->name(), is_retainable(), task->IsRelease());
  if (is_retainable() && task->IsRelease()) { 
      task->Retain(); 
      task->DisableRelease(); 
      //printf("Graph:%lu Retaining Task:%lu:%s Is retainable:%d task->release:%d ref_cnt:%d\n", this->uid(), task->uid(), task->name(), is_retainable(), task->IsRelease(), task->ref_cnt());
  }
  tasks_.push_back(task);

#ifdef AUTO_PAR
#ifdef IGNORE_MANUAL
    task->platform()->get_auto_dag()->set_auto_dep();
#endif
#endif

  end_->AddDepend(task, uid);

#ifdef AUTO_PAR
#ifdef IGNORE_MANUAL
   task->platform()->get_auto_dag()->unset_auto_dep();
#endif
#endif


#ifdef AUTO_PAR
#ifdef AUTO_FLUSH
  task->set_graph(this);
#endif
#endif
}

void Graph::Submit() {
  status_ = IRIS_SUBMITTED;
}

std::vector<Task*> Graph::formatted_tasks() { 
    vector<Task*> out;
    for(size_t i=1; i<tasks_.size(); i++) {
      out.push_back(tasks_[i]);
    }
    out.push_back(tasks_[0]);
    return out;
}

int Graph::iris_tasks(iris_task *pv) { 
    int index=0;
    for(size_t i=1; i<tasks_.size(); i++) {
      Task *task = tasks_[i];
      pv[index++] = *(task->struct_obj());
    }
    pv[index++] = *(tasks_[0]->struct_obj());
    return index;
}
int Graph::enable_mem_profiling() {
    for(size_t i=1; i<tasks_.size(); i++) {
        Task *task = tasks_[i];
        if (task->cmd_kernel() != NULL) 
            task->cmd_kernel()->kernel()->set_profile_data_transfers();
        task->set_profile_data_transfers();
    }
    return IRIS_SUCCESS;
}
void Graph::set_order(int *order) {
    for(size_t i=0; i<tasks_.size(); i++)
        tasks_order_.push_back(order[i]);
}
void Graph::Complete() {
  pthread_mutex_lock(&mutex_complete_);
  status_ = IRIS_COMPLETE;
  pthread_cond_broadcast(&complete_cond_);
  pthread_mutex_unlock(&mutex_complete_);
}

void Graph::ResetMemories() {
    GraphMetadata gmeta(this);
    map<unsigned long, BaseMem *> & mems = gmeta.mem_index_hash();
    for(auto i : mems) {
        i.second->clear();
    }
}   

void Graph::Wait() {
  platform_->TaskWait(end_brs_task_);
}

Graph* Graph::Create(Platform* platform) {
  return new Graph(platform);
}
void GraphMetadata::calibrate_compute_cost_adj_matrix(double *comp_task_adj_matrix, bool only_device_type)
{
    vector<int> unique_devices;
    map<int, vector<int>> model_2_devices;
    Platform *platform = Platform::GetPlatform();
    int ndevs = platform->ndevs();
    int nplatforms = platform->nplatforms();
    for(int i=0; i<ndevs; i++) {
        Device *dev = platform->device(i);
        int model = dev->model();
        if (model_2_devices.find(model) == model_2_devices.end()) {
            model_2_devices.insert(make_pair(model, vector<int>()));
            unique_devices.push_back(model);
        }
        model_2_devices[model].push_back(i);
    }
    vector<Task *> tasks = graph_->formatted_tasks();
    int ntasks = tasks.size()+1;
    if (comp_task_adj_matrix == NULL) {
        comp_task_adj_matrix_ = (double *)calloc(ntasks*ndevs, sizeof(size_t));
        comp_task_adj_matrix = comp_task_adj_matrix_;
    }
    map<string, uint64_t> kernel_map;
    map<vector<uint64_t>, double> knobs_to_makespan;
    for(unsigned long index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
#ifdef ENABLE_DEBUG
        printf("****** Task:%s ******\n", task->name());
#endif
        vector<uint64_t> knobs;
        if (task->cmd_kernel() != NULL)  {
            string kname = task->cmd_kernel()->kernel()->name();
            if (kernel_map.find(kname) == kernel_map.end()) {
                size_t size = kernel_map.size();
#ifdef ENABLE_DEBUG
                printf("Inserting kname: %s in map size:%lu\n", kname.c_str(), size);
#endif
                kernel_map[kname] = (int)size;
            }
            knobs.push_back(kernel_map[kname]);
            for(int i=0; i<task->cmd_kernel()->kernel_nargs(); i++) {
                KernelArg* arg = task->cmd_kernel()->kernel_arg(i);
                if (arg->mem != NULL) continue;
                void *value = arg->value;
                uint64_t data64 = 0;
                switch(arg->size) {
                    case 1:  {
                        uint8_t data = *((uint8_t *)value);
                        data64 = (uint64_t) data;
                        break;
                             }
                    case 2:  {
                        uint16_t data = *((uint16_t *)value);
                        data64 = (uint64_t) data;
                        break;
                             }
                    case 4:  {
                        uint32_t data = *((uint32_t *)value);
                        data64 = (uint64_t) data;
                        break;
                             }
                    case 8:  {
                        data64 = *((uint64_t *)value);
                        break;
                             }
                    default: _error("Size:%lu not yet handled", arg->size);
                }
                knobs.push_back(data64);
            }
        }
        task->DisableRelease();
        int dev_type_index = 0;
        for(int dev_index : unique_devices) {
            int dev_no = model_2_devices[dev_index][0];
            vector<uint64_t > dknobs = knobs;
            dknobs.push_back(dev_index);
            double time_duration = 0.0f, dtime;
            if (knobs_to_makespan.find(dknobs) == knobs_to_makespan.end()) {
                vector<double> multi_time;
                if (task->cmd_kernel() != NULL) {
                    string kname = task->cmd_kernel()->kernel()->name();
                    //Utils::PrintVectorFull<uint64_t>(dknobs, "Exploring new set of Knobs:");
                    for(int i=0; i<iterations_; i++) {
                        //printf("Running for iteration:%d kname:%s devno:%d\n", i, kname.c_str(), dev_no);
                        int ndepends = task->ndepends();
                        string tname = task->name();
                        string prev_tname = tname;
                        tname = tname + "-i" + std::to_string(i)+"-d"+std::to_string(dev_no);
                        task->set_ndepends(0);
                        task->set_name(tname.c_str());
                        task->cmd_kernel()->kernel()->set_task_name(tname.c_str());
                        _debug2("Before Task :%lu:%s ref_cnt:%d\n", task->uid(), task->name(), task->ref_cnt());
                        platform->TaskSubmit(task, dev_no, NULL, 1);
                        _debug2("After Task :%lu:%s ref_cnt:%d\n", task->uid(), task->name(), task->ref_cnt());
                        task->set_name(prev_tname.c_str());
                        dtime = task->cmd_kernel()->time_duration();
                        task->set_ndepends(ndepends);
                        multi_time.push_back(dtime);
                    }
                }
                for(double dtime : multi_time) {
                    time_duration += dtime;
                }
                time_duration = time_duration / iterations_;
#ifdef ENABLE_DEBUG
                printf("Inserting: ");
                for(uint64_t kn : dknobs) {
                    printf(" %ld", kn);
                }
                printf(" Time: %lf\n", time_duration);
#endif
                knobs_to_makespan.insert(make_pair(dknobs, time_duration));
            }
            else {
                time_duration = knobs_to_makespan[dknobs];
#ifdef ENABLE_DEBUG
                printf("Extracting: ");
                for(uint64_t kn : dknobs) {
                    printf(" %ld", kn);
                }
                printf(" Time: %lf\n", time_duration);
#endif
            }
            if (only_device_type) {
                comp_task_adj_matrix[GET2D_INDEX(nplatforms, index+1, dev_type_index)] = time_duration;
            }
            else {
                for(int dev_no : model_2_devices[dev_index]) {
                    comp_task_adj_matrix[GET2D_INDEX(ndevs, index+1, dev_no)] = time_duration;
                }
            }
            dev_type_index++;
        }
        task->EnableRelease();
    }   
#ifdef ENABLE_DEBUG
    int nentries = ndevs;
    if (only_device_type)
        nentries = (int)unique_devices.size();
    printf("N Entries:%d %d uniques:%d %ld\n", only_device_type, ndevs, nentries, unique_devices.size());
    Utils::PrintMatrixLimited<double>(comp_task_adj_matrix, ntasks, nentries, "Task Computation data(C++)-");
#endif
}
void GraphMetadata::fetch_dataobject_execution_schedules()
{
    vector<DataObjectProfile> results;
    vector<Task *> tasks = graph_->formatted_tasks();
    for(unsigned long index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
        if (task->cmd_kernel() != NULL && task->cmd_kernel()->kernel() != NULL && task->cmd_kernel()->kernel()->is_profile_data_transfers()) {
            Kernel *kernel = task->cmd_kernel()->kernel();
            for(auto i : kernel->in_mem_profiles()) {
                results.push_back(i);
            }
        }
        for(auto i : task->out_mem_profiles()) {
            results.push_back(i);
        }
    }
    dataobject_schedule_data_ = (DataObjectProfile *) malloc(sizeof(DataObjectProfile)*results.size());
    dataobject_schedule_count_ = results.size();
    std::copy(results.begin(), results.end(), dataobject_schedule_data_);
}
void GraphMetadata::fetch_task_execution_schedules(int kernel_profile)
{
    vector<TaskProfile> results;
    vector<Task *> tasks = graph_->formatted_tasks();
    for(unsigned long index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
        if (!kernel_profile) 
            results.push_back({(uint32_t) task->uid(), (uint32_t)task->dev()->devno(), task->time_start(), task->time_end()});
        else if (task->cmd_kernel() != NULL && task->cmd_kernel()->kernel() != NULL) {
            Command *cmd = task->cmd_kernel();
            results.push_back({(uint32_t) task->uid(), (uint32_t)task->dev()->devno(), cmd->time_start(), cmd->time_end()});
        }
    }
    task_schedule_data_ = (TaskProfile*) malloc(sizeof(TaskProfile)*results.size());
    task_schedule_count_ = results.size();
    std::copy(results.begin(), results.end(), task_schedule_data_);
}
void GraphMetadata::map_task_inputs_outputs()
{
    vector<Task *> tasks = graph_->formatted_tasks();
    set<unsigned long> init_reset_mems_covered;
    for(unsigned long index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
        task_uid_2_index_hash_[task->uid()] = index; 
        task_index_2_uid_hash_[index] = task->uid(); 
        task_uid_hash_[task->uid()] = task;
    }
    // Iterate through each task and track its input and output DMEM objects
    for(uint32_t index=0; index<(uint32_t)tasks.size(); index++) {
        Task *task = tasks[index];
        unsigned long uid = task->uid();
        //printf("Task: %lu\n", uid);
        vector<unsigned long> input_mems;
        set<unsigned long> reset_mems;
        vector<unsigned long> flush_input_mems;
        vector<unsigned long> output_mems;
        for(int di=0; di<task->ndepends(); di++) {
            unsigned long duid = task->depend(di)->uid();
            if (output_tasks_map_.find(duid) == output_tasks_map_.end()) 
                output_tasks_map_.insert(make_pair(duid, vector<unsigned long>()));
            output_tasks_map_[duid].push_back(uid);
        }
        for(Command *cmd : task->reset_mems()) {
            BaseMem *mem = (BaseMem *)cmd->mem();
            reset_mems.insert(mem->uid());
        }
        for(int ci=0; ci<task->ncmds(); ci++) {
            Command *cmd = task->cmd(ci);
            unsigned long muid;
            switch (cmd->type()) {
              case IRIS_CMD_H2D:          
              case IRIS_CMD_H2DNP: 
                  muid = cmd->mem()->uid();       
                  input_mems.push_back(muid);
                  if (mem_index_hash_.find(muid) == mem_index_hash_.end())
                      mem_index_hash_[muid] = cmd->mem(); 
                  //_info(" mid:%lu is added", uid);
                  break;
              case IRIS_CMD_MEM_FLUSH:    
                  // Special case
                  muid = cmd->mem()->uid();       
                  if (mem_flash_out_2_task_map_.find(muid) == mem_flash_out_2_task_map_.end())
                      mem_flash_out_2_task_map_.insert(make_pair(muid, set<unsigned long>()));
                  mem_flash_out_2_task_map_[muid].insert(uid);
                  if (mem_flash_out_2_new_id_map_.find(muid) == mem_flash_out_2_new_id_map_.end()) {
                      if (task->cmd_kernel() == NULL) 
                          mem_flash_out_2_new_id_map_[muid] = iris_create_new_uid();
                      else
                          mem_flash_out_2_new_id_map_[muid] = muid;
                      mem_flash_out_new_id_2_mid_map_[mem_flash_out_2_new_id_map_[muid]] = muid;
                      //printf("MEM FLASH %ld %ld\n", mem_flash_out_2_new_id_map_[muid], muid);
                  }
                  if (mem_flash_task_2_mem_ids_.find(uid) == mem_flash_task_2_mem_ids_.end()) 
                      mem_flash_task_2_mem_ids_.insert(make_pair(uid, set<unsigned long>()));
                  mem_flash_task_2_mem_ids_[uid].insert(muid);
                  output_mems.push_back(mem_flash_out_2_new_id_map_[muid]);
                  if (mem_index_hash_.find(muid) == mem_index_hash_.end())
                      mem_index_hash_[muid] = cmd->mem(); 
                  flush_input_mems.push_back(muid);
                  break;
              case IRIS_CMD_D2H:          
                  muid = cmd->mem()->uid();       
                  output_mems.push_back(muid);
                  flush_input_mems.push_back(muid);
                  if (mem_index_hash_.find(muid) == mem_index_hash_.end())
                      mem_index_hash_[muid] = cmd->mem(); 
                  //_info(" mid:%lu is added", muid);
                  if (cmd->mem()->GetMemHandlerType() == IRIS_DMEM_REGION) {
                      DataMemRegion *rdmem = (DataMemRegion *) cmd->mem();
                      DataMem *dmem = rdmem->get_dmem();
                      unsigned long dmem_id = dmem->uid();
                      if (mem_index_hash_.find(dmem_id) == mem_index_hash_.end())
                          mem_index_hash_[dmem_id] = dmem; 
                      //_info(" mid:%lu is added", dmem_id);
                      if (mem_regions_2_dmem_hash_.find(muid) == mem_regions_2_dmem_hash_.end())
                         mem_regions_2_dmem_hash_[muid] = dmem_id;
                  }
                  break;
              default: break;
            }
        }
        task_inputs_map_[uid]  = input_mems;
        task_outputs_map_[uid] = output_mems;
        if (task->cmd_kernel() == NULL) {
            if (output_mems.size()==0) 
                task_outputs_map_[uid] = input_mems;
            if (input_mems.size() == 0)
                task_inputs_map_[uid]  = flush_input_mems;
        }
        else {
            set<unsigned long> input_mems_set(input_mems.begin(), input_mems.end());
            set<unsigned long> output_mems_set(output_mems.begin(), output_mems.end());
            Command *cmd = task->cmd_kernel();
            int kernel_nargs = cmd->kernel_nargs();
            KernelArg* args = cmd->kernel_args();
            for (int i = 0; i < kernel_nargs; i++) {
                KernelArg* arg = cmd->kernel_arg(i);
                if (arg->mem == NULL) continue;
                BaseMem *mem = arg->mem;
                unsigned long mid = mem->uid();
                //printf("        Accessing mid:%lu\n", mid);
                if (mem_index_hash_.find(mid) == mem_index_hash_.end())
                    mem_index_hash_[mid] = mem;
                //_info(" mid:%lu is added", mid);
                int mode = arg->mode;
                if (mode == iris_r || mode == iris_rw) {
                    // If task is having reset command, its associated memory object will get reset and 
                    // it doesn't require data transfer
                    bool is_init_reset = mem->is_reset();
                    if (mem->GetMemHandlerType() == IRIS_DMEM) {
                        DataMem *dmem = (DataMem *)mem;
                        bool has_regions = dmem->is_regions_enabled();
                        if (has_regions) {
                            for(int k=0; k<dmem->get_n_regions(); k++) {
                                DataMemRegion *rdmem = dmem->get_region(k);
                                unsigned long rdmem_id = rdmem->uid();
                                if (input_mems_set.find(rdmem_id) == input_mems_set.end() && 
                                        reset_mems.find(rdmem_id) == reset_mems.end() &&
                                        (!is_init_reset || 
                                         (is_init_reset && 
                                          init_reset_mems_covered.find(rdmem_id) != init_reset_mems_covered.end()))) {
                                    task_inputs_map_[uid].push_back(rdmem_id);
                                }
                                if (is_init_reset)
                                    init_reset_mems_covered.insert(rdmem_id); 
                            }
                        }
                        else {
                            if (input_mems_set.find(mid) == input_mems_set.end() && 
                                    reset_mems.find(mid) == reset_mems.end() &&
                                    (!is_init_reset || 
                                     (is_init_reset && 
                                      init_reset_mems_covered.find(mid) != init_reset_mems_covered.end()))) {
                                task_inputs_map_[uid].push_back(mid);
                            }
                            if (is_init_reset)
                                init_reset_mems_covered.insert(mid); 
                        }
                    }
                    else {
                        if (mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
                            DataMemRegion *rdmem = (DataMemRegion *) mem;
                            DataMem *dmem = rdmem->get_dmem();
                            is_init_reset = dmem->is_reset();
                        }
                        if (input_mems_set.find(mid) == input_mems_set.end() && 
                                reset_mems.find(mid) == reset_mems.end() &&
                                (!is_init_reset || 
                                 (is_init_reset && 
                                  init_reset_mems_covered.find(mid) != init_reset_mems_covered.end()))) {
                            task_inputs_map_[uid].push_back(mid);
                        }
                        if (is_init_reset)
                            init_reset_mems_covered.insert(mid); 
                    }
                }
                if (mode == iris_w || mode == iris_rw) {
                    if (output_mems_set.find(mid) == output_mems_set.end())
                        task_outputs_map_[uid].push_back(mid);
                }
                if (mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
                    DataMemRegion *rdmem = (DataMemRegion *) mem;
                    DataMem *dmem = rdmem->get_dmem();
                    unsigned long dmem_id = dmem->uid();
                    if (mem_index_hash_.find(dmem_id) == mem_index_hash_.end())
                        mem_index_hash_[dmem_id] = dmem; 
                    //_info(" dmem mid:%lu is added", dmem_id);
                    if (mem_regions_2_dmem_hash_.find(rdmem->uid()) == mem_regions_2_dmem_hash_.end())
                       mem_regions_2_dmem_hash_[rdmem->uid()] = dmem_id;
                }
            }
        }
    }
    if (tasks.size()>0) {
        int index = tasks.size()-1;
        Task *task = tasks[index];
        unsigned long uid = task->uid();
        // Special node with name: Graph
        // This is a end node which have dependencies
        // But, it doesn't have memory inputs
        for (auto i : mem_flash_out_2_task_map_) {
            unsigned long mid = i.first;
            unsigned long tmp_mid = mem_flash_out_2_new_id_map_[mid];
            task_inputs_map_[uid].push_back(tmp_mid);
        }
        //printf("%s:%d Task:%s:%lu ndepends:%lu in:%lu out:%lu\n", __func__, __LINE__, task->name(), task->uid(), task->ndepends(), task_inputs_map_[uid].size(), task_outputs_map_[uid].size());
    }
    for( auto i : mem_index_hash_) {
        BaseMem *rdmem = i.second;
        if (rdmem->GetMemHandlerType() == IRIS_DMEM && 
                ((DataMem*)rdmem)->is_regions_enabled()) continue;
        mem_index_hash_valid_[i.first] = i.second;
    }
    for(auto i : mem_flash_out_2_new_id_map_) {
        if (mem_flash_out_2_new_id_map_[i.first] != i.second)
            mem_index_hash_valid_[i.second] = NULL;
    }
#ifdef ENABLE_DEBUG
    for(uint32_t index=0; index<(uint32_t)tasks.size(); index++) {
        Task *task = tasks[index];
        unsigned long uid = task->uid();
        printf("%s:%d Task:%s:%lu ndepends:%lu in:%lu out:%lu\n", __func__, __LINE__, task->name(), task->uid(), task->ndepends(), task_inputs_map_[uid].size(), task_outputs_map_[uid].size());
        for(unsigned long mid : task_inputs_map_[uid]) {
            //BaseMem *mem = mem_index_hash_[mid];
            printf("                Input mem:%lu size:%lu\n", mid, 0);
        }
        for(unsigned long mid : task_outputs_map_[uid]) {
            //BaseMem *mem = mem_index_hash_[mid];
            printf("                Output mem:%lu size:%lu\n", mid, 0);
        }
    }
#endif
}
void GraphMetadata::get_dependency_matrix(int8_t *dep_matrix, bool adj_matrix) {
  vector<Task *> tasks = graph_->formatted_tasks();
  int ntasks = tasks.size()+1;
  if (dep_matrix == NULL && adj_matrix) {
      dep_adj_matrix_ = (int8_t *)calloc(ntasks*ntasks, sizeof(int8_t));
      dep_matrix = dep_adj_matrix_;
  }
  if (dep_matrix == NULL && !adj_matrix) {
      dep_adj_list_   = (int8_t *)calloc(ntasks*(ntasks+1), sizeof(int8_t));
      dep_matrix = dep_adj_list_;
  }
  for(uint32_t t=0; t<(uint32_t)tasks.size(); t++) {
      Task *task = tasks[t];
      //printf("Tasks %s\n", task->name());
      for(int i=0; i<task->ndepends(); i++) {
          Task *dtask = task->depend(i);
          //printf("Tasks %s depend %s depend index %lu task index %d\n", task->name(), dtask->name(), task_uid_2_index_hash_[dtask->uid()], task_uid_2_index_hash_[task->uid()]);
          //printf("Depend %d\n", task == graph_->end());
          //printf("Map %d\n", output_tasks_map_[dtask->uid()].size());
#ifdef PRUNE_EDGES 
          if (task == graph_->end() && 
                  output_tasks_map_[dtask->uid()].size() > 1)
              continue;
#endif
          //printf("Tasks %s depend %s\n", task->name(), dtask->name());
          unsigned long did = task_uid_2_index_hash_[dtask->uid()];
          if (! adj_matrix) {
              dep_matrix[GET2D_INDEX(ntasks+1, t+1, 0)]++;
              int adj_list_index = dep_matrix[GET2D_INDEX(ntasks+1, t+1, 0)];
              dep_matrix[GET2D_INDEX(ntasks+1, t+1, adj_list_index)] = did; 
          }
          else {
              //printf ("index did %d\n",GET2D_INDEX(ntasks, t+1, did+1)); 
              dep_matrix[GET2D_INDEX(ntasks, t+1, did+1)] = 1; 
          }
      }
      if (task->ndepends() == 0) {
          if (! adj_matrix) {
              dep_matrix[GET2D_INDEX(ntasks+1, t+1, 0)]++;
              int adj_list_index = dep_matrix[GET2D_INDEX(ntasks+1, t+1, 0)];
              dep_matrix[GET2D_INDEX(ntasks+1, t+1, adj_list_index)] = 0;
          }
          else {
              //printf ("index %d\n",GET2D_INDEX(ntasks, t+1, 0)); 
              dep_matrix[GET2D_INDEX(ntasks, t+1, 0)] = 1;
          }
      }

  }
 //level_order_traversal(0, ntasks, dep_matrix); 
}

void GraphMetadata::level_order_traversal(int8_t s, int ntasks, int8_t* dep_matrix)
{
    // a queue for Level Order Traversal
    queue<int> q;
    vector<unsigned long> vec;
 
    // Stores if the current node is visited
    vector<Task *> tasks = graph_->formatted_tasks();
    ntasks = tasks.size()+1;
    std::vector<int> v_level(ntasks);
    int max_level = 0;
    for ( int i = 1; i < ntasks; i++) v_level.at(i) = 0;
    /*for (auto& a : v_level) {
	
    }*/
    v_level.at(0) = 0; 
    for ( int j = 1; j < ntasks; j++) {
    	for ( int i = 1; i < ntasks; i++) {
            if (dep_matrix[i * ntasks + j] !=  0) {
		 v_level.at(i-1) = v_level.at(j-1) + 1;
                 //std::cout << "I : " << i-1 << " j " << j-1 <<  "\n";
                 //std::cout << "here i=" << i-1 <<" cur level i : " << v_level.at(i-1) << " here j=" <<j-1 << " cur level j " << v_level.at(j-1) <<  "\n";
            //if (dep_matrix[i * ntasks + v + 1] !=  0 && !visited[i-1]) {
	    }
	}
    }
    max_level_ = 0; 
    for ( auto & i : v_level) 
	    if (max_level_ < i) max_level_ = i;
    //vector<unsigned long> vec;
    for ( int i = 0; i < max_level_; i++) {
    	for ( int j = 1; j < ntasks; j++) {
           if (v_level.at(j-1) == i) vec.push_back(j-1);	    
	}
        levels_dag_.push_back(vec);
	vec.clear();
    }

    //max_level_ = i-1; 

}

/*
void GraphMetadata::level_order_traversal(int8_t s, int ntasks, int8_t* dep_matrix)
{
    // a queue for Level Order Traversal
    queue<int> q;
    vector<unsigned long> vec;
 
    // Stores if the current node is visited
    vector<Task *> tasks = graph_->formatted_tasks();
    ntasks = tasks.size()+1;
    std::vector<bool> visited(ntasks+1);
 
    q.push(s);
 
    // -1 marks the end of level
    q.push(-1);
    visited[s] = true;
    int levels = 0;
    while (!q.empty()) {
 
        // Dequeue a vertex from queue
        int v = q.front();
        q.pop();
 
        // If v marks the end of level
        if (v == -1) {
            if (!q.empty())
                q.push(-1);
 
            // Print a newline character
	    levels += 1;
	    levels_dag_.push_back(vec);
	    vec.clear();
            continue;
        }
 
        // store the current vertex
	vec.push_back(v);
        //cout << task_index_2_uid_hash_[v] << " ";
        //cout << task_uid_hash_[task_index_2_uid_hash_[v]]->name() << " ";
 
        // Add the child vertices of the current node in queue
        for (int i = 1; i < ntasks + 1; i++) {
            if (dep_matrix[i * ntasks + v + 1] !=  0 && !visited[i-1]) {
            //if (dep_matrix[i * ntasks + v + 1] !=  0 && !visited[i-1]) {
                visited[i - 1] = true;
                q.push(i - 1);
            }
        }
    }

    max_level_ = levels; 

}
*/

bool GraphMetadata::exists_edge(unsigned long u, 
	unsigned long v, int8_t * dep_matrix, int ntasks){
   if ( dep_matrix[(u + 1) * ntasks + (v + 1)] > 0) return true;	
   if ( dep_matrix[(v + 1) * ntasks + (u + 1)] > 0) return true;	

   return false;
} 

int GraphMetadata::get_max_parallelism()
{ 
  size_t max_par = 0;
  vector<Task *> tasks = graph_->formatted_tasks();
  int ntasks = tasks.size()+1;
  int8_t * dep_matrix = (int8_t *)calloc(ntasks*ntasks, sizeof(int8_t));
  get_dependency_matrix(dep_matrix, true);
  for (int i = 0 ; i < ntasks; i++) {
  	for (int j = 0 ; j < ntasks; j++) {
  		printf("%d ", dep_matrix[i * ntasks + j]);
	}
  	printf("\n");
  }
  level_order_traversal(0, ntasks, dep_matrix);

  std::cout << "max levels : " << max_level_ << "\n";
  int l = 0; 
  for (auto& i : levels_dag_){
      std::cout << "Level " << l++ << " : ";
      for (auto& j : i) {
	  std::cout <<  j << " ";
          //cout << task_uid_hash_[task_index_2_uid_hash_[j]]->name() << " ";
      }
      std::cout << "\n";
  }
	
  vector<unsigned long> merged_dag;
  int count_level = 0 ; 
  int total_parallelism = 0 ; 
  double average_parallelism = 0 ; 
  for (auto& i : levels_dag_){
      for (auto& j : i) {
	  //std::cout << "for value "  << j << " and size of merged_dag " << merged_dag.size() << " \n";
	  if(std::find(merged_dag.begin(), merged_dag.end(), j) 
		                   != merged_dag.end()) {
	       continue;
	  }

	  int count = 0 ;
          for (auto& k : merged_dag) {
	      //std::cout << "here j " << j << " k " << k << " \n";
              if(exists_edge(j, k, dep_matrix, ntasks)) {
		  merged_dag.erase(merged_dag.begin() + count);
	          //std::cout << "erased k  " << k << " \n";
		  count--;
	      }
	      count++;
	  }
 	  merged_dag.push_back(j);
	  //std::cout << "pushed j  " << j << " \n";
	
 /*	
          if (merged_dag.size() != 0) {
	      int count = 0 ;
              for (auto& k : merged_dag) {
		  //std::cout << "value of k " << k << " \n";
		  //std::cout << "type of k " << typeid(k).name() << " \n";
		  std::cout << "here j " << j << " k " << k << " \n";
                  if(exists_edge(j, k, dep_matrix, ntasks)) {
		      //merged_dag.erase(k); 
		      merged_dag.erase(merged_dag.begin() + count);
		      std::cout << "erased k " << k << " \n";
	 	      if(std::find(merged_dag.begin(), merged_dag.end(), j) 
		                   == merged_dag.end()) {
 	                  merged_dag.push_back(j);
		          std::cout << "pushed j " << j << " \n";
                          count++; 
		      } else  {
                          count++; 
		          continue;
		      }
 		  } else {
		  }	
              } // for 
	  } else {
 	      merged_dag.push_back(j);
	      std::cout << "pushed j int empty " << j << " \n";
	  }
	*/
      }
      /*std::cout << "After merging Level :  " << count_level++ << "\n" ;
      for (auto& k : merged_dag)
	  std::cout << k << " " ;
      std::cout << " \n" ; */
      if (max_par < merged_dag.size()) max_par = merged_dag.size();
      total_parallelism += merged_dag.size();
   }
   average_parallelism = (double) total_parallelism / max_level_;
   std::cout << "Maximum theoretical parallelism:  " << max_par << "\n" ;
   std::cout << "Height of the DAG:  " << max_level_ << "\n" ;
   std::cout << "Average parallelism of a DAG:  " << average_parallelism << "\n" ;
return 0;
}

void GraphMetadata::get_2d_comm_adj_matrix(size_t *comm_task_adj_matrix)
{
    vector<Task *> tasks = graph_->formatted_tasks();
    int ntasks = tasks.size()+1;
    if (comm_task_adj_matrix == NULL) {
        comm_task_adj_matrix_ = (size_t *)calloc(ntasks*ntasks, sizeof(size_t));
        comm_task_adj_matrix = comm_task_adj_matrix_;
    }
    for(uint32_t index=0; index<(uint32_t)tasks.size(); index++) {
        Task *each_task = tasks[index];
        vector<unsigned long> & lst1_v = task_inputs_map_[each_task->uid()];
        set<unsigned long> lst1(lst1_v.begin(), lst1_v.end());
        //printf("Task:%s:%lu:%d ndepends:%lu lst1_v:%lu %lu\n", each_task->name(), each_task->uid(), index+1, each_task->ndepends(), lst1.size(), lst1_v.size());
        for(unsigned long mid : lst1) {
            BaseMem *mem = mem_index_hash_[mid];
            //printf("            probing mem:%lu size:%lu\n", mid, mem->size());
        }
        set<unsigned long> all_covered_mem;
        for(int di=0; di<each_task->ndepends(); di++) {
            Task *d_task = each_task->depend(di);
#ifdef PRUNE_EDGES 
            if (each_task == graph_->end() && 
                    mem_flash_task_2_mem_ids_.find(d_task->uid()) ==  
                    mem_flash_task_2_mem_ids_.end()) {
                // Special case. if each_task is end task and 
                // the d_task is not flushing out any
                continue;
            }
#endif
            unsigned long d_index = task_uid_2_index_hash_[d_task->uid()];
            vector<unsigned long> & lst2_v = task_outputs_map_[d_task->uid()];
            set<unsigned long> lst2(lst2_v.begin(), lst2_v.end());
            set<unsigned long> mem_list;
            //for(unsigned long mid : lst2) {
            //    BaseMem *mem = mem_index_hash_[mid];
                //printf("                dependency probing mem:%lu size:%lu\n", mid, mem->size());
            //}
            for(unsigned long mid : lst1) {
                if (lst2.find(mid) != lst2.end())
                    mem_list.insert(mid);
            } 
            set<unsigned long> dmem_region_lst;
            for(unsigned long mid : lst2) {
                if (mem_regions_2_dmem_hash_.find(mid) != 
                        mem_regions_2_dmem_hash_.end()) {
                    dmem_region_lst.insert(mid);
                }
            }
            size_t size = 0;
            for(unsigned long mid : mem_list) {
                unsigned long tmp_mid = mid;
                //printf("Processing mid:%ld\n", mid);
#ifdef PRUNE_EDGES 
                if (each_task == graph_->end() && 
                        mem_flash_out_new_id_2_mid_map_.find(mid) !=
                        mem_flash_out_new_id_2_mid_map_.end()) {
                    tmp_mid = mem_flash_out_new_id_2_mid_map_[mid];
                    //printf("    Converting Processing mid:%ld to tmp_mid:%ld\n", mid, tmp_mid);
                }
#endif
                BaseMem *mem = mem_index_hash_[tmp_mid];
#ifdef PRUNE_EDGES 
                if (each_task == graph_->end() && 
                        mem_flash_out_2_task_map_.find(mem->uid()) ==  
                        mem_flash_out_2_task_map_.end()) {
                    continue;
                }
                if (each_task == graph_->end() && 
                        mem_flash_out_2_task_map_[mem->uid()].find(d_task->uid()) ==  
                        mem_flash_out_2_task_map_[mem->uid()].end()) {
                    continue;
                }
#endif
                //printf("Common mem:%lu mid:%lu\n", mem->uid(), mid);
                size += mem->size();
                all_covered_mem.insert(mid);
            }
            //_info("partial updating %lu-> %lu size:%lu\n", index+1, d_index+1, size);
            for(unsigned long mid : dmem_region_lst) {
                // Check if input is part of DMEM region
                BaseMem *rdmem = mem_index_hash_[mid];
                unsigned long dmem_index = mem_regions_2_dmem_hash_[mid];
                //printf("                dmem dependency probing mem:%lu size:%lu dmem_index:%lu\n", mid, rdmem->size(), dmem_index);
                if (lst1.find(dmem_index) != lst1.end() && 
                        all_covered_mem.find(mid) == all_covered_mem.end()) { 
                    BaseMem *mem = mem_index_hash_[mid];
                    all_covered_mem.insert(dmem_index);
                    all_covered_mem.insert(mid);
                    size += mem->size();
                }
            }
            //_info("updating %lu-> %lu size:%lu\n", index+1, d_index+1, size);
            comm_task_adj_matrix[GET2D_INDEX(ntasks, index+1, d_index+1)] += size;
        }
        // Some of the uncovered memory inputs are not covered
        size_t size = 0;
        for(unsigned long mid : lst1) {
            if (all_covered_mem.find(mid) == all_covered_mem.end()) {
                unsigned long tmp_mid = mid;
                if ( mem_flash_out_new_id_2_mid_map_.find(mid) !=
                        mem_flash_out_new_id_2_mid_map_.end()) {
                    tmp_mid = mem_flash_out_new_id_2_mid_map_[mid];
                }
                BaseMem *mem = mem_index_hash_[tmp_mid];
                size += mem->size();
            }
        }
        if (size > 0)  {
            //_info("updating %lu-> %lu size:%lu\n", index+1, 0, size);
            comm_task_adj_matrix[GET2D_INDEX(ntasks, index+1, 0)] += size;
        }
    }
#ifdef ENABLE_DEBUG
    Utils::PrintMatrixLimited<size_t>(comm_task_adj_matrix, ntasks, ntasks, "Task Communication data(C++)");
#endif
}
void GraphMetadata::get_3d_comm_time(double *obj_2_dev_dev_time, int *mem_ids, int iterations, bool pin_memory_flag)
{
    int total_mems = mem_index_hash_valid_.size();
    int ndevs = graph_->platform()->ndevs()+1;
    int index = 0;
    map<size_t, double *> processed;
    for(auto i : mem_index_hash_valid_) {
        unsigned long mid = i.first;
        mem_ids[index] = i.first;
        BaseMem *mem = i.second;
        if (mem == NULL && 
                mem_flash_out_new_id_2_mid_map_.find(mid) != mem_flash_out_new_id_2_mid_map_.end()) {
            mem = mem_index_hash_valid_[mem_flash_out_new_id_2_mid_map_[mid]];
        }
        size_t size = mem->size();
        //printf("Mem id:%d size:%lu uid:%lu\n", index, size, mem->uid());
        double *comp_time_matrix = obj_2_dev_dev_time + GET2D_INDEX(ndevs * ndevs, index, 0);
        if (processed.find(size) != processed.end()) {
            memcpy(comp_time_matrix, processed[size], sizeof(double)*ndevs*ndevs);
        }
        else {
            graph_->platform()->CalibrateCommunicationMatrix(comp_time_matrix, size, iterations, pin_memory_flag);
            processed[size] = comp_time_matrix;
        }
        //printf("        Completed Mem id:%d size:%lu uid:%lu\n", index, size, mem->uid());
        index++;
    }
}
void GraphMetadata::get_3d_comm_data()
{
    CommData3D *comm_task_data  = NULL;
    vector<Task *> tasks = graph_->formatted_tasks();
    int ntasks = tasks.size()+1;
    vector<CommData3D> results;
    for(uint32_t index=0; index<(uint32_t)tasks.size(); index++) {
        Task *each_task = tasks[index];
        vector<unsigned long> & lst1_v = task_inputs_map_[each_task->uid()];
        set<unsigned long> lst1(lst1_v.begin(), lst1_v.end());
        //printf("Task:%s:%lu ndepends:%lu lst1_v:%lu %lu\n", each_task->name(), each_task->uid(), each_task->ndepends(), lst1.size(), lst1_v.size());
        //for(unsigned long mid : lst1) {
            //BaseMem *mem = mem_index_hash_[mid];
            //printf("            probing mem:%lu size:%lu\n", mid, mem->size());
        //}
        set<unsigned long> all_covered_mem;
        for(int di=0; di<each_task->ndepends(); di++) {
            Task *d_task = each_task->depend(di);
#ifdef PRUNE_EDGES 
            if (each_task == graph_->end() && 
                    mem_flash_task_2_mem_ids_.find(d_task->uid()) ==  
                    mem_flash_task_2_mem_ids_.end()) {
                // Special case. the each_task is end task and 
                // the d_task is not flushing out any
                continue;
            }
#endif
            unsigned long d_index = task_uid_2_index_hash_[d_task->uid()];
            vector<unsigned long> & lst2_v = task_outputs_map_[d_task->uid()];
            set<unsigned long> lst2(lst2_v.begin(), lst2_v.end());
            set<unsigned long> mem_list;
            //for(unsigned long mid : lst2) {
                //BaseMem *mem = mem_index_hash_[mid];
                //printf("                dependency probing mem:%lu size:%lu\n", mid, mem->size());
            //}
            for(unsigned long mid : lst1) {
                if (lst2.find(mid) != lst2.end())
                    mem_list.insert(mid);
            } 
            set<unsigned long> dmem_region_lst;
            for(unsigned long mid : lst2) {
                if (mem_regions_2_dmem_hash_.find(mid) != 
                        mem_regions_2_dmem_hash_.end()) {
                    dmem_region_lst.insert(mid);
                }
            }
            size_t size = 0;
            for(unsigned long mid : mem_list) {
                unsigned long tmp_mid = mid;
#ifdef PRUNE_EDGES 
                if (each_task == graph_->end() && 
                        mem_flash_out_new_id_2_mid_map_.find(mid) !=
                        mem_flash_out_new_id_2_mid_map_.end()) {
                    tmp_mid = mem_flash_out_new_id_2_mid_map_[mid];
                }
#endif
                BaseMem *mem = mem_index_hash_[tmp_mid];
#ifdef PRUNE_EDGES 
                if (each_task == graph_->end() && 
                        mem_flash_out_2_task_map_.find(mem->uid()) ==  
                        mem_flash_out_2_task_map_.end()) {
                    continue;
                }
                if (each_task == graph_->end() && 
                        mem_flash_out_2_task_map_[mem->uid()].find(d_task->uid()) ==  
                        mem_flash_out_2_task_map_[mem->uid()].end()) {
                    continue;
                }
#endif
                //printf("Common mem:%lu mid:%lu\n", mem->uid(), mid);
                size += mem->size();
                CommData3D data = {(uint32_t)d_index+1, (uint32_t)index+1, (uint32_t)mid, mem->size()};
                results.push_back(data);
                all_covered_mem.insert(mid);
            }
            //_info("partial updating %lu-> %lu size:%lu\n", index+1, d_index+1, size);
            for(unsigned long mid : dmem_region_lst) {
                // Check if input is part of DMEM region
                BaseMem *rdmem = mem_index_hash_[mid];
                unsigned long dmem_index = mem_regions_2_dmem_hash_[mid];
                //printf("                dmem dependency probing mem:%lu size:%lu dmem_index:%lu\n", mid, rdmem->size(), dmem_index);
                if (lst1.find(dmem_index) != lst1.end() && 
                        all_covered_mem.find(mid) == all_covered_mem.end()) { 
                    BaseMem *mem = mem_index_hash_[mid];
                    all_covered_mem.insert(dmem_index);
                    all_covered_mem.insert(mid);
                    size += mem->size();
                    CommData3D data = {(uint32_t)d_index+1, (uint32_t)index+1, (uint32_t)mid, mem->size()};
                    results.push_back(data);
                }
            }
            //_info("updating %lu-> %lu size:%lu\n", index+1, d_index+1, size);
        }
        // Some of the uncovered memory inputs are not covered
        size_t size = 0;
        for(unsigned long mid : lst1) {
            if (all_covered_mem.find(mid) == all_covered_mem.end()) {
                unsigned long tmp_mid = mid;
                if (mem_flash_out_new_id_2_mid_map_.find(mid) !=
                        mem_flash_out_new_id_2_mid_map_.end()) {
                    tmp_mid = mem_flash_out_new_id_2_mid_map_[mid];
                }
                BaseMem *mem = mem_index_hash_[tmp_mid];
                size += mem->size();
                CommData3D data = {0, (uint32_t)index+1, (uint32_t)mid, mem->size()};
                results.push_back(data);
            }
        }
    }
    if (comm_task_data == NULL) {
        comm_task_data_ = (CommData3D*)calloc(results.size(), sizeof(CommData3D));
        comm_task_data = comm_task_data_;
    }
    comm_task_data_size_ = results.size();
    std::copy(results.begin(), results.end(), comm_task_data);
#ifdef ENABLE_DEBUG
    Utils::PrintMatrixLimited<size_t>(comm_task_data, ntasks, ntasks, "Task Communication data(C++)");
#endif
}

} /* namespace rt */
} /* namespace iris */

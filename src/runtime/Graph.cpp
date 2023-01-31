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
#define GET2D_INDEX(NCOL, I, J)  ((I)*(NCOL)+(J))

using namespace std;
namespace iris {
namespace rt {

Graph::Graph(Platform* platform) {
  platform_ = platform;
  retain_tasks_ = false;
  if (platform) scheduler_ = platform_->scheduler();
  status_ = IRIS_NONE;

  end_ = Task::Create(platform_, IRIS_TASK_PERM, "Graph");
  tasks_.push_back(end_);

  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_cond_init(&complete_cond_, NULL);
}

Graph::~Graph() {
  pthread_mutex_destroy(&mutex_complete_);
  pthread_cond_destroy(&complete_cond_);
  if (end_) delete end_;
}

void Graph::AddTask(Task* task) {
  if (is_retainable()) task->DisableRelease();
  tasks_.push_back(task);
  end_->AddDepend(task);
}

void Graph::Submit() {
  status_ = IRIS_SUBMITTED;
}

int Graph::iris_tasks(iris_task *pv) { 
    int index=0;
    for(Task *task : tasks_) {
      pv[index++] = task->struct_obj();
    }
    return index;
}

void Graph::Complete() {
  pthread_mutex_lock(&mutex_complete_);
  status_ = IRIS_COMPLETE;
  pthread_cond_broadcast(&complete_cond_);
  pthread_mutex_unlock(&mutex_complete_);
}

void Graph::Wait() {
  end_->Wait();
}

Graph* Graph::Create(Platform* platform) {
  return new Graph(platform);
}
void GraphMetadata::calibrate_compute_cost_adj_matrix(double *comp_task_adj_matrix)
{
    vector<int> unique_devices;
    map<int, vector<int>> model_2_devices;
    Platform *platform = Platform::GetPlatform();
    int ndevs = platform->ndevs();
    for(int i=0; i<ndevs; i++) {
        Device *dev = platform->device(i);
        int model = dev->model();
        if (model_2_devices.find(model) == model_2_devices.end()) {
            model_2_devices.insert(make_pair(model, vector<int>()));
            unique_devices.push_back(model);
        }
        model_2_devices[model].push_back(i);
    }
    vector<Task *> & tasks = graph_->tasks_list();
    int ntasks = tasks.size()+1;
    if (comp_task_adj_matrix == NULL) {
        comp_task_adj_matrix_ = (double *)calloc(ntasks*ndevs, sizeof(size_t));
        comp_task_adj_matrix = comp_task_adj_matrix_;
    }
    map<string, uint64_t> kernel_map;
    map<vector<uint64_t>, double> knobs_to_makespan;
    for(unsigned long index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
        vector<uint64_t> knobs;
        if (task->cmd_kernel() != NULL)  {
            string kname = task->cmd_kernel()->kernel()->name();
            if (kernel_map.find(kname) == kernel_map.end()) {
                size_t size = kernel_map.size();
                //printf("Inserting kname: %s in map size:%lu\n", kname.c_str(), size);
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
                    default: _error("Size:%d not yet handled", arg->size);
                }
                knobs.push_back(data64);
            }
        }
        for(int dev_index : unique_devices) {
            int dev_no = model_2_devices[dev_index][0];
            vector<uint64_t > dknobs = knobs;
            dknobs.push_back(dev_index);
            double time_duration = 0.0f, dtime;
            if (knobs_to_makespan.find(dknobs) == knobs_to_makespan.end()) {
                vector<double> multi_time;
                if (task->cmd_kernel() != NULL) {
                    string kname = task->cmd_kernel()->kernel()->name();
                    Utils::PrintVectorFull<uint64_t>(dknobs, "Exploring new set of Knobs:");
                    for(int i=0; i<iterations_; i++) {
                        printf("Running for iteration:%d kname:%s devno:%d\n", i, kname.c_str(), dev_no);
                        int ndepends = task->ndepends();
                        task->set_ndepends(0);
                        platform->TaskSubmit(task, dev_no, NULL, 1);
                        dtime = task->cmd_kernel()->time_duration();
                        task->set_ndepends(ndepends);
                        multi_time.push_back(dtime);
                    }
                }
                for(double dtime : multi_time) {
                    time_duration += dtime;
                }
                time_duration = time_duration / iterations_;
                knobs_to_makespan.insert(make_pair(dknobs, time_duration));
            }
            else {
                time_duration = knobs_to_makespan[dknobs];
            }
            for(int dev_no : model_2_devices[dev_index]) {
                comp_task_adj_matrix[GET2D_INDEX(ndevs, index+1, dev_no)] = time_duration;
            }
        }
    }   
    Utils::PrintMatrixLimited<double>(comp_task_adj_matrix, ntasks, ndevs, "Task Computation data(C++)");
}
void GraphMetadata::map_task_inputs_outputs()
{
    vector<Task *> & tasks = graph_->tasks_list();
    for(unsigned long index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
        task_uid_2_index_hash_[task->uid()] = index; 
        task_uid_hash_[task->uid()] = task;
    }
    set<unsigned long> output_flushes;
    for(int index=0; index<tasks.size(); index++) {
        Task *task = tasks[index];
        unsigned long uid = task->uid();
        vector<unsigned long> input_mems;
        vector<unsigned long> output_mems;
        for(int di=0; di<task->ndepends(); di++) {
            unsigned long duid = task->depend(di)->uid();
            if (output_tasks_map_.find(duid) == output_tasks_map_.end()) 
                output_tasks_map_.insert(make_pair(duid, vector<unsigned long>()));
            output_tasks_map_[duid].push_back(uid);
        }
        for(int ci=0; ci<task->ncmds(); ci++) {
            Command *cmd = task->cmd(ci);
            unsigned long uid;
            switch (cmd->type()) {
              case IRIS_CMD_H2D:          
              case IRIS_CMD_H2DNP: 
                  uid = cmd->mem()->uid();       
                  input_mems.push_back(uid);
                  if (mem_index_hash_.find(uid) == mem_index_hash_.end())
                      mem_index_hash_[uid] = cmd->mem(); 
                  //_info(" mid:%lu is added", uid);
                  break;
              case IRIS_CMD_MEM_FLUSH:    
                  output_flushes.insert(cmd->mem()->uid());
              case IRIS_CMD_D2H:          
                  uid = cmd->mem()->uid();       
                  output_mems.push_back(uid);
                  if (mem_index_hash_.find(uid) == mem_index_hash_.end())
                      mem_index_hash_[uid] = cmd->mem(); 
                  //_info(" mid:%lu is added", uid);
                  if (cmd->mem()->GetMemHandlerType() == IRIS_DMEM_REGION) {
                      DataMemRegion *rdmem = (DataMemRegion *) cmd->mem();
                      DataMem *dmem = rdmem->get_dmem();
                      unsigned long dmem_id = dmem->uid();
                      if (mem_index_hash_.find(dmem_id) == mem_index_hash_.end())
                          mem_index_hash_[dmem_id] = dmem; 
                      //_info(" mid:%lu is added", dmem_id);
                      if (mem_regions_2_dmem_hash_.find(uid) == mem_regions_2_dmem_hash_.end())
                         mem_regions_2_dmem_hash_[uid] = dmem_id;
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
                task_inputs_map_[uid]  = output_mems;
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
                if (mem_index_hash_.find(mid) == mem_index_hash_.end())
                    mem_index_hash_[mid] = mem;
                //_info(" mid:%lu is added", mid);
                int mode = arg->mode;
                if (mode == iris_r || mode == iris_rw) {
                    if (input_mems_set.find(mid) == input_mems_set.end())
                        task_inputs_map_[uid].push_back(mid);
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
        //printf("%s:%d Task:%s:%lu ndepends:%lu in:%lu out:%lu\n", __func__, __LINE__, task->name(), task->uid(), task->ndepends(), task_inputs_map_[uid].size(), task_outputs_map_[uid].size());
    }
    if (tasks.size()>0) {
        int index = 0;
        Task *task = tasks[index];
        unsigned long uid = task->uid();
        // Special node with name: Graph
        // This is a end node which have dependencies
        // But, it doesn't have memory inputs
        for (unsigned long mid : output_flushes) {
            task_inputs_map_[uid].push_back(mid);
        }
        //printf("%s:%d Task:%s:%lu ndepends:%lu in:%lu out:%lu\n", __func__, __LINE__, task->name(), task->uid(), task->ndepends(), task_inputs_map_[uid].size(), task_outputs_map_[uid].size());
    }
}
void GraphMetadata::get_dependency_graph() {
  vector<Task *> & tasks = graph_->tasks_list();
  int ntasks = tasks.size()+1;
  dep_adj_matrix_ = (uint32_t *)calloc(ntasks*ntasks, sizeof(uint32_t));
  dep_adj_list_   = (uint32_t *)calloc(ntasks*(ntasks+1), sizeof(uint32_t));
  for(int t=0; t<tasks.size(); t++) {
      Task *task = tasks[t];
      for(int i=0; i<task->ndepends(); i++) {
          Task *dtask = task->depend(i);
          unsigned long did = task_uid_2_index_hash_[dtask->uid()];
          dep_adj_list_[GET2D_INDEX(ntasks+1, t+1, 0)]++;
          int adj_list_index = dep_adj_list_[GET2D_INDEX(ntasks+1, t+1, 0)];
          dep_adj_list_[GET2D_INDEX(ntasks+1, t+1, adj_list_index)] = did;
          dep_adj_matrix_[GET2D_INDEX(ntasks, t+1, did+1)] = 1;
      }
      if (task->ndepends() == 0) {
          dep_adj_list_[GET2D_INDEX(ntasks+1, t+1, 0)]++;
          int adj_list_index = dep_adj_list_[GET2D_INDEX(ntasks+1, t+1, 0)];
          dep_adj_list_[GET2D_INDEX(ntasks+1, t+1, adj_list_index)] = 0;
          dep_adj_matrix_[GET2D_INDEX(ntasks, t+1, 0)] = 1;
      }
  }
}
void GraphMetadata::get_2d_comm_adj_matrix(size_t *comm_task_adj_matrix)
{
    vector<Task *> & tasks = graph_->tasks_list();
    int ntasks = tasks.size()+1;
    if (comm_task_adj_matrix == NULL) {
        comm_task_adj_matrix_ = (size_t *)calloc(ntasks*ntasks, sizeof(size_t));
        comm_task_adj_matrix = comm_task_adj_matrix_;
    }
    for(int index=0; index<tasks.size(); index++) {
        Task *each_task = tasks[index];
        vector<unsigned long> & lst1_v = task_inputs_map_[each_task->uid()];
        set<unsigned long> lst1(lst1_v.begin(), lst1_v.end());
        //printf("Task:%s:%lu ndepends:%lu lst1_v:%lu %lu\n", each_task->name(), each_task->uid(), each_task->ndepends(), lst1.size(), lst1_v.size());
        for(unsigned long mid : lst1) {
            BaseMem *mem = mem_index_hash_[mid];
            //printf("            probing mem:%lu size:%lu\n", mid, mem->size());
        }
        set<unsigned long> all_covered_mem;
        for(int di=0; di<each_task->ndepends(); di++) {
            Task *d_task = each_task->depend(di);
            unsigned long d_index = task_uid_2_index_hash_[d_task->uid()];
            vector<unsigned long> & lst2_v = task_outputs_map_[d_task->uid()];
            set<unsigned long> lst2(lst2_v.begin(), lst2_v.end());
            set<unsigned long> mem_list;
            for(unsigned long mid : lst2) {
                BaseMem *mem = mem_index_hash_[mid];
                //printf("                dependency probing mem:%lu size:%lu\n", mid, mem->size());
            }
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
                BaseMem *mem = mem_index_hash_[mid];
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
                BaseMem *mem = mem_index_hash_[mid];
                size += mem->size();
            }
        }
        if (size > 0)  {
            //_info("updating %lu-> %lu size:%lu\n", index+1, 0, size);
            comm_task_adj_matrix[GET2D_INDEX(ntasks, index+1, 0)] += size;
        }
    }
    Utils::PrintMatrixLimited<size_t>(comm_task_adj_matrix, ntasks, ntasks, "Task Communication data(C++)");
}

} /* namespace rt */
} /* namespace iris */

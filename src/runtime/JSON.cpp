#include "JSON.h"
#include "Debug.h"
#include "Command.h"
#include "Graph.h"
#include "Kernel.h"
#include "Mem.h"
#include "Platform.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <bits/stdc++.h>
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
#define RAPIDJSON_ALIGN (128)
#include <csignal>
namespace iris {
namespace rt {

rapidjson::Value iris_output_tasks_(rapidjson::kArrayType);
rapidjson::Document iris_output_document_;

JSON::JSON(Platform* platform){
  platform_ = platform;
  timer_ = new Timer();
}

JSON::~JSON() {
  delete timer_;
}

int JSON::Load(Graph* graph, const char* path, void** params) {

  //read file from path
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile((char*) path, &src, &srclen) == IRIS_ERROR) {
    _error("no JSON file[%s]", path);
    return IRIS_ERROR;
  }
  printf("RAPIDJSON source = %s\n",src);
  rapidjson::Document json_;
  json_.Parse(src);
  printf("debug! length of file = %d\n",srclen);
  if (!json_.IsObject()){
    _error("failed to create JSON from file[%s]", path);
    return IRIS_ERROR;
  }
  if(!json_.HasMember("iris-graph")){
    _error("malformed JSON in file[%s]", path);
    return IRIS_ERROR;
  }
  const rapidjson::Value& iris_graph_ = json_["iris-graph"];

  //load inputs
  if(!iris_graph_.HasMember("inputs") or !iris_graph_["inputs"].IsArray()){
    _error("malformed inputs in file[%s]", path);
    return IRIS_ERROR;
  }
  const rapidjson::Value& iris_inputs_ = iris_graph_["inputs"];
  for (rapidjson::SizeType i = 0; i < iris_inputs_.Size(); i++){
    inputs_.push_back(iris_inputs_[i].GetString());
  }

  //load tasks
  if(!iris_graph_.HasMember("graph") or !iris_graph_["graph"].HasMember("tasks") or !iris_graph_["graph"]["tasks"].IsArray()){
    _error("malformed tasks in file[%s]", path);
    return IRIS_ERROR;
  }
  const rapidjson::Value& iris_tasks_ = iris_graph_["graph"]["tasks"];
  for (rapidjson::SizeType i = 0; i < iris_tasks_.Size(); i++){
    Task* task = Task::Create(platform_, IRIS_TASK_PERM, NULL);
    tasks_.push_back(task);
    //name
    if(!iris_tasks_[i].HasMember("name") or !iris_tasks_[i]["name"].IsString()){
      _error("malformed task name in file[%s]", path);
      return IRIS_ERROR;
    }
    task->set_name(iris_tasks_[i]["name"].GetString());
    //depends
    if(!iris_tasks_[i].HasMember("depends") or !iris_tasks_[i]["depends"].IsArray()){
      _error("malformed task depends in file[%s]", path);
      return IRIS_ERROR;
    }
    for (rapidjson::SizeType j = 0; j < iris_tasks_[i]["depends"].Size(); j++){
      if(!iris_tasks_[i]["depends"][j].IsString()){
        _error("malformed task depends in file[%s]", path);
        return IRIS_ERROR;
      }
      const char* dependency_name = iris_tasks_[i]["depends"][j].GetString();
      for (const auto& prior_task: tasks_){
        if (strcmp(dependency_name,prior_task->name()) == 0){
          task->AddDepend(prior_task);
          //break;
        }
      }
    }
    //target
    if(!iris_tasks_[i].HasMember("target") or !iris_tasks_[i]["target"].IsString()){
      _error("malformed task target in file[%s]", path);
      return IRIS_ERROR;
    }
    int target = iris_default;
    const char* target_str = iris_tasks_[i]["target"].GetString();
    void* p_target = GetParameterInput(params,target_str);
    if (p_target) target = (*(int*) p_target);
    //device only policies
    else if(strcmp(target_str, "cpu") == 0) target = iris_cpu;
    else if(strcmp(target_str, "gpu") == 0) target = iris_gpu;
    //iris policies
    else if(strcmp(target_str, "all") == 0) target = iris_all;
    else if(strcmp(target_str, "any") == 0) target = iris_any;
    else if(strcmp(target_str, "data") == 0) target = iris_data;
    else if(strcmp(target_str, "default")== 0) target = iris_default;
    else if(strcmp(target_str, "depend") == 0) target = iris_depend;
    else if(strcmp(target_str, "profile") == 0) target = iris_profile;
    else if(strcmp(target_str, "random") == 0) target = iris_random;
    else if(strcmp(target_str, "roundrobin") == 0) target = iris_roundrobin;
    //the policy can also just be the actual device id!
    else if(isdigit(target_str[0])){
      target = atoi(target_str);
    }
    task->set_brs_policy(target);
    //commands (populate each task with assigned commands)
    if(!iris_tasks_[i].HasMember("commands") or !iris_tasks_[i]["commands"].IsArray()){
      _error("malformed task commands in file[%s]", path);
      return IRIS_ERROR;
    }
    for (rapidjson::SizeType j = 0; j < iris_tasks_[i]["commands"].Size(); j++){
      if (iris_tasks_[i]["commands"][j].HasMember("kernel")){
        const rapidjson::Value& kernel_ = iris_tasks_[i]["commands"][j]["kernel"];
        //name
        if(!kernel_.HasMember("name") or !kernel_["name"].IsString()){
            _error("malformed command kernel name in file[%s]", path);
            return IRIS_ERROR;
        }
        const char* name = kernel_["name"].GetString();
        Kernel* kernel = platform_->GetKernel(name);
        //global_size
        if(!kernel_.HasMember("global_size") or !kernel_["global_size"].IsArray()){
            _error("malformed command kernel global_size in file[%s]", path);
            return IRIS_ERROR;
        }
        int dim = kernel_["global_size"].Size();
        size_t* gws = (size_t*) malloc(sizeof(size_t) * dim);
        for (rapidjson::SizeType l = 0; l < kernel_["global_size"].Size(); l++){
          void* p_gws = GetParameterInput(params,kernel_["global_size"][l].GetString());
          if (p_gws){
            gws[l] = (*(size_t*) p_gws);
          } else {
            gws[l] = atol(kernel_["global_size"][l].GetString());
          }
        }
        //nkparams
        if(!kernel_.HasMember("parameters") or !kernel_["parameters"].IsArray()){
            _error("malformed command kernel params in file[%s]", path);
            return IRIS_ERROR;
        }
        int nkparams = kernel_["parameters"].Size();
        void** kparams = (void**) malloc(sizeof(void*) * nkparams);
        int* kparams_info = (int*) malloc(sizeof(int) * nkparams);
        for (rapidjson::SizeType l = 0; l < kernel_["parameters"].Size(); l++){
          const rapidjson::Value& param_ = kernel_["parameters"][l];
          if (!param_.HasMember("type")){
            _error("malformed command kernel params type in file[%s]", path);
            return IRIS_ERROR;
          }
          //parameters (scalar)
          if (strcmp("scalar",param_["type"].GetString()) == 0){
            //name
            if (!param_.HasMember("name")){
              _error("malformed command kernel parameters scalar name in file[%s]", path);
              return IRIS_ERROR;
            }
            kparams[l] = GetParameterInput(params,param_["name"].GetString());
            //data_type
            //value   
            printf("parameter no:%i is scalar\n",l);
            raise(SIGINT);
          }
          //parameters (memory_object)
          else if (strcmp("memory_object",param_["type"].GetString()) == 0){
            //name
            if (!param_.HasMember("name")){
              _error("malformed command kernel parameters memory name in file[%s]", path);
              return IRIS_ERROR;
            }
            kparams[l] = GetParameterInput(params,param_["name"].GetString());
            //permissions---for the memory object
            if (!param_.HasMember("permission")){
              _error("malformed command kernel parameters memory permission in file[%s]", path);
              return IRIS_ERROR;
            }
            const char* permission = param_["permission"].GetString();
            if (strcmp("rw", permission) == 0) kparams_info[l] = iris_rw;
            else if (strcmp("r", permission) == 0) kparams_info[l] = iris_r;
            else if (strcmp("w", permission) == 0) kparams_info[l] = iris_w;
            else {
              _error("malformed command kernel parameters memory permission in file[%s]", path);
              return IRIS_ERROR;
            }
            //size_bytes (TODO: optional? or even valid?)
            //value (TODO: optional? or even valid?)
          }
          else{
            _error("malformed command kernel params in file[%s]", path);
            return IRIS_ERROR;
          }
        }
        //local_size (optional)
        size_t* lws = NULL;
        if(kernel_.HasMember("local_size")) {
          if(!kernel_["local_size"].IsArray()){
            _error("malformed command kernel local_size in file[%s]", path);
            return IRIS_ERROR;
          }
          dim = kernel_["local_size"].Size();
          lws = (size_t*) malloc(sizeof(size_t) * dim);
          for (rapidjson::SizeType l = 0; l < kernel_["local_size"].Size(); l++){
            void* p_lws = GetParameterInput(params,kernel_["local_size"][l].GetString());
            if (p_lws){
              lws[l] = (*(size_t*) p_lws);
            } else {
              lws[l] = atol(kernel_["local_size"][l].GetString());
            }
          }
        }
        //offset (optional)
        size_t* offset = NULL;
        if(kernel_.HasMember("offset")) {
          if(!kernel_["offset"].IsArray()){
            _error("malformed command kernel offset in file[%s]", path);
            return IRIS_ERROR;
          }
          dim = kernel_["offset"].Size();
          offset = (size_t*) malloc(sizeof(size_t) * dim);
          for (rapidjson::SizeType l = 0; l < kernel_["offset"].Size(); l++){
            void* p_off = GetParameterInput(params,kernel_["offset"][l].GetString());
            if (p_off){
              offset[l] = (*(size_t*) p_off);
            } else {
              offset[l] = atol(kernel_["offset"][l].GetString());
            }
          }
        }
        //TODO: currently no JSON support for mem_ranges or params_offset (added to issue tracker on: https://code.ornl.gov/brisbane/iris/-/issues/14)
        //**NOTE**: check dimensions match between validation and kernel
        size_t* params_offset = NULL;
        size_t* memory_ranges = NULL;
        Command* cmd = Command::CreateKernel(task, kernel, dim, offset, gws, lws, nkparams, kparams, params_offset, kparams_info, memory_ranges);
        task->AddCommand(cmd);
      }
      else if (iris_tasks_[i]["commands"][j].HasMember("h2d")){
        const rapidjson::Value& h2d_ = iris_tasks_[i]["commands"][j]["h2d"];
        //host_memory
        if(!h2d_.HasMember("host_memory") or !h2d_["host_memory"].IsString()){
          _error("malformed command h2d host_memory in file[%s]", path);
          return IRIS_ERROR;
        }
        void* host_mem = GetParameterInput(params, h2d_["host_memory"].GetString());
        //device_memory
        if(!h2d_.HasMember("device_memory") or !h2d_["device_memory"].IsString()){
          _error("malformed command h2d device_memory in file[%s]", path);
          return IRIS_ERROR;
        }
        iris_mem dev_mem = (iris_mem) GetParameterInput(params, h2d_["device_memory"].GetString());
        //offset
        if(!h2d_.HasMember("offset") or !h2d_["offset"].IsString()){
          _error("malformed command h2d offset in file[%s]", path);
          return IRIS_ERROR;
        }
        void* p_off = GetParameterInput(params, h2d_["offset"].GetString());
        size_t offset = p_off ? (*(size_t*) p_off) : atol(h2d_["offset"].GetString());
        //size
        if(!h2d_.HasMember("size") or !h2d_["size"].IsString()){
          _error("malformed command h2d size in file[%s]", path);
          return IRIS_ERROR;
        }
        void* p_size = GetParameterInput(params, h2d_["size"].GetString());
        size_t size = p_size ? (*(size_t*) p_size) : atol(h2d_["size"].GetString());       

        Command* cmd = Command::CreateH2D(task, dev_mem->class_obj, offset, size, host_mem);
        //name (optional)
        if(h2d_.HasMember("name")){
          if(!h2d_["name"].IsString()){
            _error("malformed command h2d name in file[%s]", path);
            return IRIS_ERROR;
          }
          cmd->set_name(const_cast<char*>(h2d_["name"].GetString()));
        }
        task->AddCommand(cmd);
      }
      else if (iris_tasks_[i]["commands"][j].HasMember("d2h")){
        const rapidjson::Value& d2h_ = iris_tasks_[i]["commands"][j]["d2h"];
        //host_memory
        if(!d2h_.HasMember("host_memory") or !d2h_["host_memory"].IsString()){
          _error("malformed command d2h host_memory in file[%s]", path);
          return IRIS_ERROR;
        }
        void* host_mem = GetParameterInput(params, d2h_["host_memory"].GetString());
        //device_memory
        if(!d2h_.HasMember("device_memory") or !d2h_["device_memory"].IsString()){
          _error("malformed command d2h device_memory in file[%s]", path);
          return IRIS_ERROR;
        }
        iris_mem dev_mem = (iris_mem) GetParameterInput(params, d2h_["device_memory"].GetString());
        //offset
        if(!d2h_.HasMember("offset") or !d2h_["offset"].IsString()){
          _error("malformed command d2h offset in file[%s]", path);
          return IRIS_ERROR;
        }
        void* p_off = GetParameterInput(params, d2h_["offset"].GetString());
        size_t offset = p_off ? (*(size_t*) p_off) : atol(d2h_["offset"].GetString());
        //size
        if(!d2h_.HasMember("size") or !d2h_["size"].IsString()){
          _error("malformed command d2h size in file[%s]", path);
          return IRIS_ERROR;
        }
        void* p_size = GetParameterInput(params, d2h_["size"].GetString());
        size_t size = p_size ? (*(size_t*) p_size) : atol(d2h_["size"].GetString());

        Command* cmd = Command::CreateD2H(task, dev_mem->class_obj, offset, size, host_mem);
        //name (optional)
        if(d2h_.HasMember("name")){
          if(!d2h_["name"].IsString()){
            _error("malformed command d2h name in file[%s]", path);
            return IRIS_ERROR;
          }
          cmd->set_name(const_cast<char*>(d2h_["name"].GetString()));
        }
        task->AddCommand(cmd);
      }
      else {
        _error("malformed command in file[%s]", path);
        return IRIS_ERROR;
      }
    }
    graph->AddTask(task);
  }
  return IRIS_SUCCESS;
}

void* JSON::GetParameterInput(void** params, const char* buf){
  for (int j = 0; j < inputs_.size(); j ++){
    if (strcmp(inputs_[j],buf) == 0){
      return(params[j]);
    }
  }
  return(NULL);
}

int JSON::RecordFlush() {
  rapidjson::Value json_d(rapidjson::kObjectType);
  rapidjson::Value iris_graph(rapidjson::kObjectType);
  //inputs
  rapidjson::Value inputs_(rapidjson::kArrayType);
  //host pointers
  rapidjson::Value tmp;
  for(std::set<void*>::iterator iter = host_ptrs_.begin(); iter != host_ptrs_.end(); iter ++){
    tmp.SetString(NameFromHostPointer(*iter).c_str(),NameFromHostPointer(*iter).length(),iris_output_document_.GetAllocator());
    inputs_.PushBack(tmp,iris_output_document_.GetAllocator());
  }
  //device memory
  for(std::set<Mem*>::iterator iter = mems_.begin(); iter != mems_.end(); iter ++){
    tmp.SetString(NameFromDeviceMem(*iter).c_str(),NameFromDeviceMem(*iter).length(),iris_output_document_.GetAllocator());
    inputs_.PushBack(tmp,iris_output_document_.GetAllocator());
  }
  iris_graph.AddMember("inputs",inputs_,iris_output_document_.GetAllocator());
  rapidjson::Value tasks_(rapidjson::kObjectType);
  tasks_.AddMember("tasks",iris_output_tasks_,iris_output_document_.GetAllocator());
  iris_graph.AddMember("graph",tasks_,iris_output_document_.GetAllocator());
  iris_output_document_.SetObject();
  iris_output_document_.AddMember("iris-graph",iris_graph,iris_output_document_.GetAllocator());
  struct Stream {
    std::ofstream of {"output.json"};
    typedef char Ch;
    void Put (Ch ch) {of.put (ch);}
    void Flush() {}
  } stream;
  rapidjson::Writer<Stream> writer (stream);
  iris_output_document_.Accept (writer);
  return IRIS_SUCCESS;
}

const std::string JSON::NameFromHostPointer(void* host_ptr){
  return std::string("hostmem-"+std::to_string(UniqueUIDFromHostPointer(host_ptr)));
}

int JSON::UniqueUIDFromHostPointer(void* host_ptr){
  host_ptrs_.insert(host_ptr);

  std::set<void*>::iterator it = host_ptrs_.begin();
  for (int i = 0; i < host_ptrs_.size(); i++) {
     if(*it == host_ptr){
       return(i);
     }
  }
}

const std::string JSON::NameFromDeviceMem(Mem* dev_mem){
  mems_.insert(dev_mem);
  return std::string("devicemem-"+std::to_string(dev_mem->uid()));
}

int JSON::RecordTask(Task* task) {
  if(task->ncmds() == 0) return IRIS_SUCCESS;//skip recording IRIS_MARKER tasks
  rapidjson::Value _task(rapidjson::kObjectType);
  //name
  rapidjson::Value _name(rapidjson::StringRef(task->name()));
  if(_name == ""){
    char buffer[64];
    int len = sprintf(buffer, "task-%i", task->uid()); // dynamically created string.
    _name.SetString(buffer, len, iris_output_document_.GetAllocator());
  }
  _task.AddMember("name",_name,iris_output_document_.GetAllocator());
  if (task->ncmds() == 0) {
    //it's and empty (likely checkpointing) task, so don't record it.
    raise(SIGINT);
    //TODO: discard empty tasks (tasks without any commands)
    return IRIS_SUCCESS;
  }
  //command(s)
  rapidjson::Value _cmds(rapidjson::kArrayType);
  rapidjson::Value tmp;//for calls to NameFromHostPointer and NameFromDeviceMem--where a simple StringRef goes out of scope and returns garbage collected junk.
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (cmd->type() == IRIS_CMD_H2D) {
      rapidjson::Value h2d_(rapidjson::kObjectType);
      if (cmd->name()) h2d_.AddMember("name",rapidjson::StringRef(cmd->name()),iris_output_document_.GetAllocator());
      //host_memory
      tmp.SetString(NameFromHostPointer(cmd->host()).c_str(),NameFromHostPointer(cmd->host()).length(),iris_output_document_.GetAllocator());
      h2d_.AddMember("host_memory",tmp,iris_output_document_.GetAllocator());
      //device_memory
      tmp.SetString(NameFromDeviceMem(cmd->mem()).c_str(),iris_output_document_.GetAllocator());
      h2d_.AddMember("device_memory",tmp,iris_output_document_.GetAllocator());
      h2d_.AddMember("offset",rapidjson::Value(cmd->off(0)),iris_output_document_.GetAllocator());
      h2d_.AddMember("size",rapidjson::Value(cmd->size()),iris_output_document_.GetAllocator());
      rapidjson::Value cmd_(rapidjson::kObjectType);
      cmd_.AddMember("h2d",h2d_,iris_output_document_.GetAllocator());
      _cmds.PushBack(cmd_,iris_output_document_.GetAllocator());
      printf("recorded h2d\n");
    }
    else if (cmd->type() == IRIS_CMD_D2H) {
      rapidjson::Value d2h_(rapidjson::kObjectType);
      if (cmd->name()) d2h_.AddMember("name",rapidjson::StringRef(cmd->name()),iris_output_document_.GetAllocator());
      //host_memory
      tmp.SetString(NameFromHostPointer(cmd->host()).c_str(),iris_output_document_.GetAllocator());
      d2h_.AddMember("host_memory",tmp,iris_output_document_.GetAllocator());
      //device_memory
      tmp.SetString(NameFromDeviceMem(cmd->mem()).c_str(),iris_output_document_.GetAllocator());
      d2h_.AddMember("device_memory",tmp,iris_output_document_.GetAllocator());
      d2h_.AddMember("offset",rapidjson::Value(cmd->off(0)),iris_output_document_.GetAllocator());
      d2h_.AddMember("size",rapidjson::Value(cmd->size()),iris_output_document_.GetAllocator());
      rapidjson::Value cmd_(rapidjson::kObjectType);
      cmd_.AddMember("d2h",d2h_,iris_output_document_.GetAllocator());
      _cmds.PushBack(cmd_,iris_output_document_.GetAllocator());
      printf("recorded d2h\n");    }
    else if (cmd->type() == IRIS_CMD_KERNEL) {
      rapidjson::Value kernel_(rapidjson::kObjectType);
      kernel_.AddMember("name",rapidjson::StringRef(cmd->name()),iris_output_document_.GetAllocator());
      //global_size
      rapidjson::Value gs_(rapidjson::kArrayType);
      for(int i = 0; i < cmd->dim(); i ++)
        gs_.PushBack(rapidjson::Value(cmd->gws(i)),iris_output_document_.GetAllocator());
      kernel_.AddMember("global_size",gs_,iris_output_document_.GetAllocator());
      //local_size
      rapidjson::Value ls_(rapidjson::kArrayType);
      for(int i = 0; i < cmd->dim(); i ++)
        ls_.PushBack(rapidjson::Value(cmd->lws(i)),iris_output_document_.GetAllocator());
      kernel_.AddMember("local_size",ls_,iris_output_document_.GetAllocator());
      //offset
      rapidjson::Value off_(rapidjson::kArrayType);
      for(int i = 0; i < cmd->dim(); i ++)
        off_.PushBack(rapidjson::Value(cmd->off(i)),iris_output_document_.GetAllocator());
      kernel_.AddMember("offset",off_,iris_output_document_.GetAllocator());
      //parameters
      rapidjson::Value params_(rapidjson::kArrayType);
      for(int i = 0; i < cmd->kernel_nargs(); i ++) {
        rapidjson::Value param_(rapidjson::kObjectType);
        KernelArg* arg = cmd->kernel_arg(i);
        if(arg->mem){//memory
          param_.AddMember("type",rapidjson::StringRef("memory_object"),iris_output_document_.GetAllocator());
          tmp.SetString(NameFromDeviceMem(arg->mem).c_str(),iris_output_document_.GetAllocator());
          param_.AddMember("value",tmp,iris_output_document_.GetAllocator());
          param_.AddMember("size_bytes",rapidjson::Value(arg->size),iris_output_document_.GetAllocator());
          //permissions
          if (arg->mode == iris_r) param_.AddMember("permissions",rapidjson::StringRef("r"),iris_output_document_.GetAllocator());
          else if (arg->mode == iris_w) param_.AddMember("permissions",rapidjson::StringRef("w"),iris_output_document_.GetAllocator());
          else if (arg->mode == iris_rw) param_.AddMember("permissions",rapidjson::StringRef("rw"),iris_output_document_.GetAllocator());
          else _error("not valid mode[%d]", arg->mode);
          //param_.AddMember("name",,iris_output_document_.GetAllocator());//we said this was optional. But currently the Mem.h class doesn't have a name member function. 
        }else{//scalar
          param_.AddMember("type",rapidjson::StringRef("scalar"),iris_output_document_.GetAllocator());
          _error("scalar support is currently unimplemented! @JSON.cpp : line %d", __LINE__);
          //param_.AddMember("name",,iris_output_document_.GetAllocator());//again still optional
          //param_.AddMember("data_type",,iris_output_document_.GetAllocator());
          //param_.AddMember("value",,iris_output_document_.GetAllocator());
        }
        params_.PushBack(param_,iris_output_document_.GetAllocator());
      }
      kernel_.AddMember("parameters",params_,iris_output_document_.GetAllocator());
      rapidjson::Value cmd_(rapidjson::kObjectType);
      cmd_.AddMember("kernel",kernel_,iris_output_document_.GetAllocator());
      _cmds.PushBack(cmd_,iris_output_document_.GetAllocator());
      printf("recorded kernel\n");
    }
  }
  _task.AddMember("commands",_cmds,iris_output_document_.GetAllocator());
  //target
  rapidjson::Value _target;
  if (task->dev()) {
    _target.SetString(rapidjson::StringRef(std::to_string(task->devno()).c_str()),iris_output_document_.GetAllocator());
    //TODO: handle passing device ids as the target directly (prioritizing it over a policy)
    raise(SIGINT);
  }
  else {
    char* tmp = task->brs_policy_string();
    if(strcmp(tmp, "unknown") == 0) raise(SIGINT);//debugging, we shouldn't ever hit this! TODO: delete this!
    _target.SetString(rapidjson::StringRef(task->brs_policy_string()), iris_output_document_.GetAllocator());
  }
  _task.AddMember("target",_target,iris_output_document_.GetAllocator());
  //depends
  rapidjson::Value _depends(rapidjson::kArrayType);
  for (int i = 0; i < task->ndepends(); i++) {
    rapidjson::Value depend(rapidjson::StringRef(task->depend(i)->name()));
    _depends.PushBack(depend, iris_output_document_.GetAllocator());
  }
  _task.AddMember("depends",_depends,iris_output_document_.GetAllocator());
  iris_output_tasks_.PushBack(_task, iris_output_document_.GetAllocator());
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

//Made in Cortona, Italy.

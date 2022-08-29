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
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#include <csignal>
namespace iris {
namespace rt {

rapidjson::Value iris_output_tasks_;
rapidjson::Document iris_output_document_;

JSON::JSON(Platform* platform){
  platform_ = platform;
  timer_ = new Timer();
  iris_output_tasks_.SetObject();
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
  rapidjson::Document json_;
  json_.Parse(src);
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
    else if(strcmp(target_str, "cpu") == 0) target = iris_cpu;
    else if(strcmp(target_str, "gpu") == 0) target = iris_gpu;
    else if(isdigit(target_str[0])){//the policy can also just be the actual device id!
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
  json_d.AddMember("iris-graph", iris_graph, iris_output_document_.GetAllocator());
  raise(SIGINT);
  //write inputs
  /*
  rapidjson::Value inputs(kArrayType);
  for (std::set<Mem*>::iterator I = mems_.begin(); I != mems_.end(); ++I) {
    sprintf(buf, "\"mem-%lu\", ", (*I)->uid());
    write(fd, buf, strlen(buf));
    memset(buf, 0, 128);
  }
  iris_graph.AddMember("inputs", inputs, iris_output_document.GetAllocator());

  for (int i = 0; i < nptrs_; i++) {
    if (nptrs_ > 1 && i < nptrs_ - 1) sprintf(buf, "\"user-%d\", ", i);
    else if (i == nptrs_ - 1) sprintf(buf, "\"user-%d\"", i);
    write(fd, buf, strlen(buf));
    memset(buf, 0, 128);
  }
  */

          // ...
  //  
  //  FILE* fp = fopen("output.json", "wb"); // non-Windows use "w"
  //   
  //   char writeBuffer[65536];
  //   FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
  //    
  //    Writer<FileWriteStream> writer(os);
  //    d.Accept(writer);
  //     
  //     fclose(fp);
  return IRIS_SUCCESS;
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
  //command(s)
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (cmd->type() == IRIS_CMD_H2D) {
      //RecordH2D(cmd, buf);
      printf("record h2d\n");
      //mems_.insert(cmd->mem());
      //sprintf(buf, "  \"h2d\": [\"mem-%lu\", \"user-%d\", \"%zu\", \"%zu\"],\n", cmd->mem()->uid(), InputPointer(cmd->host()), cmd->off(0), cmd->size());
    }
    else if (cmd->type() == IRIS_CMD_D2H) {
      printf("record d2h\n");
      //RecordD2H(cmd, buf);
      //mems_.insert(cmd->mem());
      //sprintf(buf, "  \"d2h\": [\"mem-%lu\", \"user-%d\", \"%zu\", \"%zu\"],\n", cmd->mem()->uid(), InputPointer(cmd->host()), cmd->off(0), cmd->size());
    }
    else if (cmd->type() == IRIS_CMD_KERNEL) {
      printf("record kernel\n");
      //RecordKernel(cmd, buf);
      /*
      std::string str;
      char c[256];
      sprintf(c, "  \"kernel\": [\"%s\", \"%zu\", [", cmd->kernel()->name(), cmd->gws(0));
      str.append(c);
      memset(c, 0, 256);
      int nargs = cmd->kernel_nargs();
      for (int i = 0; i < nargs; i++) {
        KernelArg* arg = cmd->kernel_arg(i);
        if (arg->mem) {
          mems_.insert(arg->mem);
          str.append("\"mem-");
          str.append(std::to_string(arg->mem->uid()));
          str.append("\"");
        } else {
        }
        if (nargs > 1 && i < nargs - 1) str.append(", ");
      }
      str.append("], [");
      for (int i = 0; i < nargs; i++) {
        KernelArg* arg = cmd->kernel_arg(i);
        if (arg->mem) {
          if (arg->mode == iris_r) str.append("\"r\"");
          else if (arg->mode == iris_w) str.append("\"w\"");
          else if (arg->mode == iris_rw) str.append("\"rw\"");
          else _error("not valid mode[%d]", arg->mode);
        } else {
        }
        if (nargs > 1 && i < nargs - 1) str.append(", ");
      }
      str.append("] ],\n");
      sprintf(buf, "%s", str.c_str());
      */
      }
  }
  //target
  rapidjson::Value _target;
  if (task->dev()) {
    //TODO: handle passing device ids as the target directly (prioritizing it over a policy)
    raise(SIGINT);
  }
  else {
    char buffer[64];
    //int len = sprintf(buffer, "0x%x", task->brs_policy());
    int len = sprintf(buffer, "%i", task->brs_policy());
    _target.SetString(rapidjson::StringRef(buffer), len, iris_output_document_.GetAllocator());
  }
  _task.AddMember("target",_target,iris_output_document_.GetAllocator());
  //depends
  rapidjson::Value _depends(rapidjson::kArrayType);
  for (int i = 0; i < task->ndepends(); i++) {
    rapidjson::Value depend(task->depend(i)->uid());
    _depends.PushBack(depend, iris_output_document_.GetAllocator());
  }
  _task.AddMember("depends",_depends,iris_output_document_.GetAllocator());
  //
  //auto const& tmp = _task["name"].GetString();
  //auto const& tmp = _task["target"].GetString();
  for (rapidjson::SizeType i = 0; i < task->ndepends(); i++){
    auto const& tmp = _task["depends"][i].GetString();
    raise(SIGINT);
  }
  //TODO: discard empty tasks (tasks without any commands)
  iris_output_tasks_.PushBack(_task, iris_output_document_.GetAllocator());
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

//Made in Cortona, Italy.

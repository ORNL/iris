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
#include "rapidjson/error/en.h"

#ifdef RAPIDJSON_ALIGN
#undef RAPIDJSON_ALIGN
#endif //RAPIDJSON_ALIGN
#define RAPIDJSON_ALIGN (128)

namespace iris {
namespace rt {

rapidjson::Value iris_output_tasks_(rapidjson::kArrayType);
rapidjson::Document iris_output_document_;

JSON::JSON(Platform* platform){
  platform_ = platform;
}

JSON::~JSON() {
}

int JSON::Load(Graph* graph, const char* path, void** params) {
  //read file from path
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile((char*) path, &src, &srclen) == IRIS_ERROR) {
    _error("no JSON file[%s]", path);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  rapidjson::Document json_;
  json_.Parse(src);
  if (!json_.IsObject()){
    _error("failed to create JSON from file[%s]", path);
    platform_->IncrementErrorCount();
    if (json_.HasParseError()) {
      fprintf(stderr, "\nError(offset %u): %s\n",
          (unsigned)json_.GetErrorOffset(),
          GetParseError_En(json_.GetParseError()));
    }
    return IRIS_ERROR;
  }
  if(!json_.HasMember("iris-graph")){
    _error("malformed JSON in file[%s]", path);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  const rapidjson::Value& iris_graph_ = json_["iris-graph"];

  //load inputs
  if(!iris_graph_.HasMember("inputs") or !iris_graph_["inputs"].IsArray()){
    _error("malformed inputs in file[%s]", path);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  const rapidjson::Value& iris_inputs_ = iris_graph_["inputs"];
  for (rapidjson::SizeType i = 0; i < iris_inputs_.Size(); i++){
    inputs_.push_back(iris_inputs_[i].GetString());
  }
  //load tasks
  if(!iris_graph_.HasMember("graph") or !iris_graph_["graph"].HasMember("tasks") or !iris_graph_["graph"]["tasks"].IsArray()){
    _error("malformed tasks in file[%s]", path);
    platform_->IncrementErrorCount();
    return IRIS_ERROR;
  }
  const rapidjson::Value& iris_tasks_ = iris_graph_["graph"]["tasks"];
  for (rapidjson::SizeType i = 0; i < iris_tasks_.Size(); i++){
    int ncmds = 0;
    if (iris_tasks_[i].HasMember("commands"))
        ncmds = (int)iris_tasks_[i]["commands"].Size();
    Task* task = Task::Create(platform_, IRIS_TASK_PERM, NULL, ncmds);
    tasks_.push_back(task);
    //name
    if(!iris_tasks_[i].HasMember("name") or !iris_tasks_[i]["name"].IsString()){
      _error("malformed task name in file[%s]", path);
      platform_->IncrementErrorCount();
      return IRIS_ERROR;
    }
    task->set_name(iris_tasks_[i]["name"].GetString());
    //depends
    if(!iris_tasks_[i].HasMember("depends") or !iris_tasks_[i]["depends"].IsArray()){
      _error("malformed task [%s] missing dependency in file[%s]", task->name(), path);
      platform_->IncrementErrorCount();
      return IRIS_ERROR;
    }
    for (rapidjson::SizeType j = 0; j < iris_tasks_[i]["depends"].Size(); j++){
      //auto tmp = iris_tasks_[i]["depends"][j].GetType();
      if(iris_tasks_[i]["depends"][j].IsNull()){
        continue;
      }
      if(!iris_tasks_[i]["depends"][j].IsString()){
        _error("malformed task depends in file[%s]", path);
        platform_->IncrementErrorCount();
        return IRIS_ERROR;
      }
      const char* dependency_name = iris_tasks_[i]["depends"][j].GetString();
      for (const auto& prior_task: tasks_){
        if (strcmp(dependency_name,prior_task->name()) == 0){
          task->AddDepend(prior_task, prior_task->uid());
          //break;
        }
      }
    }
    //target
    if(!iris_tasks_[i].HasMember("target")){
      _error("malformed task target in file[%s]", path);
      platform_->IncrementErrorCount();
      return IRIS_ERROR;
    }
    int target = iris_default;
    if(iris_tasks_[i]["target"].IsString()){
      //the scheduled target can be passed in as an input parameter,
      const char* target_str = iris_tasks_[i]["target"].GetString();
      void* p_target = GetParameterInput(params,target_str);
      if (p_target) target = (*(int*) p_target);
      //or a string
      //device only policies
      else if(strcmp(target_str, "cpu") == 0) target = iris_cpu;
      else if(strcmp(target_str, "gpu") == 0) target = iris_gpu;
      //iris policies
      else if(strcmp(target_str, "ftf") == 0) target = iris_ftf;
      else if(strcmp(target_str, "sdq") == 0) target = iris_sdq;
      else if(strcmp(target_str, "data") == 0) target = iris_data;
      else if(strcmp(target_str, "default")== 0) target = iris_default;
      else if(strcmp(target_str, "depend") == 0) target = iris_depend;
      else if(strcmp(target_str, "profile") == 0) target = iris_profile;
      else if(strcmp(target_str, "random") == 0) target = iris_random;
      else if(strcmp(target_str, "roundrobin") == 0) target = iris_roundrobin;
      //if custom is mentioned in the IRIS policy --- its a custom policy but we need to register it
      else if(strcmp(target_str, "custom") == 0) {
        target = iris_custom;
      }
      //the policy can also just be the actual device id!
      else if(isdigit(target_str[0])){
        target = atoi(target_str);
      }
    } else if (iris_tasks_[i]["target"].IsInt()) {
      target = iris_tasks_[i]["target"].GetInt();
    } else {
      _error("malformed task target (not a string or int) in file[%s]", path);
      platform_->IncrementErrorCount();
      return IRIS_ERROR;
    }

    //if we have a custom policy we have one final string passed: the name of the previously registered custom policy
    if (target == iris_custom){
      //get the last input which is the custom_policy_name
      //not strictly passed in the JSON we have to evaluate the last parameter explicitly.
      //This is because the custom policies target string isn't **strictly** part of the schema
      //this means we can change the target policy as a runtime argument.
      const char* custom_policy_name = (char*) params[iris_inputs_.Size()];//it will always be the size of the number of expected input arguments (plus one).
      task->set_opt(custom_policy_name);
    }
    task->set_brs_policy(target);
    //commands (populate each task with assigned commands)
    if(!iris_tasks_[i].HasMember("commands") or !iris_tasks_[i]["commands"].IsArray()){
      _error("malformed task, missing (or not an array) commands of task[%s] in file[%s]", task->name(), path);
      platform_->IncrementErrorCount();
      return IRIS_ERROR;
    }
    for (rapidjson::SizeType j = 0; j < iris_tasks_[i]["commands"].Size(); j++){
      if (iris_tasks_[i]["commands"][j].HasMember("kernel")){
        const rapidjson::Value& kernel_ = iris_tasks_[i]["commands"][j]["kernel"];
        //name
        if(!kernel_.HasMember("name") or !kernel_["name"].IsString()){
            _error("malformed command kernel name in file[%s]", path);
             platform_->IncrementErrorCount();
             return IRIS_ERROR;
        }
        const char* name = kernel_["name"].GetString();
        Kernel* kernel = platform_->GetKernel(name);
        //global_size
        if(!kernel_.HasMember("global_size") or (!kernel_["global_size"].IsArray() and !kernel_["global_size"].IsInt())){
            _error("malformed command kernel global_size in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
        }
        int dim = kernel_["global_size"].Size();
        size_t* gws = (size_t*) malloc(sizeof(size_t) * dim);
        for (rapidjson::SizeType l = 0; l < kernel_["global_size"].Size(); l++){
          if(kernel_["global_size"][l].IsInt()){
              gws[l] = static_cast<size_t>(kernel_["global_size"][l].GetInt());
          } else {
            void* p_gws = GetParameterInput(params,kernel_["global_size"][l].GetString());
            if (p_gws){
              gws[l] = (*(size_t*) p_gws);
            } else {
              gws[l] = atol(kernel_["global_size"][l].GetString());
            }
          }
        }
        //nkparams
        if(!kernel_.HasMember("parameters") or !kernel_["parameters"].IsArray()){
            _error("malformed command kernel params in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
        }
        int nkparams = kernel_["parameters"].Size();
        void** kparams = (void**) malloc(sizeof(void*) * nkparams);
        int* kparams_info = (int*) malloc(sizeof(int) * nkparams);
        for (rapidjson::SizeType l = 0; l < kernel_["parameters"].Size(); l++){
          const rapidjson::Value& param_ = kernel_["parameters"][l];
          if (!param_.HasMember("type")){
            _error("malformed command kernel params type in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
          }
          //parameters (scalar)
          if (strcmp("scalar",param_["type"].GetString()) == 0){
            //name
            if (!param_.HasMember("value")){
              _error("malformed command kernel parameters scalar name in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
            kparams[l] = GetParameterInput(params,param_["value"].GetString());
            //if it isn't a string to look up or be resolved as an input, it could just be a value (garbled from the recording process) Don't panic! Let the kernel typecasting do the work!
            if (!kparams[l]) kparams[l] = (void*)param_["value"].GetString();
            //Note: the type of the scalar argument can be set 3 different ways--we can statically set the data_type (the data type used, expressed as a string), or data_size (integer denoting the number of bytes), or dynamically as an input argument (again as an integer to denote the number of bytes).
            if (param_.HasMember("data_type") && param_["data_type"].IsString()) {
              const char* data_type_str = param_["data_type"].GetString();
              if (strcmp(data_type_str,"int") == 0) kparams_info[l] = sizeof(int);
              else if(strcmp(data_type_str,"short") == 0) kparams_info[l] = sizeof(short);
              else if(strcmp(data_type_str,"long") == 0) kparams_info[l] = sizeof(long);
              else if(strcmp(data_type_str,"char") == 0) kparams_info[l] = sizeof(char);
              else if(strcmp(data_type_str,"float") == 0) kparams_info[l] = sizeof(float);
              else if(strcmp(data_type_str,"double") == 0) kparams_info[l] = sizeof(double);
              else if(strcmp(data_type_str,"long double") == 0) kparams_info[l] = sizeof(long double);
              else {
                _error("malformed command kernel parameters scalar data_type in file[%s] -- we must know the number of bytes this data type uses!", path);
                platform_->IncrementErrorCount();
                return IRIS_ERROR;
              }
            } else if(param_.HasMember("data_size") && param_["data_size"].IsInt()){
              kparams_info[l] = param_["data_size"].GetInt();
            } else if(param_.HasMember("data_size") && param_["data_size"].IsString()){
              //the dynamic option
              kparams_info[l] = *(int*)GetParameterInput(params,param_["data_size"].GetString());
              if(!kparams_info[l] && kparams_info[l] != 0){
                _error("malformed command kernel parameters scalar data_size (dynamic) in file[%s]. Can be data_size (int, in bytes) when provided this way must be an **input** argument", path);
                platform_->IncrementErrorCount();
                return IRIS_ERROR;
              }
            } else {
              _error("malformed command kernel parameters scalar data_type in file[%s]. Can be data_size (int, in bytes), or data_type (as a string)", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
          }
          //parameters (memory_object)
          else if (strcmp("memory_object",param_["type"].GetString()) == 0){
            //value
            if (!param_.HasMember("value")){
              _error("malformed command kernel parameters memory value in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
            kparams[l] = GetParameterInput(params,param_["value"].GetString());
            //permissions---for the memory object
            if (!param_.HasMember("permissions")){
              _error("malformed command kernel parameters memory permission in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
            const char* permission = param_["permissions"].GetString();
            if (strcmp("rw", permission) == 0) kparams_info[l] = iris_rw;
            else if (strcmp("r", permission) == 0) kparams_info[l] = iris_r;
            else if (strcmp("w", permission) == 0) kparams_info[l] = iris_w;
            else {
              _error("malformed command kernel parameters memory permissions in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
            //name (optional)
            if (param_.HasMember("name")){
               kparams[l] = GetParameterInput(params,param_["name"].GetString());
            }
            ////TODO: implement loading size_bytes
            ////size_bytes
            //if (param_.HasMember("size_bytes")){
            //  if (param_["size_bytes"].IsInt()){
            //    kparams[l] = const_cast<char*>(std::to_string(param_["size_bytes"].GetInt()).data());
            //  } else {
            //    _error("unsupported command kernel parameters memory size_bytes in file[%s]", path);
            //    platform_->IncrementErrorCount();
            //    return IRIS_ERROR;
            //  }
            //}
          }
          else{
            _error("malformed command kernel params in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
          }
        }
        //local_size (optional)
        size_t* lws = NULL;
        if(kernel_.HasMember("local_size")) {
          if(!kernel_["local_size"].IsArray()){
            _error("malformed command kernel local_size in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
          }
          dim = kernel_["local_size"].Size();
          lws = (size_t*) malloc(sizeof(size_t) * dim);
          for (rapidjson::SizeType l = 0; l < kernel_["local_size"].Size(); l++){
            if(kernel_["local_size"][l].IsString()){
              void* p_lws = GetParameterInput(params,kernel_["local_size"][l].GetString());
              if (p_lws){
                lws[l] = (*(size_t*) p_lws);
              } else {
                lws[l] = atol(kernel_["local_size"][l].GetString());
              }
            }
            else if(kernel_["local_size"][l].IsInt()) {
              lws[l] = static_cast<size_t>(kernel_["local_size"][l].GetInt());
            }
            else {
              _error("malformed command kernel local_size in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
          }
        }
        //offset (optional)
        size_t* offset = NULL;
        if(kernel_.HasMember("offset")) {
          if(!kernel_["offset"].IsArray()){
            _error("malformed command kernel offset in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
          }
          dim = kernel_["offset"].Size();
          offset = (size_t*) malloc(sizeof(size_t) * dim);
          for (rapidjson::SizeType l = 0; l < kernel_["offset"].Size(); l++){
            if (kernel_["offset"][l].IsString()){
              void* p_off = GetParameterInput(params,kernel_["offset"][l].GetString());
              if (p_off){
                offset[l] = (*(size_t*) p_off);
              } else {
                offset[l] = atol(kernel_["offset"][l].GetString());
              }
            } else if (kernel_["offset"][l].IsInt()){
              offset[l] = static_cast<size_t>(kernel_["offset"][l].GetInt());
            }
            else {
              _error("malformed command kernel offset in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
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
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        void* host_mem = GetParameterInput(params, h2d_["host_memory"].GetString());
        //device_memory
        if(!h2d_.HasMember("device_memory") or !h2d_["device_memory"].IsString()){
          _error("malformed command h2d device_memory in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        iris_mem *dev_mem = (iris_mem *) GetParameterInput(params, h2d_["device_memory"].GetString());
        //offset
        if(!h2d_.HasMember("offset")){
          _error("missing command h2d offset in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        size_t offset;
        if(h2d_["offset"].IsInt()){
          offset = static_cast<size_t>(h2d_["offset"].GetInt());
        } else if(h2d_["offset"].IsArray()){
          //TODO: support multiple dimension offset---currently we only support 1-D --- it's easy to expand here but what about the other variables we need to pass to the CreateH2D function all?
          size_t offsets [h2d_["offset"].Size()];
          for (rapidjson::SizeType l = 0; l < h2d_["offset"].Size(); l++){
            offsets[l] = static_cast<size_t>(h2d_["offset"][l].GetInt());
          }
          offset = offsets[0];
        } else if(h2d_["offset"].IsString()){
          void* p_off = GetParameterInput(params, h2d_["offset"].GetString());
          offset = p_off ? (*(size_t*) p_off) : atol(h2d_["offset"].GetString());
        } else{
          _error("malformed command h2d offset in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        //size
        if(!h2d_.HasMember("size")){
          _error("missing command h2d size in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        size_t size;
        if(h2d_["size"].IsInt()){
          size = static_cast<size_t>(h2d_["size"].GetInt());
        } else if(h2d_["size"].IsArray()){
          //TODO: support multiple dimension sizes---currently we only support 1-D
          size_t sizes [h2d_["size"].Size()];
          for (rapidjson::SizeType l = 0; l < h2d_["size"].Size(); l++){
            sizes[l] = static_cast<size_t>(h2d_["size"][l].GetInt());
          }
          size = sizes[0];
        } else if(h2d_["size"].IsString()){
          void* p_size = GetParameterInput(params, h2d_["size"].GetString());
          size = p_size ? (*(size_t*) p_size) : atol(h2d_["size"].GetString());
        } else{
          _error("malformed command h2d size in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        Command* cmd = Command::CreateH2D(task, (Mem *)platform_->get_mem_object(*dev_mem), offset, size, host_mem);
        //name (optional)
        if(h2d_.HasMember("name")){
          if(!h2d_["name"].IsString()){
            _error("malformed command h2d name in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
          }
          cmd->set_name(const_cast<char*>(h2d_["name"].GetString()));
        }
        task->AddCommand(cmd);
      }
      else if (iris_tasks_[i]["commands"][j].HasMember("d2h")){
        const rapidjson::Value& d2h_ = iris_tasks_[i]["commands"][j]["d2h"];
        //device_memory
        if(!d2h_.HasMember("device_memory") or !d2h_["device_memory"].IsString()){
          _error("malformed command d2h device_memory in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        iris_mem *dev_mem = (iris_mem*) GetParameterInput(params, d2h_["device_memory"].GetString());
        //if IRIS data memory --- break early
        if(d2h_.HasMember("data-memory-flush")){
          iris_mem *dev_mem = (iris_mem*) GetParameterInput(params, d2h_["device_memory"].GetString());
          Command* cmd = Command::CreateMemFlushOut(task, (DataMem *)platform_->get_mem_object(*dev_mem));
          if(d2h_.HasMember("name")){
            if(!d2h_["name"].IsString()){
              _error("malformed command d2h name in file[%s]", path);
              platform_->IncrementErrorCount();
              return IRIS_ERROR;
            }
            cmd->set_name(const_cast<char*>(d2h_["name"].GetString()));
            if (!task->given_name()) task->set_name(cmd->name());
          }
          _trace("adding data_memory flush: %s\n",cmd->name());
          task->AddCommand(cmd);
          continue;
        }
        //host_memory
        if(!d2h_.HasMember("host_memory") or !d2h_["host_memory"].IsString()){
          _error("malformed command d2h host_memory in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        void* host_mem = GetParameterInput(params, d2h_["host_memory"].GetString());
        //offset
        if(!d2h_.HasMember("offset")){
          _error("missing command d2h offset in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        size_t offset;
        if(d2h_["offset"].IsInt()){
          offset = static_cast<size_t>(d2h_["offset"].GetInt());
        } else if(d2h_["offset"].IsString()){
          void* p_off = GetParameterInput(params, d2h_["offset"].GetString());
          offset = p_off ? (*(size_t*) p_off) : atol(d2h_["offset"].GetString());
        } else{
          _error("malformed command d2h offset in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        //size
        if(!d2h_.HasMember("size")){
          _error("missing command d2h size in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }
        size_t size;
        if(d2h_["size"].IsInt()){
          size = static_cast<size_t>(d2h_["size"].GetInt());
        } else if(d2h_["size"].IsString()){
          void* p_size = GetParameterInput(params, d2h_["size"].GetString());
          size = p_size ? (*(size_t*) p_size) : atol(d2h_["size"].GetString());
        } else{
          _error("malformed command d2h size in file[%s]", path);
          platform_->IncrementErrorCount();
          return IRIS_ERROR;
        }

        Command* cmd = Command::CreateD2H(task, (Mem *)platform_->get_mem_object(*dev_mem), offset, size, host_mem);
        //name (optional)
        if(d2h_.HasMember("name")){
          if(!d2h_["name"].IsString()){
            _error("malformed command d2h name in file[%s]", path);
            platform_->IncrementErrorCount();
            return IRIS_ERROR;
          }
          cmd->set_name(const_cast<char*>(d2h_["name"].GetString()));
        }
        task->AddCommand(cmd);
      }
      else {
        _error("malformed command in file[%s]", path);
        platform_->IncrementErrorCount();
        return IRIS_ERROR;
      }
    }
    graph->AddTask(task, task->uid());
  }
  graph->get_metadata()->set_json_url(path);
  return IRIS_SUCCESS;
}

void* JSON::GetParameterInput(void** params, const char* buf){
  for (unsigned long j = 0; j < inputs_.size(); j ++){
    if (strcmp(inputs_[j],buf) == 0){
      return(params[j]);
    }
  }
  return(NULL);
}

int JSON::RecordFlush() {

  for(vector<Task*>::iterator task = tracked_tasks_.begin(); task != tracked_tasks_.end(); task++){
    ProcessTask(*task);
  }
  while (!tracked_tasks_.empty()){
    tracked_tasks_.back()->Release();
    tracked_tasks_.pop_back();
  }

  rapidjson::Value json_d(rapidjson::kObjectType);
  rapidjson::Value iris_graph(rapidjson::kObjectType);
  //inputs
  rapidjson::Value inputs_(rapidjson::kArrayType);
  //host pointers
  rapidjson::Value tmp;
  for(std::vector<void*>::iterator iter = host_ptrs_.begin(); iter != host_ptrs_.end(); iter ++){
    tmp.SetString(NameFromHostPointer(*iter).c_str(),NameFromHostPointer(*iter).length(),iris_output_document_.GetAllocator());
    inputs_.PushBack(tmp,iris_output_document_.GetAllocator());
  }
  //device memory
  for(std::vector<Mem*>::iterator iter = mems_.begin(); iter != mems_.end(); iter ++){
    tmp.SetString(NameFromDeviceMem(*iter).c_str(),NameFromDeviceMem(*iter).length(),iris_output_document_.GetAllocator());
    inputs_.PushBack(tmp,iris_output_document_.GetAllocator());
  }
  iris_graph.AddMember("inputs",inputs_,iris_output_document_.GetAllocator());
  rapidjson::Value tasks_(rapidjson::kObjectType);
  tasks_.AddMember("tasks",iris_output_tasks_,iris_output_document_.GetAllocator());
  iris_graph.AddMember("graph",tasks_,iris_output_document_.GetAllocator());
  iris_output_document_.SetObject();
  iris_output_document_.AddMember("$schema", SCHEMA, iris_output_document_.GetAllocator());
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
  bool already_included = false;
  int index = 0;
  for(std::vector<void*>::iterator it = host_ptrs_.begin(); it != host_ptrs_.end();it ++){
    if (*it == host_ptr) { already_included = true; break; }
    index++;
  }
  if (!already_included) host_ptrs_.push_back(host_ptr);
  return index;
}

const std::string JSON::NameFromDeviceMem(Mem* dev_mem){
  return std::string("devicemem-"+std::to_string(UniqueUIDFromDevicePointer(dev_mem)));
}

int JSON::UniqueUIDFromDevicePointer(Mem* dev_ptr){
  bool already_included = false;
  int index = 0;
  for(std::vector<Mem*>::iterator it = mems_.begin(); it != mems_.end();it ++){
    if (*it == dev_ptr) { already_included = true; break; }
    index++;
  }
  if (!already_included) mems_.push_back(dev_ptr);
  return index;
}

int JSON::RecordTask(Task* task) {
  if(task->ncmds() == 0) return IRIS_SUCCESS;//skip recording IRIS_MARKER tasks
  task->Retain();
  tracked_tasks_.push_back(task);
  return IRIS_SUCCESS;
}

int JSON::ProcessTask(Task* task){
  rapidjson::Value _task(rapidjson::kObjectType);
  //name
  rapidjson::Value task_name_(task->name(),iris_output_document_.GetAllocator());
  _task.AddMember("name",task_name_,iris_output_document_.GetAllocator());
  if (task->ncmds() == 0) {
    //it's and empty (likely checkpointing) task, so don't record it.
    //just discard empty tasks (tasks without any commands)
    return IRIS_SUCCESS;
  }
  //command(s)
  rapidjson::Value _cmds(rapidjson::kArrayType);
  for (int i = 0; i < task->ncmds(); i++) {
    Command* cmd = task->cmd(i);
    if (cmd->type() == IRIS_CMD_H2D) {
      rapidjson::Value h2d_(rapidjson::kObjectType);
      rapidjson::Value h2d_name(cmd->name(),iris_output_document_.GetAllocator());
      if (cmd->name()) h2d_.AddMember("name",h2d_name,iris_output_document_.GetAllocator());
      //host_memory
      rapidjson::Value host_memory_name_(NameFromHostPointer(cmd->host()).c_str(),NameFromHostPointer(cmd->host()).length(),iris_output_document_.GetAllocator());
      h2d_.AddMember("host_memory",host_memory_name_,iris_output_document_.GetAllocator());
      //device_memory
      rapidjson::Value device_memory_name_(NameFromDeviceMem(cmd->mem()).c_str(),iris_output_document_.GetAllocator());
      h2d_.AddMember("device_memory",device_memory_name_,iris_output_document_.GetAllocator());
      h2d_.AddMember("offset",rapidjson::Value(cmd->off(0)),iris_output_document_.GetAllocator());
      h2d_.AddMember("size",rapidjson::Value(cmd->size()),iris_output_document_.GetAllocator());
      rapidjson::Value cmd_(rapidjson::kObjectType);
      cmd_.AddMember("h2d",h2d_,iris_output_document_.GetAllocator());
      _cmds.PushBack(cmd_,iris_output_document_.GetAllocator());
    }
    else if (cmd->type() == IRIS_CMD_D2H) {
      rapidjson::Value d2h_(rapidjson::kObjectType);
      rapidjson::Value command_name_(cmd->name(),iris_output_document_.GetAllocator());
      d2h_.AddMember("name",command_name_,iris_output_document_.GetAllocator());
      //host_memory
      rapidjson::Value host_memory_name_(NameFromHostPointer(cmd->host()).c_str(),iris_output_document_.GetAllocator());
      d2h_.AddMember("host_memory",host_memory_name_,iris_output_document_.GetAllocator());
      //device_memory
      rapidjson::Value device_memory_name_(NameFromDeviceMem(cmd->mem()).c_str(),iris_output_document_.GetAllocator());
      d2h_.AddMember("device_memory",device_memory_name_,iris_output_document_.GetAllocator());
      d2h_.AddMember("offset",rapidjson::Value(cmd->off(0)),iris_output_document_.GetAllocator());
      d2h_.AddMember("size",rapidjson::Value(cmd->size()),iris_output_document_.GetAllocator());
      rapidjson::Value cmd_(rapidjson::kObjectType);
      cmd_.AddMember("d2h",d2h_,iris_output_document_.GetAllocator());
      _cmds.PushBack(cmd_,iris_output_document_.GetAllocator());
    }
    else if (cmd->type() == IRIS_CMD_KERNEL) {
      rapidjson::Value kernel_(rapidjson::kObjectType);
      //name
      rapidjson::Value kernel_name_(cmd->name(),iris_output_document_.GetAllocator());
      kernel_.AddMember("name",kernel_name_,iris_output_document_.GetAllocator());
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
          rapidjson::Value kernel_arg_mem_name_(NameFromDeviceMem((Mem *)arg->mem).c_str(),iris_output_document_.GetAllocator());
          param_.AddMember("value",kernel_arg_mem_name_,iris_output_document_.GetAllocator());
          param_.AddMember("size_bytes",rapidjson::Value(arg->mem_size),iris_output_document_.GetAllocator());
          //permissions
          if (arg->mode == iris_r) param_.AddMember("permissions",rapidjson::StringRef("r"),iris_output_document_.GetAllocator());
          else if (arg->mode == iris_w) param_.AddMember("permissions",rapidjson::StringRef("w"),iris_output_document_.GetAllocator());
          else if (arg->mode == iris_rw) param_.AddMember("permissions",rapidjson::StringRef("rw"),iris_output_document_.GetAllocator());
          else _error("not valid mode[%d]", arg->mode);
        }else{//scalar
          param_.AddMember("type","scalar",iris_output_document_.GetAllocator());
          rapidjson::Value _value;
          _value.SetString(arg->value,arg->size,iris_output_document_.GetAllocator());
          param_.AddMember("value",_value,iris_output_document_.GetAllocator());
          param_.AddMember("data_size",arg->size,iris_output_document_.GetAllocator());
        }
        params_.PushBack(param_,iris_output_document_.GetAllocator());
      }
      kernel_.AddMember("parameters",params_,iris_output_document_.GetAllocator());
      rapidjson::Value cmd_(rapidjson::kObjectType);
      cmd_.AddMember("kernel",kernel_,iris_output_document_.GetAllocator());
      _cmds.PushBack(cmd_,iris_output_document_.GetAllocator());
    }
  }
  _task.AddMember("commands",_cmds,iris_output_document_.GetAllocator());
  //target
  rapidjson::Value _target;
  if (task->dev()) {
    _target.SetInt(task->devno());
  }
  else {
    if(strcmp(task->brs_policy_string(), "unknown") == 0) { _error("unknown policy in JSON file[%s]", task->brs_policy_string()); return IRIS_ERROR; }//we should never hit this!
    _target.SetString(task->brs_policy_string(), iris_output_document_.GetAllocator());
  }
  _task.AddMember("target",_target,iris_output_document_.GetAllocator());
  //depends
  rapidjson::Value _depends(rapidjson::kArrayType);
  for (int i = 0; i < task->ndepends(); i++) {
    Task *dep = task->depend(i);
    if (dep != NULL) {
      rapidjson::Value depend(dep->name(),iris_output_document_.GetAllocator());
      _depends.PushBack(depend, iris_output_document_.GetAllocator());
    }
  }
  _task.AddMember("depends",_depends,iris_output_document_.GetAllocator());
  iris_output_tasks_.PushBack(_task, iris_output_document_.GetAllocator());

  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

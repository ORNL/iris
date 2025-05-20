#include <iris/iris.h>
#include "Debug.h"
#include "Platform.h"
#include "Consistency.h"
#include "Scheduler.h"
#include "Graph.h"
#include "Task.h"
#include "Utils.h"
#include "Kernel.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "DataMemRegion.h"
#include "Timer.h"

using namespace iris::rt;

int iris_init(int* argc, char*** argv, int sync) {
  return Platform::GetPlatform()->Init(argc, argv, sync);
}

int iris_error_count() {
  Platform *platform = Platform::GetPlatform(); 
  return platform->NumErrors();
}

void iris_overview() {
  Platform *platform = Platform::GetPlatform();
  platform->ShowOverview();
  return;
}

void iris_task_retain(iris_task task, int flag) {
  Platform *platform = Platform::GetPlatform(); 
  platform->set_release_task_flag(!((bool)flag), task);
}

int iris_task_set_julia_policy(iris_task brs_task, const char *name) {
  Task *task = Platform::GetPlatform()->get_task_object(brs_task);
  task->set_julia_policy(name);
  return IRIS_SUCCESS;
}

void iris_enable_default_kernels(int flag) {
    Platform *platform = Platform::GetPlatform(); 
    platform->enable_default_kernels_load((bool) flag);
}

int iris_finalize() {
  Platform *platform = Platform::GetPlatform(); 
  int status = platform->Finalize();
  //delete platform;
  return status;
}

void iris_set_enable_profiler(int flag) {
  Platform::GetPlatform()->set_enable_profiler((bool)flag);
}

int iris_synchronize() {
  return Platform::GetPlatform()->Synchronize();
}

int iris_set_stream_policy(StreamPolicy policy) {
 Platform::GetPlatform()->set_stream_policy(policy);
 return IRIS_SUCCESS;
}

int iris_set_asynchronous(int flag) {
 Platform::GetPlatform()->set_async((bool)flag);
 return IRIS_SUCCESS;
}

int iris_env_set(const char* key, const char* value) {
  return Platform::GetPlatform()->EnvironmentSet(key, value, true);
}

int iris_env_get(const char* key, char** value, size_t* vallen) {
  return Platform::GetPlatform()->EnvironmentGet(key, value, vallen);
}

int iris_platform_count(int* nplatforms) {
  return Platform::GetPlatform()->PlatformCount(nplatforms);
}

int iris_platform_info(int platform, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->PlatformInfo(platform, param, value, size);
}

int iris_set_shared_memory_model(int flag)
{
    return Platform::GetPlatform()->SetSharedMemoryModel(flag);
}

int iris_mem_enable_usm_all(iris_mem mem)
{
    return Platform::GetPlatform()->SetSharedMemoryModel(mem, iris_model_all, true);
}

int iris_mem_enable_usm(iris_mem mem, DeviceModel type)
{
    return Platform::GetPlatform()->SetSharedMemoryModel(mem, type, true);
}

int iris_mem_disable_usm_all(iris_mem mem)
{
    return Platform::GetPlatform()->SetSharedMemoryModel(mem, iris_model_all, false);
}

int iris_mem_disable_usm(iris_mem mem, DeviceModel type)
{
    return Platform::GetPlatform()->SetSharedMemoryModel(mem, type, false);
}

int iris_ndevices() {
  return Platform::GetPlatform()->ndevs();
}

int iris_nstreams() {
  return Platform::GetPlatform()->nstreams();
}

int iris_ncopy_streams() {
  return Platform::GetPlatform()->ncopy_streams();
}

int iris_set_nstreams(int n) {
  Platform::GetPlatform()->set_nstreams(n);
  return IRIS_SUCCESS;
}

int iris_set_ncopy_streams(int n) {
  Platform::GetPlatform()->set_ncopy_streams(n);
  return IRIS_SUCCESS;
}

int iris_device_count(int* ndevs) {
  return Platform::GetPlatform()->DeviceCount(ndevs);
}

int iris_device_info(int device, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->DeviceInfo(device, param, value, size);
}

int iris_device_set_default(int device) {
  return Platform::GetPlatform()->DeviceSetDefault(device);
}

int iris_device_get_default(int* device) {
  return Platform::GetPlatform()->DeviceGetDefault(device);
}

int iris_device_synchronize(int ndevs, int* devices) {
  return Platform::GetPlatform()->DeviceSynchronize(ndevs, devices);
}

int iris_register_policy(const char* lib, const char* name, void* params) {
  return Platform::GetPlatform()->PolicyRegister(lib, name, params);
}

iris_task iris_task_create_struct() {
  iris_task task;
  Platform::GetPlatform()->TaskCreate(NULL, false, &task);
  return task;
}

int iris_task_create(iris_task* task) {
  return Platform::GetPlatform()->TaskCreate(NULL, false, task);
}

int iris_task_enable_julia_interface(iris_task brs_task, int type) {
  Task *task = Platform::GetPlatform()->get_task_object(brs_task);
  task->set_enable_julia_if();
  task->set_julia_kernel_type(type);
  return IRIS_SUCCESS;
}

int iris_task_create_perm(iris_task* task) {
  return Platform::GetPlatform()->TaskCreate(NULL, true, task);
}

int iris_task_create_name(const char* name, iris_task* task) {
  return Platform::GetPlatform()->TaskCreate(name, false, task);
}

int iris_task_depend(iris_task task, int ntasks, iris_task* tasks) {
  return Platform::GetPlatform()->TaskDepend(task, ntasks, tasks);
}

int iris_task_malloc(iris_task task, iris_mem mem) {
  return Platform::GetPlatform()->TaskMalloc(task, mem);
}

int iris_task_cmd_reset_mem(iris_task task, iris_mem mem, uint8_t reset) {
  return Platform::GetPlatform()->TaskMemResetInput(task, mem, reset);
}

int iris_task_set_stream_policy(iris_task brs_task, StreamPolicy policy) {
 Task *task = Platform::GetPlatform()->get_task_object(brs_task);
 if (task != NULL) task->set_stream_policy(policy);
 return IRIS_SUCCESS;
}

void iris_task_disable_asynchronous(iris_task brs_task) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    task->set_async(false);
}

int iris_task_get_metadata(iris_task brs_task, int index) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->metadata(index);
}

int *iris_task_get_metadata_all(iris_task brs_task) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->metadata();
}
int iris_task_set_metadata_all(iris_task brs_task, int *mdata, int n) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->set_metadata(mdata, n);
}
int iris_task_set_metadata(iris_task brs_task, int index, int metadata) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    task->set_metadata(index, metadata);
    return IRIS_SUCCESS;
}

int iris_task_get_metadata_count(iris_task brs_task) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->n_metadata();
}

int iris_task_h2broadcast(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskH2Broadcast(task, mem, off, size, host);
}

int iris_task_h2broadcast_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  return Platform::GetPlatform()->TaskH2Broadcast(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
}

int iris_task_d2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host, int src_dev) {
  return Platform::GetPlatform()->TaskD2D(task, mem, off, size, host, src_dev);
}

int iris_task_dmem2dmem(iris_task task, iris_mem src_mem, iris_mem dst_mem) {
  return Platform::GetPlatform()->TaskDMEM2DMEM(task, src_mem, dst_mem);
}

int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, size, host);
}

int iris_task_hidden_dmem(iris_task brs_task, iris_mem brs_mem, int mode) {
  Task *task = Platform::GetPlatform()->get_task_object(brs_task);
  BaseMem * mem = Platform::GetPlatform()->get_mem_object(brs_mem);
  if (mem->GetMemHandlerType() == IRIS_DMEM || 
          mem->GetMemHandlerType() == IRIS_DMEM_REGION) {
    DataMem *dmem = (DataMem *)mem;
    task->insert_hidden_dmem(dmem, mode);
  }
  return IRIS_SUCCESS;
}

int iris_task_dmem_h2d(iris_task task, iris_mem mem) {
  return Platform::GetPlatform()->TaskH2D(task, mem, 0, 0, NULL);
}

int iris_task_h2d_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
}

int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, size, host);
}

int iris_task_dmem_d2h(iris_task task, iris_mem mem) {
  return Platform::GetPlatform()->TaskD2H(task, mem, 0, 0, NULL);
}

int iris_task_d2h_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
}

int iris_task_dmem_flush_out(iris_task task, iris_mem mem) {
  return Platform::GetPlatform()->TaskMemFlushOut(task, mem);
}

size_t *iris_dmem_get_host_size(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->host_size();
    return NULL;
}

int iris_dmem_set_source(iris_mem brs_mem, iris_mem source_mem)
{
    BaseMem* mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    BaseMem* source = (BaseMem *)Platform::GetPlatform()->get_mem_object(source_mem);
    mem->set_source_mem(source);
    return IRIS_SUCCESS;
}

int iris_dmem_get_elem_type(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->element_type();
    return 0;
}

int iris_dmem_get_elem_size(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->elem_size();
    return 0;
}

int iris_dmem_get_dim(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->dim();
    return 0;
}

void *iris_get_dmem_valid_host(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->host_memory();
    return NULL;
}

void *iris_get_dmem_host(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->host_ptr();
    return NULL;
}

void *iris_get_dmem_host_fetch(iris_mem brs_mem) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL)  {
        void *host_ptr = mem->host_memory();
        mem->FetchDataFromDevice(host_ptr);
        return host_ptr;
    }
    return NULL;
}

void *iris_get_dmem_host_fetch_with_size(iris_mem brs_mem, size_t size) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL)  {
        void *host_ptr = mem->host_memory();
        mem->FetchDataFromDevice(host_ptr, size);
        return host_ptr;
    }
    return NULL;
}

int iris_fetch_dmem_data(iris_mem brs_mem, void *host_ptr) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    mem->FetchDataFromDevice(host_ptr);
    return IRIS_SUCCESS;
}

int iris_fetch_dmem_data_with_size(iris_mem brs_mem, void *host_ptr, size_t size) {
    DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
    mem->FetchDataFromDevice(host_ptr, size);
    return IRIS_SUCCESS;
}

int iris_task_h2d_full(iris_task task, iris_mem mem, void* host) {
  return Platform::GetPlatform()->TaskH2DFull(task, mem, host);
}

int iris_task_h2broadcast_full(iris_task task, iris_mem mem, void* host) {
  return Platform::GetPlatform()->TaskH2BroadcastFull(task, mem, host);
}

int iris_task_d2h_full(iris_task task, iris_mem mem, void* host) {
  return Platform::GetPlatform()->TaskD2HFull(task, mem, host);
}

int iris_task_kernel_object(iris_task task, iris_kernel kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws);
}

int iris_task_kernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, NULL, params_info, NULL);
}

int iris_task_kernel_v2(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, NULL);
}

int iris_task_kernel_v3(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, memranges);
}

int iris_task_kernel_selector(iris_task task, iris_selector_kernel func, void* params, size_t params_size) {
  return Platform::GetPlatform()->TaskKernelSelector(task, func, params, params_size);
}
int iris_task_kernel_launch_disabled(iris_task brs_task, int flag)
{
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    task->set_kernel_launch_disabled((bool)flag);
    return IRIS_SUCCESS;
}   

int iris_params_map(iris_task task, int *params_map) {
  return Platform::GetPlatform()->SetParamsMap(task, params_map);
}

int iris_task_python_host(iris_task task, iris_host_python_task func, int64_t params_id) {
  return Platform::GetPlatform()->TaskHost(task, func, params_id);
}

int iris_task_host(iris_task task, iris_host_task func, void* params) {
  return Platform::GetPlatform()->TaskHost(task, func, params);
}

int iris_task_custom(iris_task task, int tag, void* params, size_t params_size) {
  return Platform::GetPlatform()->TaskCustom(task, tag, params, params_size);
}

int iris_task_submit(iris_task task, int device, const char* opt, int sync) {
  return Platform::GetPlatform()->TaskSubmit(task, device, opt, sync);
}

int iris_task_set_policy(iris_task task, int policy) {
  return Platform::GetPlatform()->SetTaskPolicy(task, policy);
}

int iris_task_get_policy(iris_task task) {
  return Platform::GetPlatform()->GetTaskPolicy(task);
}
int iris_task_wait(iris_task task) {
  return Platform::GetPlatform()->TaskWait(task);
}

int iris_task_wait_all(int ntasks, iris_task* tasks) {
  return Platform::GetPlatform()->TaskWaitAll(ntasks, tasks);
}

int iris_task_kernel_cmd_only(iris_task task) {
  return Platform::GetPlatform()->TaskKernelCmdOnly(task);
}

int iris_task_release(iris_task task) {
  return Platform::GetPlatform()->TaskRelease(task);
}

int iris_task_info(iris_task task, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->TaskInfo(task, param, value, size);
}

int iris_task_release_mem(iris_task task, iris_mem mem) {
  return Platform::GetPlatform()->TaskReleaseMem(task, mem);
}

void iris_task_set_name(iris_task brs_task, const char *name) {
   Task *task = Platform::GetPlatform()->get_task_object(brs_task);
   task->set_name(name);
}

const char *iris_kernel_get_name(iris_kernel brs_kernel) {
    Kernel *k= Platform::GetPlatform()->get_kernel_object(brs_kernel);
    return k->name();
}

int iris_task_disable_consistency(iris_task brs_task) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    task->set_disable_consistency(true);
    return IRIS_SUCCESS;
}

const char *iris_task_get_name(iris_task brs_task) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return const_cast<char*>(task->name());
}

int iris_task_get_dependency_count(iris_task brs_task) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->ndepends();
}
void iris_task_get_dependencies(iris_task brs_task, iris_task *tasks) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    for(int i=0; i<task->ndepends(); i++) {
        tasks[i] = *(task->depend(i)->struct_obj());
    }
}
int iris_cmd_kernel_get_nargs(void *cmd_p) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_nargs();
}
int iris_cmd_kernel_get_arg_is_mem(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return (cmd->kernel_arg(index)->mem != NULL);
}
size_t iris_cmd_kernel_get_arg_size(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_arg(index)->size;
}
void  *iris_cmd_kernel_get_arg_value(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_arg(index)->value;
}
iris_mem iris_cmd_kernel_get_arg_mem(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return *(cmd->kernel_arg(index)->mem->struct_obj());
}
size_t iris_cmd_kernel_get_arg_mem_off(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_arg(index)->mem_off;
}
size_t iris_cmd_kernel_get_arg_mem_size(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_arg(index)->mem_size;
}
size_t iris_cmd_kernel_get_arg_off(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_arg(index)->off;
}
int    iris_cmd_kernel_get_arg_mode(void *cmd_p, int index) {
    Command *cmd = (Command *)cmd_p;
    return cmd->kernel_arg(index)->mode;
}
int iris_task_kernel_dmem_fetch_order(iris_task brs_task, int *order) {
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    if (task->cmd_kernel() && task->cmd_kernel()->kernel()) 
        task->cmd_kernel()->kernel()->set_order(order);
    return IRIS_SUCCESS;
}
iris_kernel iris_task_get_kernel(iris_task brs_task)
{
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return *(task->cmd_kernel()->kernel()->struct_obj());
}
int iris_task_is_cmd_kernel_exists(iris_task brs_task)
{
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->cmd_kernel() != NULL;
}

void *iris_task_get_cmd_kernel(iris_task brs_task)
{
    Task *task = Platform::GetPlatform()->get_task_object(brs_task);
    return task->cmd_kernel();
}

unsigned long iris_kernel_get_uid(iris_kernel brs_kernel)
{
  return brs_kernel.uid;
}
unsigned long iris_task_get_uid(iris_task brs_task) {
  return brs_task.uid;
}

int iris_mem_create(size_t size, iris_mem* mem) {
  return Platform::GetPlatform()->MemCreate(size, mem);
}
size_t iris_mem_get_size(iris_mem mem) {
  return Platform::GetPlatform()->get_mem_object(mem)->size();
}

int iris_mem_get_type(iris_mem mem) {
  return Platform::GetPlatform()->get_mem_object(mem)->GetMemHandlerType();
}

int iris_mem_get_uid(iris_mem mem) {
  return Platform::GetPlatform()->get_mem_object(mem)->uid();
}

int iris_get_mem_element_type(iris_mem brs_mem) {
    BaseMem * mem = (BaseMem*)Platform::GetPlatform()->get_mem_object(brs_mem);
    if (mem != NULL) return mem->element_type();
    return iris_unknown;
}

int iris_mem_is_reset(iris_mem mem) {
  return Platform::GetPlatform()->get_mem_object(mem)->is_reset();
}

int iris_mem_init_reset(iris_mem brs_mem, int memset_value) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->set_reset_type(iris_reset_memset);
  mem->init_reset((bool) memset_value);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_assign(iris_mem brs_mem, IRISValue value) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->set_reset_assign(value);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_arith_seq(iris_mem brs_mem, IRISValue start, IRISValue increment) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->set_reset_arith_seq(start, increment);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_geom_seq(iris_mem brs_mem, IRISValue start, IRISValue step) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->set_reset_geom_seq(start, step);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_random_uniform_seq(iris_mem brs_mem, long long seed, IRISValue min, IRISValue max) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->init_reset(true);
  mem->set_reset_type(iris_reset_random_uniform_seq);
  mem->set_reset_seed(seed);
  mem->set_reset_min(min);
  mem->set_reset_max(max);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_random_normal_seq(iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->init_reset(true);
  mem->set_reset_type(iris_reset_random_normal_seq);
  mem->set_reset_seed(seed);
  mem->set_reset_mean(mean);
  mem->set_reset_stddev(stddev);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_random_log_normal_seq(iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->init_reset(true);
  mem->set_reset_type(iris_reset_random_log_normal_seq);
  mem->set_reset_seed(seed);
  mem->set_reset_mean(mean);
  mem->set_reset_stddev(stddev);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_random_uniform_sobol_seq(iris_mem brs_mem, IRISValue min, IRISValue max) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->init_reset(true);
  mem->set_reset_type(iris_reset_random_uniform_sobol_seq);
  mem->set_reset_min(min);
  mem->set_reset_max(max);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_random_normal_sobol_seq(iris_mem brs_mem, IRISValue mean, IRISValue stddev) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->init_reset(true);
  mem->set_reset_type(iris_reset_random_normal_sobol_seq);
  mem->set_reset_mean(mean);
  mem->set_reset_stddev(stddev);
  return IRIS_SUCCESS;
}
int iris_mem_init_reset_random_log_normal_sobol_seq(iris_mem brs_mem, IRISValue mean, IRISValue stddev) {
  BaseMem * mem = (BaseMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->init_reset(true);
  mem->set_reset_type(iris_reset_random_log_normal_sobol_seq);
  mem->set_reset_mean(mean);
  mem->set_reset_stddev(stddev);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_assign(iris_task brs_task, iris_mem brs_mem, IRISValue value) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_assign; 
  reset_data.value_ = value;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_arith_seq(iris_task brs_task, iris_mem brs_mem, IRISValue start, IRISValue increment) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_arith_seq; 
  reset_data.start_ = start;
  reset_data.step_ = increment;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_geom_seq(iris_task brs_task, iris_mem brs_mem, IRISValue start, IRISValue step) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_geom_seq; 
  reset_data.start_ = start;
  reset_data.step_ = step;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_random_uniform_seq(iris_task brs_task, iris_mem brs_mem, long long seed, IRISValue min, IRISValue max) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_random_uniform_seq; 
  reset_data.seed_ = seed;
  reset_data.p1_ = min;
  reset_data.p2_ = max;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_random_normal_seq(iris_task brs_task, iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_random_normal_seq; 
  reset_data.seed_ = seed;
  reset_data.p1_ = mean;
  reset_data.p2_ = stddev;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_random_log_normal_seq(iris_task brs_task, iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_random_log_normal_seq; 
  reset_data.seed_ = seed;
  reset_data.p1_ = mean;
  reset_data.p2_ = stddev;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_random_uniform_sobol_seq(iris_task brs_task, iris_mem brs_mem, IRISValue min, IRISValue max) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_random_uniform_sobol_seq; 
  reset_data.p1_ = min;
  reset_data.p2_ = max;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_random_normal_sobol_seq(iris_task brs_task, iris_mem brs_mem, IRISValue mean, IRISValue stddev) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_random_normal_sobol_seq; 
  reset_data.p1_ = mean;
  reset_data.p2_ = stddev;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_task_cmd_init_reset_random_log_normal_sobol_seq(iris_task brs_task, iris_mem brs_mem, IRISValue mean, IRISValue stddev) {
  ResetData reset_data;
  reset_data.reset_type_ = iris_reset_random_log_normal_sobol_seq; 
  reset_data.p1_ = mean;
  reset_data.p2_ = stddev;
  Platform::GetPlatform()->TaskMemResetInput(brs_task, brs_mem, reset_data);
  return IRIS_SUCCESS;
}
int iris_data_mem_init_reset(iris_mem mem, int reset) {
  return Platform::GetPlatform()->DataMemInit(mem, (bool)reset);
}
int iris_data_mem_create_tile(iris_mem* mem, void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
  return Platform::GetPlatform()->DataMemCreate(mem, host, off, host_size, dev_size, elem_size, dim);
}
int iris_data_mem_create_nd(iris_mem *mem, void *host, size_t *size, int dim, size_t elem_size, int element_type) {
  return Platform::GetPlatform()->DataMemCreate(mem, host, size, dim, elem_size, element_type);
}
int iris_data_mem_create(iris_mem *mem, void *host, size_t size) {
  return Platform::GetPlatform()->DataMemCreate(mem, host, size);
}
int iris_data_mem_create_symbol(iris_mem *mem, void *host, size_t size, const char *symbol) {
  return Platform::GetPlatform()->DataMemCreate(mem, host, size, symbol);
}
iris_mem *iris_data_mem_create_ptr(void *host, size_t size) {
  return Platform::GetPlatform()->DataMemCreate(host, size);
}
iris_mem iris_data_mem_create_struct(void *host, size_t size) {
  iris_mem mem;
  Platform::GetPlatform()->DataMemCreate(&mem, host, size);
  return mem;
}
iris_mem iris_data_mem_create_struct_nd(void *host, size_t *size, int dim, size_t element_size, int element_type) {
  iris_mem mem;
  Platform::GetPlatform()->DataMemCreate(&mem, host, size, dim, element_size, element_type);
  return mem;
}
iris_mem iris_data_mem_create_struct_with_type(void *host, size_t size, int element_type) {
  iris_mem mem;
  Platform::GetPlatform()->DataMemCreate(&mem, host, size, element_type);
  return mem;
}
iris_mem *iris_data_mem_create_tile_ptr(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
  return Platform::GetPlatform()->DataMemCreate(host, off, host_size, dev_size, elem_size, dim);
}
iris_mem iris_data_mem_create_tile_struct(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
  iris_mem mem;
  Platform::GetPlatform()->DataMemCreate(&mem, host, off, host_size, dev_size, elem_size, dim);
  return mem;
}
iris_mem iris_data_mem_create_tile_struct_with_type(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type) {
  iris_mem mem;
  Platform::GetPlatform()->DataMemCreate(&mem, host, off, host_size, dev_size, elem_size, dim, element_type);
  return mem;
}
int iris_dmem_add_child(iris_mem brs_parent, iris_mem brs_child, size_t offset) {
  DataMem* parent = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_parent);
  DataMem* child = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_child);
  parent->AddChild(child, offset);
  return IRIS_SUCCESS;
}
int iris_data_mem_clear(iris_mem brs_mem) {
  DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->clear();
  return IRIS_SUCCESS;
}
int iris_data_mem_update(iris_mem mem, void *host) {
  return Platform::GetPlatform()->DataMemUpdate(mem, host);
}
int iris_data_mem_refresh(iris_mem brs_mem) {
  DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->RefreshHost();
  return IRIS_SUCCESS;
}
int iris_data_mem_update_host_size(iris_mem brs_mem, size_t *host_size) {
  DataMem* mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  return mem->update_host_size(host_size);
}
int iris_unregister_pin_memory(void *host) {
  return Platform::GetPlatform()->UnRegisterPin(host);
}
int iris_register_pin_memory(void *host, size_t size) {
  return Platform::GetPlatform()->RegisterPin(host, size);
}
int iris_data_mem_set_pin_flag(bool flag) {
  Platform::GetPlatform()->set_dmem_register_pin_flag((bool)flag);
  return IRIS_SUCCESS;
}
int iris_data_mem_pin(iris_mem mem) {
  return Platform::GetPlatform()->DataMemRegisterPin(mem);
}
int iris_data_mem_unpin(iris_mem mem) {
  return Platform::GetPlatform()->DataMemUnRegisterPin(mem);
}
int iris_data_mem_create_region(iris_mem *mem, iris_mem root_mem, int region) {
  return Platform::GetPlatform()->DataMemCreate(mem, root_mem, region);
}
iris_mem iris_data_mem_create_region_struct(iris_mem root_mem, int region) {
  iris_mem mem;
  Platform::GetPlatform()->DataMemCreate(&mem, root_mem, region);
  return mem;
}
iris_mem *iris_data_mem_create_region_ptr(iris_mem root_mem, int region) {
  return Platform::GetPlatform()->DataMemCreate(root_mem, region);
}
int iris_data_mem_n_regions(iris_mem brs_mem) {
  DataMem *mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  return mem->get_n_regions();
}
int iris_data_mem_update_bc(iris_mem brs_mem, int bc, int row, int col) {
  DataMem *mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  mem->update_bc_row_col((bool)bc, row, col);
  return IRIS_SUCCESS;
}
int iris_data_mem_get_rr_bc_dev(iris_mem brs_mem){
  DataMem *mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  return mem->get_rr_bc_dev();
}
unsigned long iris_data_mem_get_region_uid(iris_mem brs_mem, int region) {
  DataMem *mem = (DataMem *)Platform::GetPlatform()->get_mem_object(brs_mem);
  return mem->get_region(region)->uid();
}
int iris_data_mem_enable_outer_dim_regions(iris_mem mem) {
  return Platform::GetPlatform()->DataMemEnableOuterDimRegions(mem);
}
iris_mem iris_get_dmem_for_region(iris_mem brs_mem) {
    DataMemRegion *obj = (DataMemRegion*)Platform::GetPlatform()->get_mem_object(brs_mem);
    return *(obj->get_dmem()->struct_obj());
}

int iris_mem_arch(iris_mem mem, int device, void** arch) {
  return Platform::GetPlatform()->MemArch(mem, device, arch);
}

void *iris_mem_arch_ptr(iris_mem mem, int device) {
  void* arch;
  Platform::GetPlatform()->MemArch(mem, device, &arch);
  return arch;
}

int iris_mem_release(iris_mem mem) {
  return Platform::GetPlatform()->MemRelease(mem);
}

int iris_register_command(int tag, int device, command_handler handler) {
  return Platform::GetPlatform()->RegisterCommand(tag, device, handler);
}

int iris_register_hooks_task(hook_task pre, hook_task post) {
  return Platform::GetPlatform()->RegisterHooksTask(pre, post);
}

int iris_register_hooks_command(hook_command pre, hook_command post) {
  return Platform::GetPlatform()->RegisterHooksCommand(pre, post);
}

int iris_kernel_create(const char* name, iris_kernel* kernel) {
  return Platform::GetPlatform()->KernelCreate(name, kernel);
}

iris_kernel iris_kernel_create_struct(const char* name) {
  iris_kernel kernel;
  Platform::GetPlatform()->KernelCreate(name, &kernel);
  return kernel;
}

int iris_kernel_get(const char* name, iris_kernel* kernel) {
  return Platform::GetPlatform()->KernelGet(name, kernel);
}

int iris_kernel_setarg(iris_kernel kernel, int idx, size_t size, void* value) {
  return Platform::GetPlatform()->KernelSetArg(kernel, idx, size, value);
}

int iris_kernel_setmem(iris_kernel kernel, int idx, iris_mem mem, size_t mode) {
  return Platform::GetPlatform()->KernelSetMem(kernel, idx, mem, 0, mode);
}

int iris_kernel_setmem_off(iris_kernel kernel, int idx, iris_mem mem, size_t off, size_t mode) {
  return Platform::GetPlatform()->KernelSetMem(kernel, idx, mem, off, mode);
}

int iris_kernel_release(iris_kernel kernel) {
  return Platform::GetPlatform()->KernelRelease(kernel);
}

int iris_graph_create(iris_graph* graph) {
  return Platform::GetPlatform()->GraphCreate(graph);
}

iris_graph iris_graph_create_empty() {
  iris_graph graph;
  Platform::GetPlatform()->GraphCreate(&graph);
  return graph;
}

int iris_graph_create_null(iris_graph* graph) {
  iris_graph null_graph;
  null_graph.uid = (unsigned long) -1;
  null_graph.class_obj = NULL;
  *graph = null_graph;
  return IRIS_SUCCESS;
}

int iris_is_graph_null(iris_graph graph) {
  if (graph.uid == (unsigned long) -1) return (int)true;
  return (int)false;
}

int iris_graph_create_json(const char* json, void** params, iris_graph* graph) {
  return Platform::GetPlatform()->GraphCreateJSON(json, params, graph);
}

int iris_graph_task(iris_graph graph, iris_task task, int device, const char* opt) {
  return Platform::GetPlatform()->GraphTask(graph, task, device, opt);
}
int iris_graph_tasks_order(iris_graph brs_graph, int *order) {
    Graph *graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    graph->set_order(order);
    return IRIS_SUCCESS;
}

int iris_graph_reset_memories(iris_graph brs_graph) {
  Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
  graph->ResetMemories();
  return IRIS_SUCCESS;
}
int iris_graph_retain(iris_graph graph, int flag) {
  return Platform::GetPlatform()->GraphRetain(graph, (bool)flag);
}

int iris_graph_release(iris_graph graph) {
  return Platform::GetPlatform()->GraphRelease(graph);
}

int iris_graph_submit(iris_graph graph, int device, int sync) {
  return Platform::GetPlatform()->GraphSubmit(graph, device, sync);
}

int iris_graph_submit_with_order(iris_graph graph, int *order, int device, int sync) {
  return Platform::GetPlatform()->GraphSubmit(graph, order, device, sync);
}

int iris_graph_wait(iris_graph graph) {
  return Platform::GetPlatform()->GraphWait(graph);
}

int iris_graph_submit_with_time(iris_graph graph, double *time, int device, int sync)
{
    double st_time, end_time;
    Platform::GetPlatform()->TimerNow(&st_time);
    int status = Platform::GetPlatform()->GraphSubmit(graph, device, sync);
    Platform::GetPlatform()->TimerNow(&end_time);
    *time = end_time - st_time;
    return status;
}
int iris_graph_submit_with_order_and_time(iris_graph graph, int *order, double *time, int device, int sync)
{
    double st_time, end_time;
    Platform::GetPlatform()->TimerNow(&st_time);
    int status = Platform::GetPlatform()->GraphSubmit(graph, order, device, sync);
    Platform::GetPlatform()->TimerNow(&end_time);
    *time = end_time - st_time;
    return status;
}
int iris_graph_wait_all(int ngraphs, iris_graph* graphs) {
  return Platform::GetPlatform()->GraphWaitAll(ngraphs, graphs);
}

int iris_graph_free(iris_graph brs_graph) {
  return Platform::GetPlatform()->GraphFree(brs_graph);
}

void iris_enable_d2d() {
  Platform::GetPlatform()->enable_d2d();
}

void iris_disable_d2d() {
  Platform::GetPlatform()->disable_d2d();
}

void iris_disable_consistency_check() {
  Platform::GetPlatform()->scheduler()->consistency()->Disable();
}

void iris_enable_consistency_check() {
  Platform::GetPlatform()->scheduler()->consistency()->Enable();
}

int iris_record_start() {
  return Platform::GetPlatform()->RecordStart();
}

int iris_record_stop() {
  return Platform::GetPlatform()->RecordStop();
}

int iris_timer_now(double* time) {
  return Platform::GetPlatform()->TimerNow(time);
}

int iris_graph_enable_mem_profiling(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    graph->enable_mem_profiling();
    return IRIS_SUCCESS;
}
int iris_graph_get_tasks(iris_graph graph, iris_task *tasks) {
  return Platform::GetPlatform()->GetGraphTasks(graph, tasks);
}
int iris_graph_tasks_count(iris_graph graph)
{
    return Platform::GetPlatform()->GetGraphTasksCount(graph);
}
int iris_get_graph_max_theoretical_parallelism(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->get_max_parallelism();
    return IRIS_SUCCESS;
}

int iris_get_graph_dependency_adj_matrix(iris_graph brs_graph, int8_t *dep_matrix)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->get_dependency_matrix(dep_matrix, true);
    return IRIS_SUCCESS;
}
int iris_get_graph_dependency_adj_list(iris_graph brs_graph, int8_t *dep_matrix)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->get_dependency_matrix(dep_matrix, false);
    return IRIS_SUCCESS;
}
size_t iris_get_graph_3d_comm_data_size(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    return gm->comm_task_data_size();
}
void *iris_get_graph_3d_comm_data_ptr(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    CommData3D *comm_data = gm->comm_task_data();
    return comm_data;
}
size_t iris_get_graph_tasks_execution_schedule_count(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    return gm->task_schedule_count();
}
void *iris_get_graph_tasks_execution_schedule(iris_graph brs_graph, int kernel_profile)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->fetch_task_execution_schedules(kernel_profile);
    TaskProfile *tasks_data = gm->task_schedule_data();
    return tasks_data;
}
size_t iris_get_graph_dataobjects_execution_schedule_count(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    return gm->dataobject_schedule_count();
}
void *iris_get_graph_dataobjects_execution_schedule(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->fetch_dataobject_execution_schedules();
    DataObjectProfile *mems_data = gm->dataobject_schedule_data();
    return mems_data;
}
size_t iris_count_mems(iris_graph brs_graph)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    return gm->count_mems();
}
int iris_get_graph_3d_comm_time(iris_graph brs_graph, double *comm_time, int *mem_ids, int iterations, int pin_memory_flag)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->get_3d_comm_time(comm_time, mem_ids, iterations, pin_memory_flag);
    return IRIS_SUCCESS;
}
int iris_get_graph_3d_comm_data(iris_graph brs_graph, void *comm_data)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->get_3d_comm_data();
    return IRIS_SUCCESS;
}
int iris_get_graph_2d_comm_adj_matrix(iris_graph brs_graph, size_t *size_data)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->get_2d_comm_adj_matrix(size_data);
    return IRIS_SUCCESS;
}
int iris_calibrate_compute_cost_adj_matrix_only_for_types(iris_graph brs_graph, double *comp_data)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->calibrate_compute_cost_adj_matrix(comp_data, true);
    return IRIS_SUCCESS;
}
int iris_calibrate_compute_cost_adj_matrix(iris_graph brs_graph, double *comp_data)
{
    Graph* graph = Platform::GetPlatform()->get_graph_object(brs_graph);
    shared_ptr<GraphMetadata> gm = graph->get_metadata();
    gm->calibrate_compute_cost_adj_matrix(comp_data);
    return IRIS_SUCCESS;
}
int iris_calibrate_communication_cost(double *data, size_t data_size, int iterations, int pin_memory_flag)
{
    Platform::GetPlatform()->CalibrateCommunicationMatrix(data, data_size, iterations, (bool)pin_memory_flag);
    return IRIS_SUCCESS;
}
void iris_free_array(void *ptr)
{
    free(ptr);
}
int8_t *iris_allocate_array_int8_t(int SIZE, int8_t init)
{
    return iris::AllocateArray<int8_t>(SIZE, init);
}
int16_t *iris_allocate_array_int16_t(int SIZE, int16_t init)
{
    return iris::AllocateArray<int16_t>(SIZE, init);
}
int32_t *iris_allocate_array_int32_t(int SIZE, int32_t init)
{
    return iris::AllocateArray<int32_t>(SIZE, init);
}
int64_t *iris_allocate_array_int64_t(int SIZE, int64_t init)
{
    return iris::AllocateArray<int64_t>(SIZE, init);
}
size_t *iris_allocate_array_size_t(int SIZE, size_t init)
{
    return iris::AllocateArray<size_t>(SIZE, init);
}
float *iris_allocate_array_float(int SIZE, float init)
{
    return iris::AllocateArray<float>(SIZE, init);
}
double *iris_allocate_array_double(int SIZE, double init)
{
    return iris::AllocateArray<double>(SIZE, init);
}
int8_t *iris_allocate_random_array_int8_t(int SIZE)
{
    return iris::AllocateRandomArray<int8_t>(SIZE);
}
int16_t *iris_allocate_random_array_int16_t(int SIZE)
{
    return iris::AllocateRandomArray<int16_t>(SIZE);
}
int32_t *iris_allocate_random_array_int32_t(int SIZE)
{
    return iris::AllocateRandomArray<int32_t>(SIZE);
}
int64_t *iris_allocate_random_array_int64_t(int SIZE)
{
    return iris::AllocateRandomArray<int64_t>(SIZE);
}
size_t *iris_allocate_random_array_size_t(int SIZE)
{
    return iris::AllocateRandomArray<size_t>(SIZE);
}
float *iris_allocate_random_array_float(int SIZE)
{
    return iris::AllocateRandomArray<float>(SIZE);
}
double *iris_allocate_random_array_double(int SIZE)
{
    return iris::AllocateRandomArray<double>(SIZE);
}
void iris_print_matrix_limited_double(double *data, int M, int N, const char *description, int limit) 
{
    return iris::PrintMatrixLimited<double>(data, M, N, description, limit);
}
void iris_print_matrix_full_double(double *data, int M, int N, const char *description) 
{
    iris::PrintMatrixFull<double>(data, M, N, description);
}
void iris_print_matrix_limited_float(float *data, int M, int N, const char *description, int limit) 
{
    return iris::PrintMatrixLimited<float>(data, M, N, description, limit);
}
void iris_print_matrix_full_float(float *data, int M, int N, const char *description) 
{
    iris::PrintMatrixFull<float>(data, M, N, description);
}
void iris_print_matrix_limited_int64_t(int64_t *data, int M, int N, const char *description, int limit) 
{
    return iris::PrintMatrixLimited<int64_t>(data, M, N, description, limit);
}
void iris_print_matrix_full_int64_t(int64_t *data, int M, int N, const char *description) 
{
    iris::PrintMatrixFull<int64_t>(data, M, N, description);
}
void iris_print_matrix_limited_int32_t(int32_t *data, int M, int N, const char *description, int limit) 
{
    return iris::PrintMatrixLimited<int32_t>(data, M, N, description, limit);
}
void iris_print_matrix_full_int32_t(int32_t *data, int M, int N, const char *description) 
{
    iris::PrintMatrixFull<int32_t>(data, M, N, description);
}
void iris_print_matrix_limited_int16_t(int16_t *data, int M, int N, const char *description, int limit) 
{
    return iris::PrintMatrixLimited<int16_t>(data, M, N, description, limit);
}
void iris_print_matrix_full_int16_t(int16_t *data, int M, int N, const char *description) 
{
    iris::PrintMatrixFull<int16_t>(data, M, N, description);
}
void iris_print_matrix_limited_int8_t(int8_t *data, int M, int N, const char *description, int limit) 
{
    return iris::PrintMatrixLimited<int8_t>(data, M, N, description, limit);
}
void iris_print_matrix_full_int8_t(int8_t *data, int M, int N, const char *description) 
{
    iris::PrintMatrixFull<int8_t>(data, M, N, description);
}

int iris_logo(){
  Utils::Logo(true);
  return IRIS_SUCCESS;
}

void iris_println(const char *s)
{
    printf("%s\n", s);
}

int iris_read_bool_env(const char *env_name) {
    bool flag=false;
    Platform::GetPlatform()->EnvironmentBoolRead(env_name, flag);
    return (int) flag;
}

int iris_read_int_env(const char *env_name) {
    int val=0;
    Platform::GetPlatform()->EnvironmentIntRead(env_name, val);
    return val;
}

void *iris_dev_get_ctx(int device) {
  return Platform::GetPlatform()->GetDeviceContext(device);
}

void *iris_dev_get_stream(int device, int stream) {
  return Platform::GetPlatform()->GetDeviceStream(device, stream);
}

void iris_run_hpl_mapping(iris_graph graph)
{
    int ndevices = 0;
    iris_device_count(&ndevices);
    int dev_map[16][16];
    for(int i=0; i<16; i++) {
        for(int j=0; j<16; j++) {
            if (j>=i) dev_map[i][j] = j*j+i;
            else dev_map[i][j] = (i*(i+2))-j;
            dev_map[i][j] = dev_map[i][j]%ndevices;
        }
    }
    int nrows = ndevices;
    int ncols = ndevices;
    if (ndevices == 1) return;
    else if (ndevices == 9) {
        nrows = 3; ncols = 3;
    }
    else if (ndevices == 4) {
        nrows = 2; ncols = 2;
    }
    else if (ndevices == 6) {
        dev_map[0][0] = 0;
        dev_map[0][1] = 1;
        dev_map[1][0] = 2;
        dev_map[1][1] = 3;
        dev_map[2][0] = 4;
        dev_map[2][1] = 5;
        nrows = 3; ncols = 2;
    }
    else if (ndevices == 8) {
        dev_map[0][0] = 0;
        dev_map[0][1] = 1;
        dev_map[1][0] = 2;
        dev_map[1][1] = 3;
        dev_map[2][0] = 4;
        dev_map[2][1] = 5;
        dev_map[3][0] = 6;
        dev_map[3][1] = 7;
        nrows = 4; ncols = 2;
    }
    else if (ndevices % 2 == 1) {
        int incrementer = (ndevices+1) / 2;
        int i_pos = 0;
        for(int i=0; i<ndevices; i++) {
            int j_pos = (ndevices - i_pos)%ndevices;
            for(int j=0; j<ndevices; j++) {
                dev_map[i][j] = (j_pos + j)%ndevices;
            }
            i_pos = (i_pos+incrementer)%ndevices;
        }
    }
    /*printf("Dev Map:\n");
    for(int i=0; i<nrows; i++) {
        for(int j=0; j<ncols; j++) {
            printf("%2d ", dev_map[i][j]);
        }
        printf("\n");
    }*/
    int ntasks = iris_graph_tasks_count(graph);
    iris_task *tasks = NULL;
    if (ntasks > 0)
        tasks = (iris_task *)malloc(sizeof(iris_task)*ntasks);
    iris_graph_get_tasks(graph, tasks);
    for(int i=0; i<ntasks; i++) {
        iris_task task = tasks[i];
        int r = iris_task_get_metadata(task, 0);
        int c = iris_task_get_metadata(task, 1);
        if (r >= 0 && c >= 0) {
            int id = dev_map[r%nrows][c%ncols];
            iris_task_set_policy(task, id);
            //char *name = iris_task_get_name(task);
            //printf("Task %s r:%d c:%d dev:%d\n", name, r, c, id);
        }
    }
}
julia_policy_t julia_policy__ = NULL;
julia_kernel_t julia_kernel__ = NULL;
int iris_julia_init(void *julia_launch_func, int decoupled_init)
{
    julia_kernel__ = (julia_kernel_t) julia_launch_func;
    //int32_t target = 12; 
    //int32_t devno=0;
    //int32_t result = julia_kernel__(target, devno);
    //printf("Result %d\n", result);
    return Platform::GetPlatform()->JuliaInit((bool)decoupled_init);
}
int iris_julia_policy_init(void *julia_policy_func) 
{
    julia_policy__ = (julia_policy_t) julia_policy_func;
    return IRIS_SUCCESS;
}
int iris_init_scheduler(int use_pthread)
{
    return Platform::GetPlatform()->InitScheduler((bool)use_pthread);
}
int iris_init_worker(int dev)
{
    return Platform::GetPlatform()->InitWorker(dev);
}
int iris_start_worker(int dev, int use_pthread)
{
    fprintf(stderr, "Calling startWorker\n");
    fflush(stderr);
    return Platform::GetPlatform()->StartWorker(dev, (bool)use_pthread);
}
int iris_init_device(int dev)
{
    return Platform::GetPlatform()->InitDevice(dev);
}
int iris_init_devices_synchronize(int sync)
{
    return Platform::GetPlatform()->InitDevicesSynchronize(sync);
}
int iris_init_devices(int sync)
{
    if (Platform::GetPlatform()->disable_init_devices())
        return Platform::GetPlatform()->InitDevices(sync);
    return IRIS_SUCCESS;
}
julia_kernel_t iris_get_julia_launch_func() 
{
    return julia_kernel__;
}
julia_policy_t iris_get_julia_policy_func() 
{
    return julia_policy__;
}
int iris_vendor_kernel_launch(int dev, void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params) 
{
    return Platform::GetPlatform()->VendorKernelLaunch(dev, kernel, gridx, gridy, gridz, blockx, blocky, blockz, shared_mem_bytes, stream, params);
}
int iris_is_enabled_auto_par() {
  return (int) Platform::GetPlatform()->GetAutoPar();
}

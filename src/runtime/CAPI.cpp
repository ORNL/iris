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

void iris_task_set_retain_flag(bool flag, iris_task task) {
  Platform *platform = Platform::GetPlatform(); 
  platform->set_release_task_flag(!flag, task);
}

void iris_set_release_task_flag(bool flag) {
  Platform *platform = Platform::GetPlatform(); 
  platform->set_release_task_flag(flag);
}

int iris_finalize() {
  Platform *platform = Platform::GetPlatform(); 
  int status = platform->Finalize();
  //delete platform;
  return status;
}

int iris_synchronize() {
  return Platform::GetPlatform()->Synchronize();
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

int iris_task_create(iris_task* task) {
  return Platform::GetPlatform()->TaskCreate(NULL, false, task);
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

int iris_task_get_metadata(iris_task brs_task, int index) {
    Task* task = brs_task->class_obj;
    return task->metadata(index);
}

int iris_task_set_metadata(iris_task brs_task, int index, int metadata) {
    Task* task = brs_task->class_obj;
    task->set_metadata(index, metadata);
    return IRIS_SUCCESS;
}

int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, size, host);
}

int iris_task_h2d_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
}

int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, size, host);
}

int iris_task_d2h_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, host_sizes, dev_sizes, elem_size, dim, host);
}

int iris_task_dmem_flush_out(iris_task task, iris_mem mem) {
  return Platform::GetPlatform()->TaskMemFlushOut(task, mem);
}


int iris_task_h2d_full(iris_task task, iris_mem mem, void* host) {
  return Platform::GetPlatform()->TaskH2DFull(task, mem, host);
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

int iris_params_map(iris_task task, int *params_map) {
  return Platform::GetPlatform()->SetParamsMap(task, params_map);
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

int iris_task_set_policy(iris_task task, int device) {
  return Platform::GetPlatform()->SetTaskPolicy(task, device);
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
   Task *task = brs_task->class_obj;
   task->set_name(name);
}

char *iris_kernel_get_name(iris_kernel brs_kernel) {
    Kernel *k= brs_kernel->class_obj;
    return k->name();
}

char *iris_task_get_name(iris_task brs_task) {
    Task *task = brs_task->class_obj;
    return task->name();
}

int iris_task_get_dependency_count(iris_task brs_task) {
    Task *task = brs_task->class_obj;
    return task->ndepends();
}
void iris_task_get_dependencies(iris_task brs_task, iris_task *tasks) {
    Task *task = brs_task->class_obj;
    for(int i=0; i<task->ndepends(); i++) {
        tasks[i] = task->depend(i)->struct_obj();
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
    return cmd->kernel_arg(index)->mem->struct_obj();
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
iris_kernel iris_task_get_kernel(iris_task brs_task)
{
    Task *task = brs_task->class_obj;
    return task->cmd_kernel()->kernel()->struct_obj();
}
int iris_task_is_cmd_kernel_exists(iris_task brs_task)
{
    Task *task = brs_task->class_obj;
    return task->cmd_kernel() != NULL;
}

void *iris_task_get_cmd_kernel(iris_task brs_task)
{
    Task *task = brs_task->class_obj;
    return task->cmd_kernel();
}

unsigned long iris_kernel_get_uid(iris_kernel brs_kernel)
{
  return brs_kernel->class_obj->uid();
}
unsigned long iris_task_get_uid(iris_task brs_task) {
  return brs_task->class_obj->uid();
}

int iris_mem_create(size_t size, iris_mem* mem) {
  return Platform::GetPlatform()->MemCreate(size, mem);
}
size_t iris_mem_get_size(iris_mem mem) {
  return mem->class_obj->size();
}

int iris_mem_get_type(iris_mem mem) {
  return mem->class_obj->GetMemHandlerType();
}

int iris_mem_get_uid(iris_mem mem) {
  return mem->class_obj->uid();
}

int iris_mem_is_reset(iris_mem mem) {
  return mem->class_obj->is_reset();
}

int iris_data_mem_init_reset(iris_mem mem, int reset) {
  return Platform::GetPlatform()->DataMemInit(mem, (bool)reset);
}
int iris_data_mem_create_tile(iris_mem* mem, void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
  return Platform::GetPlatform()->DataMemCreate(mem, host, off, host_size, dev_size, elem_size, dim);
}
int iris_data_mem_create(iris_mem *mem, void *host, size_t size) {
  return Platform::GetPlatform()->DataMemCreate(mem, host, size);
}
int iris_data_mem_update(iris_mem mem, void *host) {
  return Platform::GetPlatform()->DataMemUpdate(mem, host);
}
int iris_data_mem_create_region(iris_mem *mem, iris_mem root_mem, int region) {
  return Platform::GetPlatform()->DataMemCreate(mem, root_mem, region);
}
int iris_data_mem_enable_outer_dim_regions(iris_mem mem) {
  return Platform::GetPlatform()->DataMemEnableOuterDimRegions(mem);
}
iris_mem iris_get_dmem_for_region(iris_mem brs_mem) {
    DataMemRegion *obj = (DataMemRegion*)brs_mem->class_obj;
    return obj->get_dmem()->struct_obj();
}

int iris_mem_arch(iris_mem mem, int device, void** arch) {
  return Platform::GetPlatform()->MemArch(mem, device, arch);
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

int iris_graph_create_json(const char* json, void** params, iris_graph* graph) {
  return Platform::GetPlatform()->GraphCreateJSON(json, params, graph);
}

int iris_graph_task(iris_graph graph, iris_task task, int device, const char* opt) {
  return Platform::GetPlatform()->GraphTask(graph, task, device, opt);
}

int iris_graph_retain(iris_graph graph) {
  return Platform::GetPlatform()->GraphRetain(graph);
}

int iris_graph_release(iris_graph graph) {
  return Platform::GetPlatform()->GraphRelease(graph);
}

int iris_graph_submit(iris_graph graph, int device, int sync) {
  return Platform::GetPlatform()->GraphSubmit(graph, device, sync);
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

int iris_graph_get_tasks(iris_graph graph, iris_task *tasks) {
  return Platform::GetPlatform()->GetGraphTasks(graph, tasks);
}
int iris_graph_tasks_count(iris_graph graph)
{
    return Platform::GetPlatform()->GetGraphTasksCount(graph);
}
int iris_get_graph_2d_comm_adj_matrix(iris_graph brs_graph, size_t *size_data)
{
    Graph* graph = brs_graph->class_obj;
    GraphMetadata gm(graph);
    gm.get_2d_comm_adj_matrix(size_data);
    return IRIS_SUCCESS;
}
void iris_free_array(void *ptr)
{
    free(ptr);
}
int8_t *iris_allocate_array_int8_t(int SIZE, int8_t init)
{
    return Utils::AllocateArray<int8_t>(SIZE, init);
}
int16_t *iris_allocate_array_int16_t(int SIZE, int16_t init)
{
    return Utils::AllocateArray<int16_t>(SIZE, init);
}
int32_t *iris_allocate_array_int32_t(int SIZE, int32_t init)
{
    return Utils::AllocateArray<int32_t>(SIZE, init);
}
int64_t *iris_allocate_array_int64_t(int SIZE, int64_t init)
{
    return Utils::AllocateArray<int64_t>(SIZE, init);
}
size_t *iris_allocate_array_size_t(int SIZE, size_t init)
{
    return Utils::AllocateArray<size_t>(SIZE, init);
}
float *iris_allocate_array_float(int SIZE, float init)
{
    return Utils::AllocateArray<float>(SIZE, init);
}
double *iris_allocate_array_double(int SIZE, double init)
{
    return Utils::AllocateArray<double>(SIZE, init);
}
int8_t *iris_allocate_random_array_int8_t(int SIZE)
{
    return Utils::AllocateRandomArray<int8_t>(SIZE);
}
int16_t *iris_allocate_random_array_int16_t(int SIZE)
{
    return Utils::AllocateRandomArray<int16_t>(SIZE);
}
int32_t *iris_allocate_random_array_int32_t(int SIZE)
{
    return Utils::AllocateRandomArray<int32_t>(SIZE);
}
int64_t *iris_allocate_random_array_int64_t(int SIZE)
{
    return Utils::AllocateRandomArray<int64_t>(SIZE);
}
size_t *iris_allocate_random_array_size_t(int SIZE)
{
    return Utils::AllocateRandomArray<size_t>(SIZE);
}
float *iris_allocate_random_array_float(int SIZE)
{
    return Utils::AllocateRandomArray<float>(SIZE);
}
double *iris_allocate_random_array_double(int SIZE)
{
    return Utils::AllocateRandomArray<double>(SIZE);
}

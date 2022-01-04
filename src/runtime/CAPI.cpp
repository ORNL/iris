#include <iris/iris.h>
#include <iris/brisbane.h>
#include "Debug.h"
#include "Platform.h"

using namespace brisbane::rt;

int iris_init(int* argc, char*** argv, int sync) {
  return Platform::GetPlatform()->Init(argc, argv, sync);
}

int iris_finalize() {
  return Platform::GetPlatform()->Finalize();
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

int iris_task_depend(iris_task task, int ntasks, iris_task* tasks) {
  return Platform::GetPlatform()->TaskDepend(task, ntasks, tasks);
}

int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, size, host);
}

int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, size, host);
}

int iris_task_h2d_full(iris_task task, iris_mem mem, void* host) {
  return Platform::GetPlatform()->TaskH2DFull(task, mem, host);
}

int iris_task_d2h_full(iris_task task, iris_mem mem, void* host) {
  return Platform::GetPlatform()->TaskD2HFull(task, mem, host);
}

int iris_task_kernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, NULL, params_info, NULL);
}

int iris_task_submit(iris_task task, int device, const char* opt, int sync) {
  return Platform::GetPlatform()->TaskSubmit(task, device, opt, sync);
}

int iris_task_wait(iris_task task) {
  return Platform::GetPlatform()->TaskWait(task);
}

int iris_task_wait_all(int ntasks, iris_task* tasks) {
  return Platform::GetPlatform()->TaskWaitAll(ntasks, tasks);
}

int iris_task_release(iris_task task) {
  return Platform::GetPlatform()->TaskRelease(task);
}

int iris_mem_create(size_t size, iris_mem* mem) {
  return Platform::GetPlatform()->MemCreate(size, mem);
}

int iris_mem_release(iris_mem mem) {
  return Platform::GetPlatform()->MemRelease(mem);
}

int iris_timer_now(double* time) {
  return Platform::GetPlatform()->TimerNow(time);
}

int brisbane_init(int* argc, char*** argv, int sync) {
  return Platform::GetPlatform()->Init(argc, argv, sync);
}

int brisbane_finalize() {
  return Platform::GetPlatform()->Finalize();
}

int brisbane_synchronize() {
  return Platform::GetPlatform()->Synchronize();
}

int brisbane_env_set(const char* key, const char* value) {
  return Platform::GetPlatform()->EnvironmentSet(key, value, true);
}

int brisbane_env_get(const char* key, char** value, size_t* vallen) {
  return Platform::GetPlatform()->EnvironmentGet(key, value, vallen);
}

int brisbane_platform_count(int* nplatforms) {
  return Platform::GetPlatform()->PlatformCount(nplatforms);
}

int brisbane_platform_info(int platform, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->PlatformInfo(platform, param, value, size);
}

int brisbane_platform_build_program(int model, char* path) {
  return Platform::GetPlatform()->PlatformBuildProgram(model, path);
}

int brisbane_device_count(int* ndevs) {
  return Platform::GetPlatform()->DeviceCount(ndevs);
}

int brisbane_device_info(int device, int param, void* value, size_t* size) {
  return Platform::GetPlatform()->DeviceInfo(device, param, value, size);
}

int brisbane_device_set_default(int device) {
  return Platform::GetPlatform()->DeviceSetDefault(device);
}

int brisbane_device_get_default(int* device) {
  return Platform::GetPlatform()->DeviceGetDefault(device);
}

int brisbane_device_synchronize(int ndevs, int* devices) {
  return Platform::GetPlatform()->DeviceSynchronize(ndevs, devices);
}

int brisbane_register_policy(const char* lib, const char* name, void* params) {
  return Platform::GetPlatform()->PolicyRegister(lib, name, params);
}

int brisbane_register_command(int tag, int device, command_handler handler) {
  return Platform::GetPlatform()->RegisterCommand(tag, device, handler);
}

int brisbane_register_hooks_task(hook_task pre, hook_task post) {
  return Platform::GetPlatform()->RegisterHooksTask(pre, post);
}

int brisbane_register_hooks_command(hook_command pre, hook_command post) {
  return Platform::GetPlatform()->RegisterHooksCommand(pre, post);
}

int brisbane_kernel_create(const char* name, brisbane_kernel* kernel) {
  return Platform::GetPlatform()->KernelCreate(name, kernel);
}

int brisbane_kernel_get(const char* name, brisbane_kernel* kernel) {
  return Platform::GetPlatform()->KernelGet(name, kernel);
}

int brisbane_kernel_setarg(brisbane_kernel kernel, int idx, size_t size, void* value) {
  return Platform::GetPlatform()->KernelSetArg(kernel, idx, size, value);
}

int brisbane_kernel_setmem(brisbane_kernel kernel, int idx, brisbane_mem mem, size_t mode) {
  return Platform::GetPlatform()->KernelSetMem(kernel, idx, mem, 0, mode);
}

int brisbane_kernel_setmem_off(brisbane_kernel kernel, int idx, brisbane_mem mem, size_t off, size_t mode) {
  return Platform::GetPlatform()->KernelSetMem(kernel, idx, mem, off, mode);
}

int brisbane_kernel_setmap(brisbane_kernel kernel, int idx, void* host, size_t mode) {
  return Platform::GetPlatform()->KernelSetMap(kernel, idx, host, mode);
}

int brisbane_kernel_release(brisbane_kernel kernel) {
  return Platform::GetPlatform()->KernelRelease(kernel);
}

int brisbane_task_create(brisbane_task* task) {
  return Platform::GetPlatform()->TaskCreate(NULL, false, task);
}

int brisbane_task_create_perm(brisbane_task* task) {
  return Platform::GetPlatform()->TaskCreate(NULL, true, task);
}

int brisbane_task_create_name(const char* name, brisbane_task* task) {
  return Platform::GetPlatform()->TaskCreate(name, false, task);
}

int brisbane_task_depend(brisbane_task task, int ntasks, brisbane_task* tasks) {
  return Platform::GetPlatform()->TaskDepend(task, ntasks, tasks);
}

int brisbane_task_malloc(brisbane_task task, brisbane_mem mem) {
  return Platform::GetPlatform()->TaskMalloc(task, mem);
}

int brisbane_task_h2d(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskH2D(task, mem, off, size, host);
}

int brisbane_task_d2h(brisbane_task task, brisbane_mem mem, size_t off, size_t size, void* host) {
  return Platform::GetPlatform()->TaskD2H(task, mem, off, size, host);
}

int brisbane_task_h2d_full(brisbane_task task, brisbane_mem mem, void* host) {
  return Platform::GetPlatform()->TaskH2DFull(task, mem, host);
}

int brisbane_task_d2h_full(brisbane_task task, brisbane_mem mem, void* host) {
  return Platform::GetPlatform()->TaskD2HFull(task, mem, host);
}

int brisbane_task_map(brisbane_task task, void* host, size_t size) {
  return Platform::GetPlatform()->TaskMap(task, host, size);
}

int brisbane_task_mapto(brisbane_task task, void* host, size_t size) {
  return Platform::GetPlatform()->TaskMapTo(task, host, size);
}

int brisbane_task_mapto_full(brisbane_task task, void* host) {
  return Platform::GetPlatform()->TaskMapToFull(task, host);
}

int brisbane_task_mapfrom(brisbane_task task, void* host, size_t size) {
  return Platform::GetPlatform()->TaskMapFrom(task, host, size);
}

int brisbane_task_mapfrom_full(brisbane_task task, void* host) {
  return Platform::GetPlatform()->TaskMapFromFull(task, host);
}

int brisbane_task_kernel_obsolete(brisbane_task task, brisbane_kernel kernel, int dim, size_t* off, size_t* gws, size_t* lws) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws);
}

int brisbane_task_kernel(brisbane_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, NULL, params_info, NULL);
}

int brisbane_task_kernel_v2(brisbane_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, NULL);
}

int brisbane_task_kernel_v3(brisbane_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges) {
  return Platform::GetPlatform()->TaskKernel(task, kernel, dim, off, gws, lws, nparams, params, params_off, params_info, memranges);
}

int brisbane_task_kernel_selector(brisbane_task task, brisbane_selector_kernel func, void* params, size_t params_size) {
  return Platform::GetPlatform()->TaskKernelSelector(task, func, params, params_size);
}

int brisbane_task_host(brisbane_task task, brisbane_host_task func, void* params) {
  return Platform::GetPlatform()->TaskHost(task, func, params);
}

int brisbane_task_custom(brisbane_task task, int tag, void* params, size_t params_size) {
  return Platform::GetPlatform()->TaskCustom(task, tag, params, params_size);
}

int brisbane_task_submit(brisbane_task task, int device, const char* opt, int sync) {
  return Platform::GetPlatform()->TaskSubmit(task, device, opt, sync);
}

int brisbane_task_wait(brisbane_task task) {
  return Platform::GetPlatform()->TaskWait(task);
}

int brisbane_task_wait_all(int ntasks, brisbane_task* tasks) {
  return Platform::GetPlatform()->TaskWaitAll(ntasks, tasks);
}

int brisbane_task_add_subtask(brisbane_task task, brisbane_task subtask) {
  return Platform::GetPlatform()->TaskAddSubtask(task, subtask);
}

int brisbane_task_kernel_cmd_only(brisbane_task task) {
  return Platform::GetPlatform()->TaskKernelCmdOnly(task);
}

int brisbane_task_release(brisbane_task task) {
  return Platform::GetPlatform()->TaskRelease(task);
}

int brisbane_task_release_mem(brisbane_task task, brisbane_mem mem) {
  return Platform::GetPlatform()->TaskReleaseMem(task, mem);
}

int brisbane_mem_create(size_t size, brisbane_mem* mem) {
  return Platform::GetPlatform()->MemCreate(size, mem);
}

int brisbane_mem_arch(brisbane_mem mem, int device, void** arch) {
  return Platform::GetPlatform()->MemArch(mem, device, arch);
}

int brisbane_mem_map(void* host, size_t size) {
  return Platform::GetPlatform()->MemMap(host, size);
}

int brisbane_mem_unmap(void* host) {
  return Platform::GetPlatform()->MemUnmap(host);
}

int brisbane_mem_reduce(brisbane_mem mem, int mode, int type) {
  return Platform::GetPlatform()->MemReduce(mem, mode, type);
}

int brisbane_mem_release(brisbane_mem mem) {
  return Platform::GetPlatform()->MemRelease(mem);
}

int brisbane_graph_create(brisbane_graph* graph) {
  return Platform::GetPlatform()->GraphCreate(graph);
}

int brisbane_graph_create_json(const char* json, void** params, brisbane_graph* graph) {
  return Platform::GetPlatform()->GraphCreateJSON(json, params, graph);
}

int brisbane_graph_task(brisbane_graph graph, brisbane_task task, int device, const char* opt) {
  return Platform::GetPlatform()->GraphTask(graph, task, device, opt);
}

int brisbane_graph_submit(brisbane_graph graph, int device, int sync) {
  return Platform::GetPlatform()->GraphSubmit(graph, device, sync);
}

int brisbane_graph_wait(brisbane_graph graph) {
  return Platform::GetPlatform()->GraphWait(graph);
}

int brisbane_graph_wait_all(int ngraphs, brisbane_graph* graphs) {
  return Platform::GetPlatform()->GraphWaitAll(ngraphs, graphs);
}

int brisbane_record_start() {
  return Platform::GetPlatform()->RecordStart();
}

int brisbane_record_stop() {
  return Platform::GetPlatform()->RecordStop();
}

int brisbane_timer_now(double* time) {
  return Platform::GetPlatform()->TimerNow(time);
}


#include <iris/iris.h>
#include "Debug.h"
#include "Platform.h"

using namespace iris::rt;

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

int iris_task_custom(iris_task task, int tag, void* params, size_t params_size) {
  return Platform::GetPlatform()->TaskCustom(task, tag, params, params_size);
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

int iris_mem_create(size_t size, iris_mem* mem) {
  return Platform::GetPlatform()->MemCreate(size, mem);
}

int iris_mem_intermediate(iris_mem mem, int flag) {
  return Platform::GetPlatform()->MemSetIntermediate(mem, (bool)flag);
}

int iris_mem_release(iris_mem mem) {
  return Platform::GetPlatform()->MemRelease(mem);
}

int iris_timer_now(double* time) {
  return Platform::GetPlatform()->TimerNow(time);
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



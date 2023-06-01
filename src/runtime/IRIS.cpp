#include <iris/iris.hpp>
#include "Debug.h"
#include "Structs.h"
#include "Platform.h"
#include "Timer.h"
#include "Task.h"
#include <assert.h>

using namespace iris;
typedef iris::rt::Platform PlatformIRIS;
typedef iris::rt::Task TaskIRIS;
typedef iris::rt::BaseMem BaseMemIRIS;
typedef iris::rt::Mem MemIRIS;
Mem::Mem(size_t size) {
#ifdef ENABLE_SMART_PTR_MEM
    assert(PlatformIRIS::GetPlatform()->MemCreate(size, &mem_) == IRIS_SUCCESS);
#else
    iris_mem_create(size, &mem_);
#endif
}
Mem::~Mem() {
#ifndef ENABLE_SMART_PTR_MEM
    iris_mem_release(mem_);
#endif
}
DMem::DMem(void *host, size_t size) {
#ifdef ENABLE_SMART_PTR_MEM
    assert(PlatformIRIS::GetPlatform()->DataMemCreate(&mem_, host, size) == IRIS_SUCCESS);
#else
    assert(iris_data_mem_create(&mem_, host, size) == IRIS_SUCCESS);
#endif
}
DMem::DMem(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
#ifdef ENABLE_SMART_PTR_MEM
    assert(PlatformIRIS::GetPlatform()->DataMemCreate(&mem_, host, off, host_size, dev_size, elem_size, dim) == IRIS_SUCCESS);
#else
    assert(iris_data_mem_create_tile(&mem_, host, off, host_size, dev_size, elem_size, dim) == IRIS_SUCCESS);
#endif
}
DMem::~DMem() {
#ifndef ENABLE_SMART_PTR_MEM
    iris_mem_release(mem_);
#endif
}
int DMem::update(void *host) {
#ifdef ENABLE_SMART_PTR_MEM
    return PlatformIRIS::GetPlatform()->DataMemUpdate(mem_, host);
#else
    return iris_data_mem_update(mem_, host);
#endif
}
int DMem::reset(bool reset) { 
#ifdef ENABLE_SMART_PTR_MEM
    return PlatformIRIS::GetPlatform()->DataMemInit(mem_, (bool)reset);
#else
    return iris_data_mem_init_reset(mem_, reset); 
#endif
}
int DMem::enable_outer_dim_regions() {
#ifdef ENABLE_SMART_PTR_MEM
    return PlatformIRIS::GetPlatform()->DataMemEnableOuterDimRegions(mem_);
#else
    return iris_data_mem_enable_outer_dim_regions(mem_);
#endif
}
DMemRegion::DMemRegion(iris_mem_type root_mem, int region) {
#ifdef ENABLE_SMART_PTR_MEM
    assert(PlatformIRIS::GetPlatform()->DataMemCreate(&mem_, root_mem, region) == IRIS_SUCCESS);
#else
    iris_data_mem_create_region(&mem_, root_mem, region);
#endif
}
DMemRegion::DMemRegion(DMem *root_mem, int region) {
    dmem_ = root_mem;
#ifdef ENABLE_SMART_PTR_MEM
    assert(PlatformIRIS::GetPlatform()->DataMemCreate(&mem_, root_mem->mem(), region) == IRIS_SUCCESS);
#else
    assert(iris_data_mem_create_region(&mem_, root_mem->mem(), region) == IRIS_SUCCESS);
#endif
}
Task::Task(const char *name, bool perm, bool retainable) 
{
    retainable_ = retainable;
#ifdef ENABLE_SMART_PTR_TASK
    assert(PlatformIRIS::GetPlatform()->TaskCreate(name, perm, &task_) == IRIS_SUCCESS);
    if (retainable) platform->set_release_task_flag(!retainable, task_);
#else
    assert(PlatformIRIS::GetPlatform()->TaskCreate(name, perm, &task_) == IRIS_SUCCESS);
    if (retainable) iris_task_retain(task_, !retainable);
#endif
}
int Task::set_order(int *order) {
    return iris_task_kernel_dmem_fetch_order(task_, order);
}
int Task::h2d(Mem* mem, size_t off, size_t size, void* host) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskH2D(task_, mem->mem(), off, size, host);
#else
    return iris_task_h2d(task_, mem->mem(), off, size, host);
#endif
}

int Task::h2broadcast(Mem* mem, size_t off, size_t size, void* host) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskH2Broadcast(task_, mem->mem(), off, size, host);
#else
    return iris_task_h2broadcast(task_, mem->mem(), off, size, host);
#endif
}

int Task::h2broadcast_full(Mem* mem, void* host) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskH2BroadcastFull(task_, mem->mem(), host);
#else
    return iris_task_h2broadcast_full(task_, mem->mem(), host);
#endif
}

int Task::h2d_full(Mem* mem, void* host) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskH2DFull(task_, mem->mem(), host);
#else
    return iris_task_h2d_full(task_, mem->mem(), host);
#endif
}

int Task::d2h(Mem* mem, size_t off, size_t size, void* host) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskD2H(task_, mem->mem(), off, size, host);
#else
    return iris_task_d2h(task_, mem->mem(), off, size, host);
#endif
}

int Task::d2h_full(Mem* mem, void* host) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskD2HFull(task_, mem->mem(), host);
#else
    return iris_task_d2h_full(task_, mem->mem(), host);
#endif
}

int Task::flush_out(DMem & mem) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskMemFlushOut(task_, mem.mem());
#else
    return iris_task_dmem_flush_out(task_, mem.mem());
#endif
}

int Task::kernel(const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
    void** new_params = new void*[nparams];
    for (int i = 0; i < nparams; i++) {
        if (params_info[i] == iris_w ||
                params_info[i] == iris_r ||
                params_info[i] == iris_rw) {
#ifdef ENABLE_SMART_PTR_MEM
            iris_mem_type dpmem = ((iris::BaseMem*) params[i])->mem();
            new_params[i] = (BaseMemIRIS*)(dpmem.get());
#else
            new_params[i] = ((BaseMem*) params[i])->mem_ptr();
#endif
        } else new_params[i] = params[i];
    }
#ifdef ENABLE_SMART_PTR_TASK
    int ret = PlatformIRIS::GetPlatform()->TaskKernel(task_, kernel, dim, off, gws, lws, nparams, new_params, NULL, params_info, NULL);
#else
    int ret = iris_task_kernel(task_, kernel, dim, off, gws, lws, nparams, new_params, params_info);
#endif
    delete[] new_params;
    return ret;
}
int Task::submit(int device, const char* opt, bool sync) {
#ifdef ENABLE_SMART_PTR_TASK
    return PlatformIRIS::GetPlatform()->TaskSubmit(task_, device, opt, sync ? 1 : 0);
#else
    return iris_task_submit(task_, device, opt, sync ? 1 : 0);
#endif
}
void Task::depends_on(int ntasks, Task **tasks) {
#ifdef ENABLE_SMART_PTR_TASK
    _error("Task::Depend is not supported for smart pointer\n");
#else
    TaskIRIS *c_task = (TaskIRIS *)(PlatformIRIS::GetPlatform()->get_task_object(task()));
    for (int i=0; i<ntasks; i++) {
        if (tasks[i] != NULL)  {
            TaskIRIS *i_task = (TaskIRIS*)(PlatformIRIS::GetPlatform()->get_task_object(tasks[i]->task()));
            c_task->AddDepend(i_task, tasks[i]->task().uid);
        }
    }
#endif
}
void Task::depends_on(vector<Task *> tasks) {
#ifdef ENABLE_SMART_PTR_TASK
    _error("Task::Depend is not supported for smart pointer\n");
#else
    TaskIRIS *c_task = (TaskIRIS *)(PlatformIRIS::GetPlatform()->get_task_object(task()));
    for (Task *d_task : tasks) {
        TaskIRIS *i_task = (TaskIRIS*)(PlatformIRIS::GetPlatform()->get_task_object(d_task->task()));
        c_task->AddDepend(i_task, d_task->task().uid);
    }
#endif
}
void Task::depends_on(Task & d_task) {
    TaskIRIS *c_task = (TaskIRIS *)(PlatformIRIS::GetPlatform()->get_task_object(task()));
    TaskIRIS *i_task = (TaskIRIS*)(PlatformIRIS::GetPlatform()->get_task_object(d_task.task()));
    c_task->AddDepend(i_task, d_task.task().uid);
}
Graph::Graph(bool retainable) {
    retainable_ = retainable;
    iris_graph_create(&graph_);
    if (retainable) iris_graph_retain(graph_, retainable);
    is_released_ = false;
}
Graph::~Graph() {
    if (!is_released_)
        iris_graph_release(graph_);
}
void Graph::retainable() {
    retainable_ = true;
    iris_graph_retain(graph_, true);
}
int Graph::add_task(Task & task, int device, const char *opt) {
#ifdef ENABLE_SMART_PTR_TASK
    _error("Graph::AddTask is not supported for smart pointer\n");
    return IRIS_ERROR;
#else
    return iris_graph_task(graph_, task.task(), device, opt);
#endif
}
int Graph::release() {
    is_released_ = true;
    return iris_graph_release(graph_);
}
int Graph::set_order(int *order) {
    return iris_graph_tasks_order(graph_, order);
}
int Graph::submit(int device, int sync) {
    return iris_graph_submit(graph_, device, sync);
}
int Graph::wait() {
    return iris_graph_wait(graph_);
}

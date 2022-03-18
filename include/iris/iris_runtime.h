#ifndef IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H
#define IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H

#include <iris/iris_errno.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRIS_MAX_NPLATFORMS     32
#define IRIS_MAX_NDEVS          (1 << 5) - 1

#define iris_default            (1 << 5)
#define iris_cpu                (1 << 6)
#define iris_nvidia             (1 << 7)
#define iris_amd                (1 << 8)
#define iris_gpu_intel          (1 << 9)
#define iris_gpu                (iris_nvidia | iris_amd | iris_gpu_intel)
#define iris_phi                (1 << 10)
#define iris_fpga               (1 << 11)
#define iris_hexagon            (1 << 12)
#define iris_dsp                (iris_hexagon)
#define iris_roundrobin         (1 << 18)
#define iris_depend             (1 << 19)
#define iris_data               (1 << 20)
#define iris_profile            (1 << 21)
#define iris_random             (1 << 22)
#define iris_pending            (1 << 23)
#define iris_any                (1 << 24)
#define iris_all                (1 << 25)
#define iris_custom             (1 << 26)

#define iris_cuda               1
//#define iris_hexagon            2
#define iris_hip                3
#define iris_levelzero          4
#define iris_opencl             5
#define iris_openmp             6

#define iris_r                  -1
#define iris_w                  -2
#define iris_rw                 -3

#define iris_int                (1 << 0)
#define iris_long               (1 << 1)
#define iris_float              (1 << 2)
#define iris_double             (1 << 3)

#define iris_normal             (1 << 10)
#define iris_reduction          (1 << 11)
#define iris_sum                ((1 << 12) | iris_reduction)
#define iris_max                ((1 << 13) | iris_reduction)
#define iris_min                ((1 << 14) | iris_reduction)

#define iris_platform           0x1001
#define iris_vendor             0x1002
#define iris_name               0x1003
#define iris_type               0x1004

#define iris_ncmds              1
#define iris_ncmds_kernel       2
#define iris_ncmds_memcpy       3
#define iris_cmds               4

typedef struct _iris_task*  iris_task;
typedef struct _iris_mem*   iris_mem;
typedef struct _iris_kernel*    iris_kernel;
typedef struct _iris_graph*     iris_graph;

typedef int (*iris_host_task)(void* params, const int* device);
typedef int (*command_handler)(void* params, void* device);
typedef int (*hook_task)(void* task);
typedef int (*hook_command)(void* command);

typedef int (*iris_selector_kernel)(iris_task task, void* params, char* kernel_name);

extern int iris_init(int* argc, char*** argv, int sync);
extern int iris_finalize();
extern int iris_synchronize();

extern int iris_env_set(const char* key, const char* value);
extern int iris_env_get(const char* key, char** value, size_t* vallen);

extern int iris_platform_count(int* nplatforms);
extern int iris_platform_info(int platform, int param, void* value, size_t* size);
extern int iris_set_shared_memory_model(int flag);

extern int iris_device_count(int* ndevs);
extern int iris_device_info(int device, int param, void* value, size_t* size);
extern int iris_device_set_default(int device);
extern int iris_device_get_default(int* device);
extern int iris_device_synchronize(int ndevs, int* devices);

extern int iris_register_policy(const char* lib, const char* name, void* params);
extern int iris_register_command(int tag, int device, command_handler handler);
extern int iris_register_hooks_task(hook_task pre, hook_task post);
extern int iris_register_hooks_command(hook_command pre, hook_command post);

extern int iris_kernel_create(const char* name, iris_kernel* kernel);
extern int iris_kernel_get(const char* name, iris_kernel* kernel);
extern int iris_kernel_setarg(iris_kernel kernel, int idx, size_t size, void* value);
extern int iris_kernel_setmem(iris_kernel kernel, int idx, iris_mem mem, size_t mode);
extern int iris_kernel_setmem_off(iris_kernel kernel, int idx, iris_mem mem, size_t off, size_t mode);
extern int iris_kernel_setmap(iris_kernel kernel, int idx, void* host, size_t mode);
extern int iris_kernel_release(iris_kernel kernel);

extern int iris_task_create(iris_task* task);
extern int iris_task_create_perm(iris_task* task);
extern int iris_task_create_name(const char* name, iris_task* task);
extern int iris_task_depend(iris_task task, int ntasks, iris_task* tasks);
extern int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_h2d_full(iris_task task, iris_mem mem, void* host);
extern int iris_task_d2h_full(iris_task task, iris_mem mem, void* host);
extern int iris_task_kernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info);
extern int iris_task_kernel_v2(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info);
extern int iris_task_kernel_v3(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges);
extern int iris_task_host(iris_task task, iris_host_task func, void* params);
extern int iris_task_host(iris_task task, iris_host_task func, void* params);
extern int iris_task_custom(iris_task task, int tag, void* params, size_t params_size);
extern int iris_task_submit(iris_task task, int device, const char* opt, int sync);
extern int iris_task_wait(iris_task task);
extern int iris_task_wait_all(int ntasks, iris_task* tasks);
extern int iris_task_add_subtask(iris_task task, iris_task subtask);
extern int iris_task_kernel_cmd_only(iris_task task);
extern int iris_task_release(iris_task task);
extern int iris_task_release_mem(iris_task task, iris_mem mem);
extern int iris_params_map(iris_task task, int *params_map);
extern int iris_task_info(iris_task task, int param, void* value, size_t* size);

extern int iris_mem_create(size_t size, iris_mem* mem);
extern int iris_mem_arch(iris_mem mem, int device, void** arch);
extern int iris_mem_intermediate(iris_mem mem, int flag);
extern int iris_mem_reduce(iris_mem mem, int mode, int type);
extern int iris_mem_release(iris_mem mem);

extern int iris_timer_now(double* time);

extern int iris_graph_create(iris_graph* graph);
extern int iris_graph_create_json(const char* json, void** params, iris_graph* graph);
extern int iris_graph_task(iris_graph graph, iris_task task, int device, const char* opt);
extern int iris_graph_submit(iris_graph graph, int device, int sync);
extern int iris_graph_wait(iris_graph graph);
extern int iris_graph_wait_all(int ngraphs, iris_graph* graphs);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H */


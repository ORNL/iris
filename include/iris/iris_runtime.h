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
#define iris_any                (1 << 23)
#define iris_all                (1 << 24)
#define iris_custom             (1 << 25)

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

typedef struct _brisbane_task*  iris_task;
typedef struct _brisbane_mem*   iris_mem;
typedef struct _brisbane_kernel*    iris_kernel;
typedef struct _brisbane_graph*     iris_graph;

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

extern int iris_task_create(iris_task* task);
extern int iris_task_depend(iris_task task, int ntasks, iris_task* tasks);
extern int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_h2d_full(iris_task task, iris_mem mem, void* host);
extern int iris_task_d2h_full(iris_task task, iris_mem mem, void* host);
extern int iris_task_kernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info);
extern int iris_task_kernel_v2(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info);
extern int iris_task_custom(iris_task task, int tag, void* params, size_t params_size);
extern int iris_task_submit(iris_task task, int device, const char* opt, int sync);
extern int iris_task_wait(iris_task task);
extern int iris_task_wait_all(int ntasks, iris_task* tasks);
extern int iris_task_kernel_cmd_only(iris_task task);
extern int iris_task_release(iris_task task);
extern int iris_task_info(iris_task task, int param, void* value, size_t* size);

extern int iris_mem_create(size_t size, iris_mem* mem);
extern int iris_mem_release(iris_mem mem);

extern int iris_timer_now(double* time);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H */


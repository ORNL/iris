#ifndef IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H
#define IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H

#include <iris/iris_errno.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
typedef int8_t bool;
#endif

#ifndef UNDEF_IRIS_MACROS
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
#define iris_xr                 -4
#define iris_xw                 -5
#define iris_xrw                -6

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

#endif // UNDEF_IRIS_MACROS

typedef struct _iris_task*  iris_task;
typedef struct _iris_mem*   iris_mem;
typedef struct _iris_kernel*    iris_kernel;
typedef struct _iris_graph*     iris_graph;

typedef int (*iris_host_task)(void* params, const int* device);
typedef int (*command_handler)(void* params, void* device);
typedef int (*hook_task)(void* task);
typedef int (*hook_command)(void* command);

typedef int (*iris_selector_kernel)(iris_task task, void* params, char* kernel_name);

/**
 * Initializes the IRIS execution environment.
 * @param argc pointer to the number of arguments
 * @param argv argument array
 * @param sync 0: non-blocking, 1: blocking
 * @return All IRIS functions return an error value. IRIS_SUCCESS, IRIS_ERR
 */
extern int iris_init(int* argc, char*** argv, int sync);

/**
  * Return number of errors occurred in IRIS
  */
extern int iris_error_count();

/**
 * Terminates the IRIS execution environment.
 */
extern int iris_finalize();

/**
 * Waits for all the submitted tasks to complete.
 */
extern int iris_synchronize();

/**
  * If task need to be submitted again and again.
  */
extern void iris_set_release_task_flag(bool flag);
extern void iris_task_set_retain_flag(bool flag, iris_task);

/**
 * Sets an IRIS environment variable.
 * @param key key string
 * @param value value to be stored into key
 */
extern int iris_env_set(const char* key, const char* value);

/**
 * Gets an IRIS environment variable.
 * @param key key string
 * @param value pointer to the value to be retrieved
 * @param vallen size in bytes of value
 */
extern int iris_env_get(const char* key, char** value, size_t* vallen);

/**
 * Gets the number of errors encountered during the IRIS environments history.
 */
extern int iris_error_count();

/**
 * Returns the number of platforms.
 * @param nplatforms pointer to the number of platform
 */
extern int iris_platform_count(int* nplatforms);

/**
 * Returns the platform information.
 * @param platform platform number
 * @param param information type
 * @param value information value
 * @param size size in bytes of value
 */
extern int iris_platform_info(int platform, int param, void* value, size_t* size);
extern int iris_set_shared_memory_model(int flag);

/**
 * Returns the number of devices.
 * @param ndevs pointer to the number of devices
 */
extern int iris_device_count(int* ndevs);

/**
 * Returns the device information.
 * @param device device number
 * @param param information type
 * @param value information value
 * @param size size in bytes of value
 */
extern int iris_device_info(int device, int param, void* value, size_t* size);
extern int iris_device_set_default(int device);
extern int iris_device_get_default(int* device);

/**
 * Waits for all the submitted tasks in a device to complete.
 * @param ndevs number of devices
 * @param devices device array
 */
extern int iris_device_synchronize(int ndevs, int* devices);

/**
 * Registers a new device selector
 * @param lib shared library path
 * @param name selector name
 * @param params parameter to the selector init function
 */
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

/**
 * Creates a new task.
 * @param task pointer of the new task
 */
extern int iris_task_create(iris_task* task);
extern int iris_task_create_perm(iris_task* task);
extern int iris_task_create_name(const char* name, iris_task* task);

/**
 * Adds a dependency to a task.
 * @param task source task
 * @param ntasks number of tasks
 * @param tasks target tasks array
 */
extern int iris_task_depend(iris_task task, int ntasks, iris_task* tasks);
extern int iris_task_malloc(iris_task task, iris_mem mem);
extern int iris_task_cmd_reset_mem(iris_task task, iris_mem mem, uint8_t reset);
extern int iris_task_get_metadata(iris_task brs_task, int index);
extern int iris_task_set_metadata(iris_task brs_task, int index, int metadata);

/**
 * Adds a H2D command to the target task.
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host source host address
 */
extern int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_h2d_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);

/**
 * Adds a D2H command to the target task.
 * @param task target task
 * @param mem source memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host target host address
 */
extern int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_d2h_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);
extern int iris_task_dmem_flush_out(iris_task task, iris_mem mem);

/**
 * Adds a H2D command with the size of the target memory to the target task.
 * @param task target task
 * @param mem target memory object
 * @param host source host address
 */
extern int iris_task_h2d_full(iris_task task, iris_mem mem, void* host);

/**
 * Adds a D2H command with the size of the source memory to the target task.
 * @param task target task
 * @param mem source memory object
 * @param host target host address
 */
extern int iris_task_d2h_full(iris_task task, iris_mem mem, void* host);
extern int iris_task_kernel_object(iris_task task, iris_kernel kernel, int dim, size_t* off, size_t* gws, size_t* lws);

/**
 * Launch a kernel
 * @param task target task
 * @param kernel kernel name
 * @param dim dimension
 * @param off global workitem space offsets
 * @param gws global workitem space
 * @param lws local workitem space
 * @param nparams number of kernel parameters
 * @param params kernel parameters
 * @param params_info kernel parameters information
 */
extern int iris_task_kernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info);
extern int iris_task_kernel_v2(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info);
extern int iris_task_kernel_v3(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges);
extern int iris_task_kernel_selector(iris_task task, iris_selector_kernel func, void* params, size_t params_size);
extern int iris_task_host(iris_task task, iris_host_task func, void* params);
extern int iris_task_host(iris_task task, iris_host_task func, void* params);
extern int iris_task_custom(iris_task task, int tag, void* params, size_t params_size);

/**
 * Submits a task.
 * @param task target task
 * @param device device_selector
 * @param opt option string
 * @param sync 0: non-blocking, 1: blocking
 */
extern int iris_task_submit(iris_task task, int device, const char* opt, int sync);
extern int iris_task_set_policy(iris_task task, int device);

/**
 * Waits for the task to complete.
 * @param task target task
 */
extern int iris_task_wait(iris_task task);

/**
 * Waits for all the tasks to complete.
 * @param ntasks number of tasks
 * @param tasks target tasks array
 */
extern int iris_task_wait_all(int ntasks, iris_task* tasks);
extern int iris_task_add_subtask(iris_task task, iris_task subtask);
extern int iris_task_kernel_cmd_only(iris_task task);

/**
 * Releases a target.
 * @param task target task
 */
extern int iris_task_release(iris_task task);
extern int iris_task_release_mem(iris_task task, iris_mem mem);
extern int iris_params_map(iris_task task, int *params_map);
extern int iris_task_info(iris_task task, int param, void* value, size_t* size);

extern int iris_mem_create(size_t size, iris_mem* mem);
extern int iris_data_mem_init_reset(iris_mem mem, int reset);
extern int iris_data_mem_create(iris_mem* mem, void *host, size_t size);
extern int iris_data_mem_update(iris_mem mem, void *host);
extern int iris_data_mem_create_region(iris_mem* mem, iris_mem root_mem, int region);
extern int iris_data_mem_enable_outer_dim_regions(iris_mem mem);
extern int iris_data_mem_create_tile(iris_mem* mem, void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
extern int iris_mem_arch(iris_mem mem, int device, void** arch);
extern int iris_mem_reduce(iris_mem mem, int mode, int type);
extern int iris_mem_release(iris_mem mem);

extern int iris_graph_create(iris_graph* graph);
extern int iris_graph_free(iris_graph graph);
extern int iris_graph_create_json(const char* json, void** params, iris_graph* graph);
extern int iris_graph_task(iris_graph graph, iris_task task, int device, const char* opt);
extern int iris_graph_retain(iris_graph graph);
extern int iris_graph_release(iris_graph graph);
extern int iris_graph_submit(iris_graph graph, int device, int sync);
extern int iris_graph_wait(iris_graph graph);
extern int iris_graph_wait_all(int ngraphs, iris_graph* graphs);

extern int iris_record_start();
extern int iris_record_stop();

/**
 * Returns current time in seconds.
 * @param time pointer of time
 */
extern int iris_timer_now(double* time);

// Enable/Disable methods
extern void iris_enable_d2d();
extern void iris_disable_d2d();
extern void iris_disable_consistency_check();
extern void iris_enable_consistency_check();

// Task internal members access
extern char *iris_kernel_get_name(iris_kernel brs_kernel);
extern char *iris_task_get_name(iris_task brs_task);
extern void iris_task_set_name(iris_task brs_task, const char *name);
extern int iris_task_get_dependency_count(iris_task brs_task);
extern void iris_task_get_dependencies(iris_task brs_task, iris_task *tasks);
extern unsigned long iris_task_get_uid(iris_task brs_task);
extern unsigned long iris_kernel_get_uid(iris_kernel brs_kernel);
extern iris_kernel iris_task_get_kernel(iris_task brs_task);
extern int iris_task_is_cmd_kernel_exists(iris_task brs_task);
extern void *iris_task_get_cmd_kernel(iris_task brs_task);

// Memory member access
extern size_t iris_mem_get_size(iris_mem mem);
extern int iris_mem_get_type(iris_mem mem);
extern int iris_mem_get_uid(iris_mem mem);
extern int iris_mem_is_reset(iris_mem mem);
extern iris_mem iris_get_dmem_for_region(iris_mem dmem_region_obj);

// Command kernel member access
extern int iris_cmd_kernel_get_nargs(void *cmd);
extern int    iris_cmd_kernel_get_arg_is_mem(void *cmd, int index);
extern size_t iris_cmd_kernel_get_arg_size(void *cmd, int index);
extern void  *iris_cmd_kernel_get_arg_value(void *cmd, int index);
extern iris_mem iris_cmd_kernel_get_arg_mem(void *cmd, int index);
extern size_t iris_cmd_kernel_get_arg_mem_off(void *cmd, int index);
extern size_t iris_cmd_kernel_get_arg_mem_size(void *cmd, int index);
extern size_t iris_cmd_kernel_get_arg_off(void *cmd, int index);
extern int    iris_cmd_kernel_get_arg_mode(void *cmd, int index);

// Graph data 
extern int iris_graph_get_tasks(iris_graph graph, iris_task *tasks);
extern int iris_graph_tasks_count(iris_graph graph);
extern int iris_graph_submit_with_time(iris_graph graph, double *time, int device, int sync);
extern int iris_get_graph_dependency_adj_list(iris_graph brs_graph, int8_t *dep_matrix);
extern int iris_get_graph_dependency_adj_matrix(iris_graph brs_graph, int8_t *dep_matrix);
extern int iris_get_graph_2d_comm_adj_matrix(iris_graph brs_graph, size_t *size_data);
extern int iris_calibrate_compute_cost_adj_matrix(iris_graph brs_graph, double *comp_data);
extern void iris_free_array(void *ptr);
extern int8_t *iris_allocate_array_int8_t(int SIZE, int8_t init);
extern int16_t *iris_allocate_array_int16_t(int SIZE, int16_t init);
extern int32_t *iris_allocate_array_int32_t(int SIZE, int32_t init);
extern int64_t *iris_allocate_array_int64_t(int SIZE, int64_t init);
extern size_t *iris_allocate_array_size_t(int SIZE, size_t init);
extern float *iris_allocate_array_float(int SIZE, float init);
extern double *iris_allocate_array_double(int SIZE, double init);
extern int8_t *iris_allocate_random_array_int8_t(int SIZE);
extern int16_t *iris_allocate_random_array_int16_t(int SIZE);
extern int32_t *iris_allocate_random_array_int32_t(int SIZE);
extern int64_t *iris_allocate_random_array_int64_t(int SIZE);
extern size_t *iris_allocate_random_array_size_t(int SIZE);
extern float *iris_allocate_random_array_float(int SIZE);
extern double *iris_allocate_random_array_double(int SIZE);
extern void iris_print_matrix_full_double(double *data, int M, int N, const char *description);
extern void iris_print_matrix_limited_double(double *data, int M, int N, const char *description, int limit);
extern void iris_print_matrix_full_float(float *data, int M, int N, const char *description);
extern void iris_print_matrix_limited_float(float *data, int M, int N, const char *description, int limit);
extern void iris_print_matrix_full_int64_t(int64_t *data, int M, int N, const char *description);
extern void iris_print_matrix_limited_int64_t(int64_t *data, int M, int N, const char *description, int limit);
extern void iris_print_matrix_full_int32_t(int32_t *data, int M, int N, const char *description);
extern void iris_print_matrix_limited_int32_t(int32_t *data, int M, int N, const char *description, int limit);
extern void iris_print_matrix_full_int16_t(int16_t *data, int M, int N, const char *description);
extern void iris_print_matrix_limited_int16_t(int16_t *data, int M, int N, const char *description, int limit);
extern void iris_print_matrix_full_int8_t(int8_t *data, int M, int N, const char *description);
extern void iris_print_matrix_limited_int8_t(int8_t *data, int M, int N, const char *description, int limit);
#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H */


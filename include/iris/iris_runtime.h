#ifndef IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H
#define IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H

#include <iris/iris_errno.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
namespace iris {
namespace rt {
class Kernel;
class BaseMem;
class Mem;
class Task;
class Graph;
} /* namespace rt */
} /* namespace iris */
#endif

struct _iris_task {
#ifdef __cplusplus
  iris::rt::Task* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

struct _iris_kernel {
#ifdef __cplusplus
  iris::rt::Kernel* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

struct _iris_mem {
#ifdef __cplusplus
    iris::rt::BaseMem* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

struct _iris_graph {
#ifdef __cplusplus
    iris::rt::Graph* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};
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
#define iris_ocl                (1 << 26)
#define iris_custom             (1 << 27)

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

#define iris_dt_h2d             1
#define iris_dt_d2o             2
#define iris_dt_o2d             3
#define iris_dt_d2h             4
#define iris_dt_d2d             5
#define iris_dt_d2h_h2d         6
#define iris_dt_error           0

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
#define iris_backend            0x1005

#define iris_ncmds              1
#define iris_ncmds_kernel       2
#define iris_ncmds_memcpy       3
#define iris_cmds               4

#endif // UNDEF_IRIS_MACROS

typedef struct _iris_task      iris_task;
typedef struct _iris_mem       iris_mem;
typedef struct _iris_kernel    iris_kernel;
typedef struct _iris_graph     iris_graph;

typedef int (*iris_host_task)(void* params, const int* device);
typedef int (*command_handler)(void* params, void* device);
typedef int (*hook_task)(void* task);
typedef int (*hook_command)(void* command);

typedef int (*iris_selector_kernel)(iris_task task, void* params, char* kernel_name);

/**
 * OLD
 * Initializes the IRIS execution environment.
 * @param argc pointer to the number of arguments
 * @param argv argument array
 * @param sync 0: non-blocking, 1: blocking
 * @return All IRIS functions return an error value. IRIS_SUCCESS, IRIS_ERR
 */
extern int iris_init(int* argc, char*** argv, int sync);

/**
 * OLD
 * Return number of errors occurred in IRIS
 */
extern int iris_error_count();

/**
 * OLD
 * Terminates the IRIS execution environment.
 */
extern int iris_finalize();

/**
 * OLD
 * Waits for all the submitted tasks to complete.
 */
extern int iris_synchronize();

/**
 * OLD
  * If task need to be submitted again and again.
  */
extern void iris_task_retain(iris_task task, bool flag);

/**
 * OLD
 * Sets an IRIS environment variable.
 * @param key key string
 * @param value value to be stored into key
 */
extern int iris_env_set(const char* key, const char* value);

/**
 * OLD
 * Gets an IRIS environment variable.
 * @param key key string
 * @param value pointer to the value to be retrieved
 * @param vallen size in bytes of value
 */
extern int iris_env_get(const char* key, char** value, size_t* vallen);

/**
 * OLD
 * Gets the number of errors encountered during the IRIS environments history.
 */
extern int iris_error_count();

/**
 * OLD
 * Returns the number of platforms.
 * @param nplatforms pointer to the number of platform
 */
extern int iris_platform_count(int* nplatforms);

/**
 * OLD
 * Returns the platform information.
 * @param platform platform number
 * @param param information type
 * @param value information value
 * @param size size in bytes of value
 */
extern int iris_platform_info(int platform, int param, void* value, size_t* size);

/**
 * UPDATED
 * @brief Sets shared memory model
 * Using this function shared memory model can be set
 * @param flag 0: non shared memory, 1: shared memory 
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_set_shared_memory_model(int flag);

/**
 * OLD
 * Enable/disable profiler
 */
extern void iris_set_enable_profiler(int flag);
/**
 * OLD
 * Returns the number of devices.
 * @param ndevs pointer to the number of devices
 */
extern int iris_device_count(int* ndevs);

/**
 * OLD
 * Returns the device information.
 * @param device device number
 * @param param information type
 * @param value information value
 * @param size size in bytes of value
 */
extern int iris_device_info(int device, int param, void* value, size_t* size);


/**
 * UPDATED
 * @brief Sets the default device
 * Using this function default device can be set
 * @param device integer value representing the desired default device 
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_device_set_default(int device);


/**
 * UPDATED
 * @brief Gets the default device
 * Using this function default device can be obtained
 * @param device IRIS returns the default device on this variable 
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_device_get_default(int* device);

/**
 * OLD
 * Waits for all the submitted tasks in a device to complete.
 * @param ndevs number of devices
 * @param devices device array
 */
extern int iris_device_synchronize(int ndevs, int* devices);

/**
 * OLD
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
 * OLD
 * Creates a new task.
 * @param task pointer of the new task
 */
extern int iris_task_create(iris_task* task);

/**
 * UPDATED
 * QUESTION
 * @brief Creates a task with PERM????
 * Using this function IRIS creates a task???
 * @param task the task pointer
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_task_create_perm(iris_task* task);


/**
 * UPDATED
 * @brief Creates a task with a given name
 * Using this function IRIS creates a task object where the name is set from the function argument
 * @param name name of the task
 * @param task the task pointer
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_task_create_name(const char* name, iris_task* task);

/**
 * OLD
 * Adds a dependency to a task.
 * @param task source task
 * @param ntasks number of tasks
 * @param tasks target tasks array
 */
extern int iris_task_depend(iris_task task, int ntasks, iris_task* tasks);
extern int iris_task_malloc(iris_task task, iris_mem mem);
extern int iris_task_cmd_reset_mem(iris_task task, iris_mem mem, uint8_t reset);

/**
 * UPDATED
 * @brief Gets task meta data
 * This function used for getting optional task metadata through the specified index
 * @param brs_task iris task object
 * @param index index to obtain the correct meta data
 * @return returns the metadata for that index
 */
extern int iris_task_get_metadata(iris_task brs_task, int index);

/**
 * UPDATED
 * @brief Sets task meta data
 * This function used for setting optional task metadata through the specified index
 * @param brs_task iris task object
 * @param index index to set the correct meta data
 * @param meta_data the meta data needs to be saved
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_task_set_metadata(iris_task brs_task, int index, int metadata);

/**
 * OLD
 * Adds a H2Broadcast command to the target task.
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host source host address
 */
extern int iris_task_h2broadcast(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_h2broadcast_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);
extern int iris_task_h2broadcast_full(iris_task task, iris_mem mem, void* host);

/**
 * OLD
 * Adds a D2D command to the target task.
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host source host address
 * @param src_dev
 */
extern int iris_task_d2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host, int src_dev);
/**
 * OLD
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
 * OLD
 * Adds a D2H command to the target task.
 * @param task target task
 * @param mem source memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host target host address
 */
extern int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host);
extern int iris_task_d2h_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);


/**
 * UPDATED
 * @brief Adds a flush command to a task
 * This function flushes the given memory object to host
 * @param task iris task object
 * @param mem iris memory object that is specifed to the flush to host side
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_task_dmem_flush_out(iris_task task, iris_mem mem);

/**
 * OLD
 * Adds a H2D command with the size of the target memory to the target task.
 * @param task target task
 * @param mem target memory object
 * @param host source host address
 */
extern int iris_task_h2d_full(iris_task task, iris_mem mem, void* host);

/**
 * OLD
 * Adds a D2H command with the size of the source memory to the target task.
 * @param task target task
 * @param mem source memory object
 * @param host target host address
 */
extern int iris_task_d2h_full(iris_task task, iris_mem mem, void* host);
extern int iris_task_kernel_object(iris_task task, iris_kernel kernel, int dim, size_t* off, size_t* gws, size_t* lws);

/**
 * OLD
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
extern int iris_task_kernel_launch_disabled(iris_task task, int flag);
extern int iris_task_host(iris_task task, iris_host_task func, void* params);
extern int iris_task_host(iris_task task, iris_host_task func, void* params);
extern int iris_task_custom(iris_task task, int tag, void* params, size_t params_size);

/**
 * OLD
 * Submits a task.
 * @param task target task
 * @param device device_selector
 * @param opt option string
 * @param sync 0: non-blocking, 1: blocking
 */
extern int iris_task_submit(iris_task task, int device, const char* opt, int sync);


/**
 * UPDATED
 * @brief Sets a scheduling policy for a task
 * This function sets scheduling policy for a task 
 * @param task iris task object
 * @param device device or scheduling policy
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_task_set_policy(iris_task task, int device);

/**
 * OLD
 * Waits for the task to complete.
 * @param task target task
 */
extern int iris_task_wait(iris_task task);

/**
 * OLD
 * Waits for all the tasks to complete.
 * @param ntasks number of tasks
 * @param tasks target tasks array
 */
extern int iris_task_wait_all(int ntasks, iris_task* tasks);

/**
 * UPDATED
 * @brief Adds a subtask for a task
 * This function adds a subtask for a task
 * @param task iris task object
 * @param subtask the subtask that is going to be added
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_task_add_subtask(iris_task task, iris_task subtask);

/**
 * UPDATED
 * @brief Retruns whether a task only has kernel command
 * This function returns whether a task has only kernel command or not 
 * @param task iris task object
 * @return returns true if only kernel present in the task otherwise false
 */
extern int iris_task_kernel_cmd_only(iris_task task);

/**
 * OLD
 * Releases a target.
 * @param task target task
 */
extern int iris_task_release(iris_task task);
extern int iris_task_release_mem(iris_task task, iris_mem mem);
extern int iris_params_map(iris_task task, int *params_map);
extern int iris_task_info(iris_task task, int param, void* value, size_t* size);

/**
 * UPDATED
 * @brief Registers pin memory
 * This function enables pinning of host memory
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_register_pin_memory(void *host, size_t size);

/**
 * UPDATED
 * @brief Cretes IRIS memory object
 * This function creates IRIS memory object for a given size
 * @param size size of the memory
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_mem_create(size_t size, iris_mem* mem);

/**
 * UPDATED
 * @brief Resets a memory object by setting the dirty flag for host
 * This function resets a memory object by setting the dirty flag for host
 * @param mem pointer to the memory object
 * @param reset 0: no reseting 1: reset
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_data_mem_init_reset(iris_mem mem, int reset);

/**
 * UPDATED
 * @brief Cretes IRIS data memory object
 * This function creates IRIS data memory object for a given size
 * @param mem pointer to the memory object
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_data_mem_create(iris_mem* mem, void *host, size_t size);

/**
 * UPDATED
 * @brief Frees memory for a DMEM object for all the devices
 * This function Resets a memory object by setting the dirty flag for host
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_data_mem_clear(iris_mem mem);


/**
 * UPDATED
 * QUESTION : it pins a host memory for all the available platforms
 * @brief  Pins a host memory for all the available platforms
 * This function pins a host memory for all the available platforms
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern int iris_data_mem_pin(iris_mem mem);
extern int iris_data_mem_update(iris_mem mem, void *host);
extern int iris_data_mem_create_region(iris_mem* mem, iris_mem root_mem, int region);
extern int iris_data_mem_enable_outer_dim_regions(iris_mem mem);
extern int iris_data_mem_create_tile(iris_mem* mem, void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
extern int iris_data_mem_n_regions(iris_mem brs_mem);
extern unsigned long iris_data_mem_get_region_uid(iris_mem brs_mem, int region);
extern int iris_mem_arch(iris_mem mem, int device, void** arch);
extern int iris_mem_reduce(iris_mem mem, int mode, int type);
extern int iris_mem_release(iris_mem mem);

extern int iris_graph_create(iris_graph* graph);
extern int iris_graph_free(iris_graph graph);
extern int iris_graph_tasks_order(iris_graph brs_graph, int *order);
extern int iris_graph_create_json(const char* json, void** params, iris_graph* graph);
extern int iris_graph_task(iris_graph graph, iris_task task, int device, const char* opt);
extern int iris_graph_retain(iris_graph graph, bool flag);
extern int iris_graph_release(iris_graph graph);
extern int iris_graph_submit(iris_graph graph, int device, int sync);
extern int iris_graph_submit_with_order(iris_graph graph, int *order, int device, int sync);
extern int iris_graph_submit_with_order_and_time(iris_graph graph, int *order, double *time, int device, int sync);
extern int iris_graph_submit_with_time(iris_graph graph, double *time, int device, int sync);
extern int iris_graph_wait(iris_graph graph);
extern int iris_graph_wait_all(int ngraphs, iris_graph* graphs);

extern int iris_record_start();
extern int iris_record_stop();

/**
 * OLD
 * Returns current time in seconds.
 * @param time pointer of time
 */
extern int iris_timer_now(double* time);


/**
 * UPDATED
 * @brief Enables peer to peer transfer
 * This function enables peer to peer transfer
 */
extern void iris_enable_d2d();

/**
 * UPDATED
 * @brief Disables peer to peer transfer
 * This function disables peer to peer transfer
 */
extern void iris_disable_d2d();
extern void iris_disable_consistency_check();
extern void iris_enable_consistency_check();


/**
 * UPDATED
 * @brief Returns a kernel name
 * This function returns a kernel name
 * @param brs_kernel kernel object
 * @return This function returns name of the kernel.
 */
extern char *iris_kernel_get_name(iris_kernel brs_kernel);


/**
 * UPDATED
 * @brief Retruns a task name
 * This function returns a task name
 * @param brs_task task object
 * @return This function returns name of the task.
 */
extern char *iris_task_get_name(iris_task brs_task);


/**
 * UPDATED
 * @brief Sets a task name
 * This function Sets a task name
 * @param brs_task task object
 * @param name name of the task
 * @return This function returns an integer indicating IRIS STATUS.
 */
extern void iris_task_set_name(iris_task brs_task, const char *name);
extern int iris_task_get_dependency_count(iris_task brs_task);
extern void iris_task_get_dependencies(iris_task brs_task, iris_task *tasks);
extern unsigned long iris_task_get_uid(iris_task brs_task);
extern unsigned long iris_kernel_get_uid(iris_kernel brs_kernel);
extern iris_kernel iris_task_get_kernel(iris_task brs_task);
extern int iris_task_kernel_dmem_fetch_order(iris_task brs_task, int *order);
extern int iris_task_disable_consistency(iris_task brs_task);
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
extern int iris_graph_enable_mem_profiling(iris_graph brs_graph);
extern int iris_graph_reset_memories(iris_graph graph);
extern int iris_graph_get_tasks(iris_graph graph, iris_task *tasks);
extern int iris_graph_tasks_count(iris_graph graph);
extern int iris_get_graph_dependency_adj_list(iris_graph brs_graph, int8_t *dep_matrix);
extern int iris_get_graph_dependency_adj_matrix(iris_graph brs_graph, int8_t *dep_matrix);
extern size_t iris_get_graph_3d_comm_data_size(iris_graph brs_graph);
extern void *iris_get_graph_3d_comm_data_ptr(iris_graph brs_graph);
extern void *iris_get_graph_tasks_execution_schedule(iris_graph brs_graph, int kernel_profile);
extern size_t iris_get_graph_tasks_execution_schedule_count(iris_graph brs_graph);
extern void *iris_get_graph_dataobjects_execution_schedule(iris_graph brs_graph);
extern size_t iris_get_graph_dataobjects_execution_schedule_count(iris_graph brs_graph);
extern int iris_get_graph_3d_comm_data(iris_graph brs_graph, void *comm_data);
extern int iris_get_graph_2d_comm_adj_matrix(iris_graph brs_graph, size_t *size_data);
extern int iris_calibrate_compute_cost_adj_matrix(iris_graph brs_graph, double *comp_data);
extern int iris_calibrate_compute_cost_adj_matrix_only_for_types(iris_graph brs_graph, double *comp_data);
extern int iris_calibrate_communication_cost(double *data, size_t data_size, int iterations, int pin_memory_flag);
extern int iris_get_graph_3d_comm_time(iris_graph brs_graph, double *comm_time, int *mem_ids, int iterations, int pin_memory_flag);
extern size_t iris_count_mems(iris_graph brs_graph);
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


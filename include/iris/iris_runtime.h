#ifndef IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H
#define IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H

#include <iris/iris_errno.h>
#include <stddef.h>
#include <stdint.h>

#define DMEM_MAX_DIM 6

#ifdef __cplusplus
namespace iris {
namespace rt {
class Kernel;
class BaseMem;
class Mem;
class Task;
class Device;
class Graph;
} /* namespace rt */
} /* namespace iris */
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct __attribute__ ((packed)) _iris_task {
#endif // DOXYGEN_SHOULD_SKIP_THIS
#ifdef __cplusplus
  iris::rt::Task* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct __attribute__ ((packed)) _iris_kernel {
#endif // DOXYGEN_SHOULD_SKIP_THIS
#ifdef __cplusplus
  iris::rt::Kernel* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct __attribute__ ((packed)) _iris_mem {
#endif // DOXYGEN_SHOULD_SKIP_THIS
#ifdef __cplusplus
  iris::rt::BaseMem* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct __attribute__ ((packed)) _iris_device {
#endif // DOXYGEN_SHOULD_SKIP_THIS
#ifdef __cplusplus
  iris::rt::Device* class_obj;
#else
  void *class_obj;
#endif
  unsigned long uid;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct __attribute__ ((packed)) _iris_graph {
#endif // DOXYGEN_SHOULD_SKIP_THIS
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
#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef int8_t bool;
#endif
#endif

enum StreamPolicy {
    STREAM_POLICY_DEFAULT = 0,
    STREAM_POLICY_SAME_FOR_TASK = 1,
    STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL = 2
};
typedef enum StreamPolicy StreamPolicy;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
#ifndef UNDEF_IRIS_MACROS
#define IRIS_MAX_NPLATFORMS     32
#define IRIS_MAX_NDEVS          (1 << 8) - 1
#define IRIS_MAX_KERNEL_NARGS     64
#define IRIS_MAX_DEVICE_NSTREAMS   11
#define IRIS_MAX_DEVICE_NCOPY_STREAMS   3

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
#define iris_sdq                (1 << 24)
#define iris_ftf                (1 << 25)
#define iris_all                (1 << 25)
#define iris_ocl                (1 << 26)
#define iris_block_cycle        (1 << 27)
#define iris_custom             (1 << 28)
#define iris_julia_policy       (1 << 29)

enum DeviceModel {
iris_cuda = 1,
iris_hip = 3,
iris_levelzero = 4,
iris_opencl = 5,
iris_openmp = 6,
iris_model_all = (1 << 25) // Same as iris_all
};
typedef enum DeviceModel DeviceModel;
//#define iris_cuda               1
////#define iris_hexagon            2
//#define iris_hip                3
//#define iris_levelzero          4
//#define iris_opencl             5
//#define iris_openmp             6

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

#define iris_julia_native               0
#define iris_core_native                1
#define iris_julia_kernel_abstraction   2
#define iris_julia_jacc                 3


#define iris_unknown            (0 << 16)
#define iris_int                (1 << 16)
#define iris_uint               (2 << 16)
#define iris_float              (3 << 16)
#define iris_double             (4 << 16)
#define iris_char               (5 << 16)
#define iris_int8               (6 << 16)
#define iris_uint8              (7 << 16)
#define iris_int16              (8 << 16)
#define iris_uint16             (9 << 16)
#define iris_int32              (10 << 16)
#define iris_uint32             (11 << 16)
#define iris_int64              (12 << 16)
#define iris_uint64             (13 << 16)
#define iris_long               (14 << 16)
#define iris_unsigned_long      (15 << 16)
#define iris_bool               (16 << 16)
#define iris_custom_type        (17 << 16)
#define iris_pointer            (0x4000 << 16)

#define iris_reset_memset                      0
#define iris_reset_assign                      1 
#define iris_reset_arith_seq                   2
#define iris_reset_geom_seq                    3
#define iris_reset_random_uniform_seq          4
#define iris_reset_random_normal_seq           5 
#define iris_reset_random_log_normal_seq       6
#define iris_reset_random_uniform_sobol_seq    7
#define iris_reset_random_normal_sobol_seq     8
#define iris_reset_random_log_normal_sobol_seq 9

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
#define iris_task_time_submit   5
#define iris_task_time_start    6
#define iris_task_time_end      7

// The event wait flags aligned with HIP and CUDA
#define iris_event_wait_default          0
#define iris_event_wait_external         1

// The event flags aligned with HIP and CUDA
#define iris_event_default          0
#define iris_event_blocking_sync    1
#define iris_event_disable_timing   2
#define iris_event_interprocess     3

// Stream flags
#define iris_stream_default         0
#define iris_stream_non_blocking    1

#endif // UNDEF_IRIS_MACROS
#endif // DOXYGEN_SHOULD_SKIP_THIS

typedef struct _iris_task      iris_task;
typedef struct _iris_mem       iris_mem;
typedef struct _iris_kernel    iris_kernel;
typedef struct _iris_graph     iris_graph;
typedef struct _iris_device    iris_device;

typedef int (*iris_host_task)(void* params, const int* device);
typedef int (*iris_host_python_task)(int64_t* params_id, const int* device);
typedef int (*command_handler)(void* params, void* device);
typedef int (*hook_task)(void* task);
typedef int (*hook_command)(void* command);

typedef int (*iris_selector_kernel)(iris_task task, void* params, char* kernel_name);

typedef union _IRISValue {
    int8_t   i8;
    int16_t  i16;
    int32_t  i32;
    int64_t  i64;
    uint8_t  u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float    f32;
    double   f64;
}IRISValue;

typedef struct _ResetData {
    IRISValue value_;
    IRISValue start_;
    IRISValue step_;
    IRISValue p1_;
    IRISValue p2_;
    int reset_type_;
    long long seed_;
}ResetData;
/**@brief Initializes the IRIS execution environment.
 *
 * This function initializes the IRIS execution environment.
 *
 * @param argc pointer to the number of arguments
 * @param argv argument array
 * @param sync 0: non-blocking, 1: blocking
 * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
 */
extern int iris_init(int* argc, char*** argv, int sync);


/**@brief Return number of errors occurred in IRIS
 *
 * @return This function returns the number of errors
 */
extern int iris_error_count();


/**@brief Terminates the IRIS execution environment.
 *
 * this funciton put end to IRIS execution environment.
 *
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_finalize();


/**@brief Puts a synchronization for tasks to complete
 *
 * This function makes IRIS Wait for all the submitted tasks to complete.
 *
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_synchronize();

/**@brief Enable default kernels compilation and loading
  *
  * This function enables the runtime compilation of default kernels and loading the shared library for OpenMP, CUDA and HIP.
  * @return Nothing
  */
extern void iris_enable_default_kernels(int flag);

/**@brief Makes sure a can be submitted again and again.
 *
 * This function makes a task with an option to be submitted again and again.
 *
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern void iris_task_retain(iris_task task, int flag);

extern int iris_task_set_julia_policy(iris_task task, const char *name);

/**@brief Sets an IRIS environment variable.
 *
 * @param key key string
 * @param value value to be stored into key
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_env_set(const char* key, const char* value);


/**@brief Gets an IRIS environment variable.
 *
 * @param key key string
 * @param value pointer to the value to be retrieved
 * @param vallen size in bytes of value
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_env_get(const char* key, char** value, size_t* vallen);


/**@brief Prints an overview of the system available to IRIS; specifically
 * platforms, devices and their corresponding backends.
 * It is logged to standard output.
 */
extern void iris_overview();


/**@brief Returns the number of platforms.
 *
 * @param nplatforms pointer to the number of platform
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_platform_count(int* nplatforms);


/**@brief Returns the platform information.
 *
 * @param platform platform number
 * @param param information type
 * @param value information value
 * @param size size in bytes of value
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_platform_info(int platform, int param, void* value, size_t* size);


/**@brief set IRIS stream policy type 
 *
 * IRIS by default has stream policy type as STREAM_POLICY_DEFAULT (It selects stream policy based on device type)
 *
 * @param policy : Policy type of IRIS of data-type StreamPolicy 
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_set_stream_policy(StreamPolicy policy);

/**@brief Enable IRIS asynchronous task execution feature
 *
 * IRIS by default has asynchronous disabled
 *
 * @param flag 0: disabled, 1: enabled
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_set_asynchronous(int flag);

/**@brief Sets shared memory model for OpenMP device
 *
 * Using this function shared memory model can be set
 *
 * @param flag 0: non shared memory, 1: shared memory
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_set_shared_memory_model(int flag);

/**@brief Enable shared memory model for the given memory and device type 
 *
 * Using this function shared memory model can be set
 *
 * @param mem iris memory object
 * @param type : Device types (iris_cuda, iris_hip, iris_openmp, iris_opencl) 
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_enable_usm(iris_mem mem, DeviceModel type);

/**@brief Enable shared memory model for the given memory 
 *
 * Using this function shared memory model can be set
 *
 * @param mem iris memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_enable_usm_all(iris_mem mem);

/**@brief Enable shared memory model for the given memory and device type
 *
 * Using this function shared memory model can be set
 *
 * @param mem iris memory object
 * @param type : Device types (iris_cuda, iris_hip, iris_openmp, iris_opencl) 
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_disable_usm(iris_mem mem, DeviceModel type);

/**@brief Enable shared memory model for the given memory
 *
 * Using this function shared memory model can be set
 *
 * @param mem iris memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_disable_usm_all(iris_mem mem);

/**@brief Enable/disable profiler
 *
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern void iris_set_enable_profiler(int flag);


/**@brief Returns the number of devices.
 *
 * @param ndevs pointer to the number of devices
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_device_count(int* ndevs);

/**@brief Returns the number of devices
 *
 * @return This function returns an integer indicating number of devices
 */
extern int iris_ndevices();

/**@brief Set the number of streams
 *
 * @return This function set an integer indicating number of streams
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_set_nstreams(int n);

/**@brief Set the number of copy streams
 *
 * @param This function takes an integer indicating number of copy streams
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_set_ncopy_streams(int n);

/**@brief Returns the number of streams
 *
 * @param This function takes an integer indicating number of streams
 */
extern int iris_nstreams();

/**@brief Returns the number of copy streams.
 *
 * @param ndevs pointer to the number of copy streams 
 * @return This function returns an integer indicating number of copy streams
 */
extern int iris_ncopy_streams();

/**@brief Returns the device information.
 *
 * @param device device number
 * @param param information type
 * @param value information value
 * @param size size in bytes of value
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_device_info(int device, int param, void* value, size_t* size);



/**@brief Sets the default device
 *
 * Using this function default device can be set
 *
 * @param device integer value representing the desired default device
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_device_set_default(int device);



/**@brief Gets the default device
 *
 * Using this function default device can be obtained
 *
 * @param device IRIS returns the default device on this variable
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_device_get_default(int* device);


/**@brief Waits for all the submitted tasks in a device to complete.
 *
 * @param ndevs number of devices
 * @param devices device array
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_device_synchronize(int ndevs, int* devices);


/**@brief Registers a new device selector
 *
 * @param lib shared library path
 * @param name selector name
 * @param params parameter to the selector init function
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_register_policy(const char* lib, const char* name, void* params);

/**@brief Registers a custom command specific to the device with the given command handler
 *
 * @param tag unique identification to register the custom command
 * @param device device selection (iris_openmp, iris_cuda, iris_hip, iris_levelzero, iris_opencl)
 * @param handler handler function for the command
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_register_command(int tag, int device, command_handler handler);

/**@brief Register functions to be called for each task before execution and after execution
 *
 * @param pre Function with signature int (*function)(void *task) to be called before task execution
 * @param post Function with signature int (*function)(void *task) to be called after task execution
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_register_hooks_task(hook_task pre, hook_task post);

/**@brief Register functions to be called for each command before execution and after execution
 *
 * @param pre Function with signature int (*function)(void *task) to be called before command execution
 * @param post Function with signature int (*function)(void *task) to be called after command execution
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_register_hooks_command(hook_command pre, hook_command post);


/**@brief Creates a kernel with a given name
 *
 * @param name kernel name string
 * @param kernel
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_kernel_create(const char* name, iris_kernel* kernel);

/**@brief Creates a kernel with a given name
 *
 * @param name kernel name string
 * @return This function returns an iris_kernel object
 */
extern iris_kernel iris_kernel_create_struct(const char* name);


/**@brief Creates a kernel with a given name
 *
 * @param name kernel name string
 * @param kernel a pointer to a kernel object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_kernel_get(const char* name, iris_kernel* kernel);


/**@brief Sets the arguments for a given kernel
 *
 * @param kernel a kernel object
 * @param idx index of the parameter
 * @param size size of the argument
 * @param value value that needs to be set
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_kernel_setarg(iris_kernel kernel, int idx, size_t size, void* value);


/**@brief Sets memory object as an arguments for a given kernel
 *
 * @param kernel a kernel object
 * @param idx index of the parameter
 * @param mem iris memory object
 * @param mode specifying the mode of the memory object iris_r, iris_w, or iris_rw
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_kernel_setmem(iris_kernel kernel, int idx, iris_mem mem, size_t mode);


/**@brief Sets memory object as an arguments with an offset for a given kernel
 *
 * @param kernel a kernel object
 * @param idx index of the parameter
 * @param mem iris memory object
 * @param off offset for the memory object
 * @param mode specifying the mode of the memory object iris_r, iris_w, or iris_rw
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_kernel_setmem_off(iris_kernel kernel, int idx, iris_mem mem, size_t off, size_t mode);


/**@brief Release a kernel
 *
 * @param kernel a kernel object that is to be releases
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_kernel_release(iris_kernel kernel);


/**@brief Creates a new task.
 *
 * @return This function returns an iris_task struct object 
 */
extern iris_task iris_task_create_struct();

/**@brief Creates a new task.
 *
 * @param task pointer of the new task
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_create(iris_task* task);


/**@brief Creates a task with permanent life time. Task memory won't be released after execution. It can't be used to submit the task again and again. Application programmer should call task release API after successful completion of the task executions.
 *
 * @param task the task pointer
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_create_perm(iris_task* task);



/**@brief Creates a task with a given name
 *
 * Using this function IRIS creates a task object where the name is set from the function argument
 *
 * @param name name of the task
 * @param task the task pointer
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_create_name(const char* name, iris_task* task);


/**@brief Adds a dependency to a task.
 *
 * Adds a dependency to a task.
 * @param task source task
 * @param ntasks number of tasks
 * @param tasks target tasks array
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_depend(iris_task task, int ntasks, iris_task* tasks);


/**@brief Mallocs for a memory object in a given task
 *
 * @param task iris task object
 * @param mem memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_malloc(iris_task task, iris_mem mem);

/**@brief Resets a memory object for a given task
 *
 * @param task iris task object
 * @param mem memory object
 * @param reset using the value the memory object is initialized
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_cmd_reset_mem(iris_task task, iris_mem mem, uint8_t reset);
extern int iris_task_cmd_init_reset_assign(iris_task brs_task, iris_mem brs_mem, IRISValue value);
extern int iris_task_cmd_init_reset_arith_seq(iris_task brs_task, iris_mem brs_mem, IRISValue start, IRISValue increment);
extern int iris_task_cmd_init_reset_geom_seq(iris_task brs_task, iris_mem brs_mem, IRISValue start, IRISValue step);
extern int iris_task_cmd_init_reset_random_uniform_seq(iris_task brs_task, iris_mem brs_mem, long long seed, IRISValue min, IRISValue max);
extern int iris_task_cmd_init_reset_random_normal_seq(iris_task brs_task, iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev);
extern int iris_task_cmd_init_reset_random_log_normal_seq(iris_task brs_task, iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev);
extern int iris_task_cmd_init_reset_random_uniform_sobol_seq(iris_task brs_task, iris_mem brs_mem, IRISValue min, IRISValue max);
extern int iris_task_cmd_init_reset_random_normal_sobol_seq(iris_task brs_task, iris_mem brs_mem, IRISValue mean, IRISValue stddev);
extern int iris_task_cmd_init_reset_random_log_normal_sobol_seq(iris_task brs_task, iris_mem brs_mem, IRISValue mean, IRISValue stddev);


/**@brief set task level IRIS stream policy type 
 *
 * IRIS by default has stream policy type as STREAM_POLICY_DEFAULT (It selects stream policy based on device type)
 *
 * @param policy : Policy type of IRIS of data-type StreamPolicy 
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_set_stream_policy(iris_task brs_task, StreamPolicy policy);

/**@brief Disable task asynchronous execution
 *
 * This function disables asynchronous task execution (data transfers, kernel execution) for the given task even if it is supported by the device. If the device is not supported for asynchronous task execution, this flag is ignored.
 *
 * @param brs_task iris task object
 */
extern void iris_task_disable_asynchronous(iris_task brs_task);

/**@brief Gets task meta data array pointer
 *
 * This function used for getting optional task metadata 
 *
 * @param brs_task iris task object
 * @return returns the pointer of metadata array 
 */
extern int *iris_task_get_metadata_all(iris_task brs_task);


/**@brief Sets task meta data array
 *
 * This function used for setting optional task metadata 
 *
 * @param brs_task iris task object
 * @param meta_data the meta data array pointer needs to be saved
 * @param n count of metadata
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_set_metadata_all(iris_task brs_task, int *mdata, int n);

/**@brief Gets task meta data
 *
 * This function used for getting optional task metadata through the specified index
 *
 * @param brs_task iris task object
 * @param index index to obtain the correct meta data
 * @return returns the metadata for that index
 */
extern int iris_task_get_metadata(iris_task brs_task, int index);


/**@brief Sets task meta data
 *
 * This function used for setting optional task metadata through the specified index
 *
 * @param brs_task iris task object
 * @param index index to set the correct meta data
 * @param meta_data the meta data needs to be saved
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_set_metadata(iris_task brs_task, int index, int metadata);

/**@brief Get task meta count
 *
 * This function used to get total metadata count 
 *
 * @param brs_task iris task object
 * @return This function returns an integer indicating count of metadata 
 */
extern int iris_task_get_metadata_count(iris_task brs_task);

/**@brief Adds a H2Broadcast command to the target task.
 *
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_h2broadcast(iris_task task, iris_mem mem, size_t off, size_t size, void* host);

/**@brief Adds a H2Broadcast command to the target task that broadcast a portion of the memory
 *
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param host_sizes size in bytes for host
 * @param dev_sizes size in bytes for host
 * @param elem_size size of an element
 * @param dim dimension of the memory
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_h2broadcast_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);

/**@brief Adds a H2Broadcast command to the target task that broadcasts the full host memory
 *
 * @param task target task
 * @param mem target memory object
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_h2broadcast_full(iris_task task, iris_mem mem, void* host);


/**@brief Adds a D2D command to the target task.
 *
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host source host address
 * @param src_dev
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_d2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host, int src_dev);


/**@brief Adds a source DMEM to destination DMEM command to the target task.
 *
 * @param task target task
 * @param src_mem source DMEM memory object
 * @param dst_mem target DMEM memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_dmem2dmem(iris_task task, iris_mem src_mem, iris_mem dst_mem);

/**@brief Adds a H2D command to the target task.
 *
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_h2d(iris_task task, iris_mem mem, size_t off, size_t size, void* host);

/**@brief Add hidden DMEM to a task 
 *
 * @param task target task
 * @param mem target DMEM memory object
 * @param mode Mode of DMEM object to task (iris_r, iris_w, iris_rw)
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_hidden_dmem(iris_task brs_task, iris_mem brs_mem, int mode);

/**@brief Adds a H2D command to the target task.
 *
 * @param task target task
 * @param mem target memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_dmem_h2d(iris_task task, iris_mem mem);

/**@brief Adds a H2D command to the target task for a portion of the memory
 *
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param host_sizes indexes for specifying host memory size
 * @param dev_sizes indexes for specifying device memory size
 * @param elem_size size of each element
 * @param dim dimension of the memory object
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_h2d_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);


/**@brief Adds a D2H command to the target task.
 *
 * @param task target task
 * @param mem source memory object
 * @param off offset in bytes
 * @param size size in bytes
 * @param host target host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_d2h(iris_task task, iris_mem mem, size_t off, size_t size, void* host);

/**@brief Adds a D2H command to the target task.
 *
 * @param task target task
 * @param mem source memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_dmem_d2h(iris_task task, iris_mem mem);

/**@brief Adds a D2H command to the target task for a portion of the memory
 *
 * @param task target task
 * @param mem target memory object
 * @param off offset in bytes
 * @param host_sizes indexes for specifying host memory size
 * @param dev_sizes indexes for specifying device memory size
 * @param elem_size size of each element
 * @param dim dimension of the memory object
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_d2h_offsets(iris_task task, iris_mem mem, size_t *off, size_t *host_sizes,  size_t *dev_sizes, size_t elem_size, int dim, void* host);


/**@brief Initialize Worker for the given device number
 *
 * @param dev  iris device number
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_init_worker(int dev);

/**@brief Start Worker for the given device number
 *
 * @param dev  iris device number
 * @param use_pthread either to use native pthread (1/0)
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_start_worker(int dev, int use_pthread);

/**@brief Initialize and Start scheduler 
 *
 * @param use_pthread either to use native pthread (1/0)
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_init_scheduler(int use_pthread);

/**@brief Initialize Device with init task for the given device number
 *
 * @param dev  iris device number
 * @param use_pthread either to use native pthread (1/0)
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_init_device(int dev);

/**@brief Synchronize all initialized Devices
 *
 * @param sync 0: non-blocking, 1: blocking
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_init_devices_synchronize(int sync);

/**@brief Initialize devices 
 *
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_init_devices(int sync);

/**@brief Enable Julia Interface for task kernels inside
 *
 * @param task iris task object
 * @param type type of julia kernel (iris_julia_native, iris_core_native, iris_julia_kernel_abstraction, iris_julia_jacc)
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_enable_julia_interface(iris_task task, int type);

/**@brief Adds a flush command to a task
 *
 * This function flushes the given memory object to host
 *
 * @param task iris task object
 * @param mem iris memory object that is specifed to the flush to host side
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_dmem_flush_out(iris_task task, iris_mem mem);


/**@brief Adds a H2D command with the size of the target memory to the target task.
 *
 * @param task target task
 * @param mem target memory object
 * @param host source host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_h2d_full(iris_task task, iris_mem mem, void* host);


/**@brief Adds a D2H command with the size of the source memory to the target task.
 *
 * @param task target task
 * @param mem source memory object
 * @param host target host address
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_d2h_full(iris_task task, iris_mem mem, void* host);


/**@brief Launches a kernel
 *
 * @param task target task
 * @param kernel kernel name
 * @param dim dimension
 * @param off global workitem space offsets
 * @param gws global workitem space
 * @param lws local workitem space
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_kernel_object(iris_task task, iris_kernel kernel, int dim, size_t* off, size_t* gws, size_t* lws);


/**@brief Launches a kernel
 *
 * @param task target task
 * @param kernel kernel name
 * @param dim dimension
 * @param off global workitem space offsets
 * @param gws global workitem space
 * @param lws local workitem space
 * @param nparams number of kernel parameters
 * @param params kernel parameters
 * @param params_info kernel parameters information
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_kernel(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info);

/**@brief Launches a kernel with parameter offset
 *
 * @param task target task
 * @param kernel kernel name
 * @param dim dimension
 * @param off global workitem space offsets
 * @param gws global workitem space
 * @param lws local workitem space
 * @param nparams number of kernel parameters
 * @param params kernel parameters
 * @param params_off kernel parameters offset
 * @param params_info kernel parameters information
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_kernel_v2(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info);

/**@brief Launches a kernel with parameter offset and memory ranges
 *
 * @param task target task
 * @param kernel kernel name
 * @param dim dimension
 * @param off global workitem space offsets
 * @param gws global workitem space
 * @param lws local workitem space
 * @param nparams number of kernel parameters
 * @param params kernel parameters
 * @param params_off kernel parameters offset
 * @param params_info kernel parameters information
 * @param memranges sizes of the memory object from the offset
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_kernel_v3(iris_task task, const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, size_t* params_off, int* params_info, size_t* memranges);

/**@brief passes a kernel selector function for selecting a kernel from a task
 *
 * @param task target task
 * @param func function to select a kernel from task
 * @param params kernel parameters
 * @param params_size size of the parameters
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_kernel_selector(iris_task task, iris_selector_kernel func, void* params, size_t params_size);

/**@brief disable kernel launch from a task
 *
 * @param task target task
 * @param flag bool value, 0: launch enable, 1: launch disable
 * @param params kernel parameters
 * @param params_size size of the parameters
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_kernel_launch_disabled(iris_task task, int flag);

/**@brief executes a function at the host side
 *
 * @param task target task
 * @param func function to be executed
 * @param params kernel parameters
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_python_host(iris_task task, iris_host_python_task func, int64_t params_id);

/**@brief executes a function at the host side
 *
 * @param task target task
 * @param func function to be executed
 * @param params kernel parameters
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_host(iris_task task, iris_host_task func, void* params);

/**@brief add a custom command to the task which specified registerng a command
 *
 * @param task target task
 * @param tag custom command tag id
 * @param params parameters
 * @param param_size parameter size
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_custom(iris_task task, int tag, void* params, size_t params_size);


/**@brief Submits a task.
 *
 * @param task target task
 * @param device device_selector
 * @param opt option string
 * @param sync 0: non-blocking, 1: blocking
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_submit(iris_task task, int device, const char* opt, int sync);



/**@brief Sets a scheduling policy for a task
 *
 * This function sets scheduling policy for a task
 *
 * @param task iris task object
 * @param policy either device index or scheduling policy
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_set_policy(iris_task task, int policy);

/**@brief Gets a scheduling policy for a task
 *
 * This function gets scheduling policy for a task
 *
 * @param task iris task object
 * @return This function returns the policy.
 */
extern int iris_task_get_policy(iris_task task);



/**@brief  Waits for the task to complete.
 *
 * @param task target task
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_wait(iris_task task);


/**@brief Waits for all the tasks to complete.
 *
 * @param ntasks number of tasks
 * @param tasks target tasks array
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_wait_all(int ntasks, iris_task* tasks);


/**@brief Adds a subtask for a task
 *
 * This function adds a subtask for a task
 *
 * @param task iris task object
 * @param subtask the subtask that is going to be added
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
//extern int iris_task_add_subtask(iris_task task, iris_task subtask);


/**@brief Retruns whether a task only has kernel command
 *
 * This function returns whether a task has only kernel command or not
 *
 * @param task iris task object
 * @return returns true if only kernel present in the task otherwise false
 */
extern int iris_task_kernel_cmd_only(iris_task task);


/**@brief Releases a task.
 *
 * @param task target task
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_release(iris_task task);

/**@brief Releases memory from a task
 *
 * @param task target task
 * @param mem memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_release_mem(iris_task task, iris_mem mem);

/**@brief Adds parameter map for a kernel in a task
 *
 * @param task target task
 * @param params_map parameter map
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_params_map(iris_task task, int *params_map);

/**@brief Gets parameter info for a task
 *
 * @param task target task
 * @param params parameter type -- options include:
 * iris_ncmds: the number of commands associated with this task
 * iris_ncmds_kernel: the number of kernel commands associated with this task
 * iris_ncmds_memcpy: the number of memory copy commands associated with this task
 * iris_cmds: an array of command types associated with this task
 * iris_task_time_submit: the timestamp of the task submission to the IRIS runtime
 * iris_task_time_start: the timestamp of the first compute kernel in this task starting
 * iris_task_time_end: the timestamp of the last compute kernel completion
 * @param value gets the value
 * @param size gets the size
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_task_info(iris_task task, int param, void* value, size_t* size);


/**@brief Unregisters pin memory
 *
 * This function disables pinning of host memory
 *
 * @param host host pointer of the data structure
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_unregister_pin_memory(void *host);

/**@brief Registers pin memory
 *
 * This function enables pinning of host memory
 *
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_register_pin_memory(void *host, size_t size);


/**@brief Cretes IRIS memory object
 *
 * This function creates IRIS memory object for a given size
 *
 * @param size size of the memory
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_create(size_t size, iris_mem* mem);


/**@brief Resets a memory object by setting the dirty flag for host
 *
 * This function resets a memory object by setting the dirty flag for host
 *
 * @param mem pointer to the memory object
 * @param reset 0: no reseting 1: reset
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_init_reset(iris_mem mem, int reset);

/**@brief Get DMem element type
 *
 * This function returns a DMEM element type if it is configured
 *
 * @param mem pointer to the memory object
 * @return This function returns an DMem element type
 */
extern int iris_get_dmem_element_type(iris_mem mem);

/**@brief Get DMem host pointer 
 *
 * This function returns a DMEM host assigned pointer; It will create host memory if it is null and return the address
 *
 * @param mem pointer to the memory object
 * @return This function returns an DMem host assigned pointer
 */
extern void *iris_get_dmem_valid_host(iris_mem mem);

/**@brief Get DMem host pointer 
 *
 * This function returns a DMEM host assigned pointer. It returns NULL if the host pointer set to DMEM is null
 *
 * @param mem pointer to the memory object
 * @return This function returns an DMem host assigned pointer
 */
extern void *iris_get_dmem_host(iris_mem mem);

/**@brief Get DMem host pointer 
 *
 * This function returns a DMEM host assigned pointer after fetching data from device. It returns NULL if the host pointer set to DMEM is null
 *
 * @param mem pointer to the memory object
 * @return This function returns an DMem host assigned pointer
 */
extern void *iris_get_dmem_host_fetch(iris_mem mem);

/**@brief Get DMem host pointer 
 *
 * This function returns a DMEM host assigned pointer after fetching data from device. It returns NULL if the host pointer set to DMEM is null
 *
 * @param mem pointer to the memory object
 * @param size data size to transfer
 * @return This function returns an DMem host assigned pointer
 */
extern void *iris_get_dmem_host_fetch_with_size(iris_mem mem, size_t size);

/**@brief Fetch DMem data and copy to host pointer
 *
 * This function copies data from active DMEM device to host assigned pointer 
 *
 * @param mem pointer to the memory object
 * @param host_ptr pointer to the host object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_fetch_dmem_data(iris_mem mem, void *host_ptr);

/**@brief Fetch DMem data and copy to host pointer
 *
 * This function copies data from active DMEM device to host assigned pointer 
 *
 * @param mem pointer to the memory object
 * @param host_ptr pointer to the host object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_fetch_dmem_data_with_size(iris_mem mem, void *host_ptr, size_t size);


/**@brief Add child DMEM to DMEM which make sure to have data of child mem available to the device and also set aits device pointer into the parent DMEM structure 
 *
 * This function copies data from active DMEM device to host assigned pointer 
 *
 * @param parent pointer to the memory object
 * @param child pointer to the memory object
 * @param offset offset to the parent memory structure
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_dmem_add_child(iris_mem parent, iris_mem child, size_t offset);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @return This function returns an iris_mem dmem object
 */
extern iris_mem iris_data_mem_create_struct(void *host, size_t size);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param host host pointer of the data structure
 * @param size ND-size array of the memory
 * @param dim  total dimensions
 * @param element_size Size of the element
 * @param element_type IRIS element type
 * @return This function returns an iris_mem dmem object
 */
extern iris_mem iris_data_mem_create_struct_nd(void *host, size_t *size, int dim, size_t element_size, int element_type);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @param type type of the memory element
 * @return This function returns an iris_mem dmem object
 */
extern iris_mem iris_data_mem_create_struct_with_type(void *host, size_t size, int element_type);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param mem pointer to the memory object
 * @param host host pointer of the data structure
 * @param size ND-size array of the memory
 * @param dim  total dimensions
 * @param element_size Size of the element
 * @param element_type IRIS element type
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_create_nd(iris_mem* mem, void *host, size_t *size, int dim, size_t element_size, int element_type);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param mem pointer to the memory object
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_create(iris_mem* mem, void *host, size_t size);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param mem pointer to the memory object
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @symbol symbol name to be looked into architecture kernel library files
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_create_symbol(iris_mem* mem, void *host, size_t size, const char *symbol);

/**@brief Cretes IRIS data memory object
 *
 * This function creates IRIS data memory object for a given size
 *
 * @param mem pointer to the memory object
 * @param host host pointer of the data structure
 * @param size size of the memory
 * @return This function returns a pointer to the IRIS memory object
 */
extern iris_mem *iris_data_mem_create_ptr(void *host, size_t size);


/**@brief Frees memory for a DMEM object for all the devices
 *
 * This function Resets a memory object by setting the dirty flag for host
 *
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_clear(iris_mem mem);



/**@brief  UnPin/Unregister a host memory for all the available devices
 *
 * This function un pins a host memory for all the available devices
 *
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_unpin(iris_mem mem);

extern int iris_data_mem_set_pin_flag(bool flag);

/**@brief  Pins a host memory for all the available devices
 *
 * This function pins a host memory for all the available devices
 *
 * @param mem pointer to the memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_pin(iris_mem mem);

/**@brief data memory object update for a task
 *
 * @param mem memory object
 * @param host host pointer to the memory
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_update(iris_mem mem, void *host);

/**@brief data memory object refresh for a task. It will make host copy valid and device copies invalid
 *
 * @param mem memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_refresh(iris_mem mem);

/**@brief data memory object update for a task
 *
 * @param mem memory object
 * @param host_size host size pointer to the array
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_update_host_size(iris_mem mem, size_t *host_size);

/**@brief creates data memory region
 *
 * @param mem pointer to a memory object
 * @param root_mem root memory object
 * @param region index for the region
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_create_region(iris_mem* mem, iris_mem root_mem, int region);

/**@brief creates data memory region
 *
 * @param mem pointer to a memory object
 * @param root_mem root memory object
 * @param region index for the region
 * @return This function returns an IRIS dmem regions structure
 */
extern iris_mem iris_data_mem_create_region_struct(iris_mem root_mem, int region);

/**@brief creates data memory region
 *
 * @param root_mem root memory object
 * @param region index for the region
 * @return This function returns a pointer to the IRIS memory object
 */
extern iris_mem *iris_data_mem_create_region_ptr(iris_mem root_mem, int region);

/**@brief enable decomposition along the outer dimension
 *
 * @param mem memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_enable_outer_dim_regions(iris_mem mem);

/**@brief Creates a memory tile from host memory
 *
 * @param mem memory object
 * @param host host memory pointer
 * @param off host memory pointer
 * @param host_size indexes to specify sizes from host memory
 * @param dev_size indexes to specify sizes from device memory
 * @param elem_size element size
 * @param dim dimension
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_data_mem_create_tile(iris_mem* mem, void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);

/**@brief Creates a memory tile from host memory
 *
 * @param host host memory pointer
 * @param off host memory pointer
 * @param host_size indexes to specify sizes from host memory
 * @param dev_size indexes to specify sizes from device memory
 * @param elem_size element size
 * @param dim dimension
 * @return This function returns a iris_mem object
 */
extern iris_mem iris_data_mem_create_tile_struct(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);

/**@brief Creates a memory tile from host memory
 *
 * @param host host memory pointer
 * @param off host memory pointer
 * @param host_size indexes to specify sizes from host memory
 * @param dev_size indexes to specify sizes from device memory
 * @param elem_size element size
 * @param dim dimension
 * @param type type of memory element
 * @return This function returns a iris_mem object
 */
extern iris_mem iris_data_mem_create_tile_struct_with_type(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type);

/**@brief Creates a memory tile from host memory
 *
 * @param host host memory pointer
 * @param off host memory pointer
 * @param host_size indexes to specify sizes from host memory
 * @param dev_size indexes to specify sizes from device memory
 * @param elem_size element size
 * @param dim dimension
 * @return This function returns a pointer to the IRIS memory object
 */
extern iris_mem *iris_data_mem_create_tile_ptr(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
extern int iris_data_mem_update_bc(iris_mem mem, int bc, int row, int col);
extern int iris_data_mem_get_rr_bc_dev(iris_mem mem);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
extern int iris_data_mem_n_regions(iris_mem brs_mem);

extern unsigned long iris_data_mem_get_region_uid(iris_mem brs_mem, int region);
#endif

/**@brief returns the device pointer for a memory object
*
 * @param mem iris memory object
 * @param device device id
 * @param arch device pointer
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_arch(iris_mem mem, int device, void** arch);

/**@brief returns the device pointer for a memory object
*
 * @param mem iris memory object
 * @param device device id
 * @return This function arch device pointer
 */
extern void *iris_mem_arch_ptr(iris_mem mem, int device);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
//extern int iris_mem_reduce(iris_mem mem, int mode, int type);
#endif

/**@brief releases memory object
*
 * @param mem iris memory object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_mem_release(iris_mem mem);

/**@brief Creates a graph
 *
 * @param graph pointer to the graph
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_create(iris_graph* graph);

/**@brief Creates a graph
 *
 * @return This function returns empty graph
 */
extern iris_graph iris_graph_create_empty();

/**@brief Creates a null graph
 *
 * @param graph pointer to the graph
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_create_null(iris_graph* graph);

/**@brief Is it a null graph
 *
 * @param graph data structure
 * @return This function returns a boolean flag indicating whether graph is null or not 
 */
extern int iris_is_graph_null(iris_graph graph);

/**@brief Frees a graph
 *
 * @param graph graph object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_free(iris_graph graph);

/**@brief submits a graph with given order
*
 * @param graph graph object
 * @param order array of indexes of the tasks in that graph
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_tasks_order(iris_graph brs_graph, int *order);

/**@brief Creates json for a graph
 *
 * @param json file path to json file
 * @param params parameters
 * @param graph a pointer to the graph data structure that has the generated graph
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_create_json(const char* json, void** params, iris_graph* graph);

/**@brief Adds a task to a graph
 *
 * @param graph graph object
 * @param task task object to be added
 * @param device policy/device id
 * @param opt optional parameter for custom policy
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_task(iris_graph graph, iris_task task, int device, const char* opt);

/**@brief Retain a graph object for the next submission
 *
 * @param graph graph object
 * @param flag 0: not retain, 1:retain
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_retain(iris_graph graph, int flag);

/**@brief Releases a graph object
 *
 * @param graph graph object
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_release(iris_graph graph);

/**@brief submits a graph object for execution
 *
 * @param graph graph object
 * @param device policy/device id
 * @param sync 0: non-blocking, 1: blocking
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_submit(iris_graph graph, int device, int sync);

/**@brief submits a graph with order defined
 *
 * @param graph graph object
 * @param order array of task index specifying the order
 * @param device policy/device id
 * @param sync 0: non-blocking, 1: blocking
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_submit_with_order(iris_graph graph, int *order, int device, int sync);

/**@brief submits a graph with order defined and it returns the time
 *
 * @param graph graph object
 * @param order array of task index specifying the order
 * @param time  Time pointer, in which the graph execution time is returned
 * @param device policy/device id
 * @param sync 0: non-blocking, 1: blocking
 * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
 */
extern int iris_graph_submit_with_order_and_time(iris_graph graph, int *order, double *time, int device, int sync);

/**@brief Submit the IRIS graph and returns the time along with state
  *
  * @param graph iris_graph (IRIS Graph) object
  * @param time  Time pointer, in which the graph execution time is returned
  * @param device IRIS device selection policy
  * @param sync 0: non-blocking, 1: blocking
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_graph_submit_with_time(iris_graph graph, double *time, int device, int sync);

/**@brief Wait for the completion of IRIS graph
  *
  * @param graphs iris_graph (IRIS Graph) object
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_graph_wait(iris_graph graph);

/**@brief Wait for the completion of all array of IRIS graphs
  *
  * @param ngraphs Number of graphs
  * @param graphs Array of iris_graph (IRIS Graph) objects
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_graph_wait_all(int ngraphs, iris_graph* graphs);

/**@brief Start recording task graph to generate JSON graph
  *
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_record_start();

/**@brief Stop recording task graph to generate JSON graph
  *
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_record_stop();


/**@brief Returns current time in seconds.
 *
 * Returns current time in seconds.
 * @param time pointer of time
 * @return This function returns current time.
 */
extern int iris_timer_now(double* time);



/**@brief Enables peer to peer transfer
 *
 * This function enables peer to peer transfer
 */
extern void iris_enable_d2d();


/**@brief Disables peer to peer transfer
 *
 * This function disables peer to peer transfer
 */
extern void iris_disable_d2d();

#ifndef DOXYGEN_SHOULD_SKIP_THIS
extern void iris_disable_consistency_check();
extern void iris_enable_consistency_check();
#endif


/**@brief Returns a kernel name
 *
 * This function returns a kernel name
 *
 * @param brs_kernel kernel object
 * @return This function returns name of the kernel.
 */
extern const char *iris_kernel_get_name(iris_kernel brs_kernel);



/**@brief Retruns a task name
 *
 * This function returns a task name
 *
 * @param brs_task task object
 * @return This function returns name of the task.
 */
extern const char *iris_task_get_name(iris_task brs_task);



/**@brief Sets a task name
 *
 * This function Sets a task name
 *
 * @param brs_task task object
 * @param name name of the task
 */
extern void iris_task_set_name(iris_task brs_task, const char *name);


/**@brief Gets dependency counts for a task
 *
 * This function returns dependency count for a task
 *
 * @param brs_task task object
 * @return This function returns dependency count for a task
 */
extern int iris_task_get_dependency_count(iris_task brs_task);


/**@brief Gets all the dependent tasks for a given task
 *
 * This function provide all the dependent tasks for a given task
 *
 * @param brs_task task object
 * @param task a list of dependent task for brs_task
 */
extern void iris_task_get_dependencies(iris_task brs_task, iris_task *tasks);


/**@brief Gets unique ID for a task
 *
 * This function provides IRIS generated unique ID for a given task object
 *
 * @param brs_task task object
 * @return This function returns the unique id
 */
extern unsigned long iris_task_get_uid(iris_task brs_task);


/**@brief Gets unique ID for a kernel
 *
 * This function provides IRIS generated unique ID for a given kernel object
 *
 * @param brs_kernel kernel object
 * @return This function returns the unique id for the kernel
 */
extern unsigned long iris_kernel_get_uid(iris_kernel brs_kernel);


#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**@brief Returns kernel for a task
 *
 * This function returns kernel for a given task
 *
 * @param brs_task kernel object
 * @return This function returns the kernel object extracted from a given task
 */
extern iris_kernel iris_task_get_kernel(iris_task brs_task);
extern int iris_task_kernel_dmem_fetch_order(iris_task brs_task, int *order);
extern int iris_task_disable_consistency(iris_task brs_task);
extern int iris_task_is_cmd_kernel_exists(iris_task brs_task);
extern void *iris_task_get_cmd_kernel(iris_task brs_task);

// Memory member access
extern size_t iris_mem_get_size(iris_mem mem);
extern int iris_mem_get_type(iris_mem mem);
extern int iris_mem_get_uid(iris_mem mem);
extern int iris_get_mem_element_type(iris_mem brs_mem);
extern int iris_mem_is_reset(iris_mem mem);
extern int iris_mem_init_reset(iris_mem brs_mem, int memset_value);
extern int iris_mem_init_reset_assign(iris_mem brs_mem, IRISValue value);
extern int iris_mem_init_reset_arith_seq(iris_mem brs_mem, IRISValue start, IRISValue increment);
extern int iris_mem_init_reset_geom_seq(iris_mem brs_mem, IRISValue start, IRISValue step);
extern int iris_mem_init_reset_random_uniform_seq(iris_mem brs_mem, long long seed, IRISValue min, IRISValue max);
extern int iris_mem_init_reset_random_normal_seq(iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev);
extern int iris_mem_init_reset_random_log_normal_seq(iris_mem brs_mem, long long seed, IRISValue mean, IRISValue stddev);
extern int iris_mem_init_reset_random_uniform_sobol_seq(iris_mem brs_mem, IRISValue min, IRISValue max);
extern int iris_mem_init_reset_random_normal_sobol_seq(iris_mem brs_mem, IRISValue mean, IRISValue stddev);
extern int iris_mem_init_reset_random_log_normal_sobol_seq(iris_mem brs_mem, IRISValue mean, IRISValue stddev);

// DMem Memory member access
extern int iris_dmem_get_dim(iris_mem mem);
extern int iris_dmem_get_elem_size(iris_mem mem);
extern int iris_dmem_get_elem_type(iris_mem mem);
extern size_t *iris_dmem_get_host_size(iris_mem mem);
extern iris_mem iris_get_dmem_for_region(iris_mem dmem_region_obj);
extern int iris_dmem_set_source(iris_mem brs_mem, iris_mem source_mem);

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
extern int iris_get_graph_max_theoretical_parallelism(iris_graph graph);
extern int iris_get_graph_dependency_adj_list(iris_graph brs_graph, int8_t *dep_matrix);

/**@brief Get dependency graph for the given input graph
  *
  * @param brs_graph Input IRIS task graph
  * @param dep_matris A pointer which will be returned with 2D Dependency matrix ( Task x Task)  data
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_get_graph_dependency_adj_matrix(iris_graph brs_graph, int8_t *dep_matrix);

/**@brief Fetch number of formmatted task communication data (Used in Python API)
  *
  * @param brs_graph Input IRIS task graph
  * @return Number of formatted tuple array of communication data (parent-task-id, child-task-id, memory-id, data-size)
  */
extern size_t iris_get_graph_3d_comm_data_size(iris_graph brs_graph);

/**@brief Fetch formmatted task communication data (Used in Python API)
  *
  * @param brs_graph Input IRIS task graph
  * @return Formatted tuple array of communication data (parent-task-id, child-task-id, memory-id, data-size)
  */
extern void *iris_get_graph_3d_comm_data_ptr(iris_graph brs_graph);

/**@brief Fetch formmatted task graph execution schedules (Used in Python API)
  *
  * @param brs_graph Input IRIS task graph
  * @param kernel_profile 1: Returns only kernel profile, 0: Incldues data transfers
  * @return Formatted task execution schedule
  */
extern void *iris_get_graph_tasks_execution_schedule(iris_graph brs_graph, int kernel_profile);

/**@brief Fetch number of task graph execution schedules (Used in Python API)
  *
  * @param brs_graph Input IRIS task graph
  * @return Number of task execution schedule count
  */
extern size_t iris_get_graph_tasks_execution_schedule_count(iris_graph brs_graph);

/**@brief Fetch formatted data objects execution schedules after execution of task graph (Used in Python API)
  *
  * @param brs_graph Input IRIS graph
  * @return Formatted data objects scheduled
  */
extern void *iris_get_graph_dataobjects_execution_schedule(iris_graph brs_graph);

/**@brief Fetch number of data objects execution schedules after execution of task graph (Used in Python API)
  *
  * @param brs_graph Input IRIS graph
  * @return Number of data objects scheduled
  */
extern size_t iris_get_graph_dataobjects_execution_schedule_count(iris_graph brs_graph);

/**@brief Fetch a tuple array of communication data (Used in Python API)
  *
  * @param brs_graph Input IRIS graph
  * @param comm_data A tuple array of formatted communication data
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_get_graph_3d_comm_data(iris_graph brs_graph, void *comm_data);

/**@brief Fetch 2D communication data size matrix for the input IRIS graph
  *
  * @param brs_graph IRIS graph object
  * @param size_data A 2-D matrix to return with task to task communication data size
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_get_graph_2d_comm_adj_matrix(iris_graph brs_graph, size_t *size_data);

/**@brief Calibrate computation cost matrix for the given input graph
  *
  * @param brs_graph Input IRIS Graph object
  * @param comp_data A 2D array object to hold computation cost matrix (Task x Device)
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_calibrate_compute_cost_adj_matrix(iris_graph brs_graph, double *comp_data);

/**@brief Calibrate computation cost matrix for the given input graph only for device types
  *
  * @param brs_graph Input IRIS Graph object
  * @param comp_data A 2D array object to hold computation cost matrix (Task x Devicetype )
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_calibrate_compute_cost_adj_matrix_only_for_types(iris_graph brs_graph, double *comp_data);

/**@brief Calibrate communication cost matrix for the given input data size.
  *
  * @param data Returns communication cost in 2D (device x device) array
  * @param data_size Input data size for calibration
  * @param iterations Number of iterations used in the calibration of communication time for each combination of memory object x device x devicea
  * @param pin_memory_flag Enable/Disable PIN memory
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_calibrate_communication_cost(double *data, size_t data_size, int iterations, int pin_memory_flag);

/**@brief Calibrate communication time for all data memory objects of the graph
  *
  * @param brs_graph IRIS graph object
  * @param comp_time Returns A 3-dimensional (Memory Object x Device x Device) communication time in the comp_time array pointer
  * @param mem_ids Returns the IRIS memory object IDs in the mem_ids array pointer
  * @param iterations Number of iterations used in the calibration of communication time for each combination of memory object x device x devicea
  * @param pin_memory_flag Enable/Disable PIN memory
  * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
  */
extern int iris_get_graph_3d_comm_time(iris_graph brs_graph, double *comm_time, int *mem_ids, int iterations, int pin_memory_flag);
#endif // DOXYGEN_SHOULD_SKIP_THIS

/**@brief Count number of memory objects used in the IRIS graph
  *
  * @param brs_graph IRIS graph object
  * @return return number of memory objects used in iris_graph
  */
extern size_t iris_count_mems(iris_graph brs_graph);

/**@brief Free memory location using IRIS api
  *
  * @param ptr Input pointer of the object
  */
extern void iris_free_array(void *ptr);



/**@brief Mallocs int8_t type array for a given size with a given initial value
 *
 * This function mallocs int8_t type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern int8_t *iris_allocate_array_int8_t(int SIZE, int8_t init);



/**@brief Mallocs int16_t type array for a given size with a given initial value
 *
 * This function mallocs int16_t type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern int16_t *iris_allocate_array_int16_t(int SIZE, int16_t init);


/**@brief Mallocs int32_t type array for a given size with a given initial value
 *
 * This function mallocs int32_t type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern int32_t *iris_allocate_array_int32_t(int SIZE, int32_t init);


/**@brief Mallocs int64_t type array for a given size with a given initial value
 *
 * This function mallocs int64_t type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern int64_t *iris_allocate_array_int64_t(int SIZE, int64_t init);


/**@brief Mallocs size_t type array for a given size with a given initial value
 *
 * This function mallocs size_t type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern size_t *iris_allocate_array_size_t(int SIZE, size_t init);


/**@brief Mallocs float type array for a given size with a given initial value
 *
 * This function mallocs float type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern float *iris_allocate_array_float(int SIZE, float init);


/**@brief Mallocs double type array for a given size with a given initial value
 *
 * This function mallocs double type array for a given size with a given initial value
 *
 * @param SIZE size of the array
 * @param init initialization value
 * @return This function returns the pointer to the newly allocated array
 */
extern double *iris_allocate_array_double(int SIZE, double init);


/**@brief Mallocs int8_t type array for a given size with a random value
 *
 * This function mallocs int8_t type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern int8_t *iris_allocate_random_array_int8_t(int SIZE);


/**@brief Mallocs int16_t type array for a given size with a random value
 *
 * This function mallocs int16_t type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern int16_t *iris_allocate_random_array_int16_t(int SIZE);


/**@brief Mallocs int32_t type array for a given size with a random value
 *
 * This function mallocs int32_t type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern int32_t *iris_allocate_random_array_int32_t(int SIZE);


/**@brief Mallocs int64_t type array for a given size with a random value
 *
 * This function mallocs int64_t type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern int64_t *iris_allocate_random_array_int64_t(int SIZE);


/**@brief Mallocs size_t type array for a given size with a random value
 *
 * This function mallocs size_t type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern size_t *iris_allocate_random_array_size_t(int SIZE);


/**@brief Mallocs float type array for a given size with a random value
 *
 * This function mallocs float type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern float *iris_allocate_random_array_float(int SIZE);


/**@brief Mallocs double type array for a given size with a random value
 *
 * This function mallocs double type array for a given size with a random value
 *
 * @param SIZE size of the array
 * @return This function returns the pointer to the newly allocated array
 */
extern double *iris_allocate_random_array_double(int SIZE);


/**@brief Prints a full matrix data structure of double type
 *
 * This function prints a full matrix data structure of double type of M rows and N columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 */
extern void iris_print_matrix_full_double(double *data, int M, int N, const char *description);


/**@brief Prints a matrix data structure of double type for a given limit of rows and columns
 *
 * This function prints a matrix data structure of double type for limited rows and columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 * @param limit printing limit for rows and columns
 */
extern void iris_print_matrix_limited_double(double *data, int M, int N, const char *description, int limit);


/**@brief Prints a full matrix data structure of float type
 *
 * This function prints a full matrix data structure of float type of M rows and N columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 */
extern void iris_print_matrix_full_float(float *data, int M, int N, const char *description);


/**@brief Prints a matrix data structure of float type for a given limit of rows and columns
 *
 * This function prints a matrix data structure of float type for limited rows and columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 * @param limit printing limit for rows and columns
 */
extern void iris_print_matrix_limited_float(float *data, int M, int N, const char *description, int limit);


/**@brief Prints a full matrix data structure of int64_t type
 *
 * This function prints a full matrix data structure of int64_t type of M rows and N columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 */
extern void iris_print_matrix_full_int64_t(int64_t *data, int M, int N, const char *description);


/**@brief Prints a matrix data structure of int64_t type for a given limit of rows and columns
 *
 * This function prints a matrix data structure of int64_t type for limited rows and columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 * @param limit printing limit for rows and columns
 */
extern void iris_print_matrix_limited_int64_t(int64_t *data, int M, int N, const char *description, int limit);


/**@brief Prints a full matrix data structure of int32_t type
 *
 * This function prints a full matrix data structure of int32_t type of M rows and N columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 */
extern void iris_print_matrix_full_int32_t(int32_t *data, int M, int N, const char *description);


/**@brief Prints a matrix data structure of int32_t type for a given limit of rows and columns
 *
 * This function prints a matrix data structure of int32_t type for limited rows and columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 * @param limit printing limit for rows and columns
 */
extern void iris_print_matrix_limited_int32_t(int32_t *data, int M, int N, const char *description, int limit);


/**@brief Prints a full matrix data structure of int16_t type
 *
 * This function prints a full matrix data structure of int16_t type of M rows and N columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 */
extern void iris_print_matrix_full_int16_t(int16_t *data, int M, int N, const char *description);


/**@brief Prints a matrix data structure of int16_t type for a given limit of rows and columns
 *
 * This function prints a matrix data structure of int16_t type for limited rows and columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 * @param limit printing limit for rows and columns
 */
extern void iris_print_matrix_limited_int16_t(int16_t *data, int M, int N, const char *description, int limit);


/**@brief Prints a full matrix data structure of int8_t type
 *
 * This function prints a full matrix data structure of int8_t type of M rows and N columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 */
extern void iris_print_matrix_full_int8_t(int8_t *data, int M, int N, const char *description);


/**@brief Prints a matrix data structure of int8_t type for a given limit of rows and columns
 *
 * This function prints a matrix data structure of int8_t type for limited rows and columns with a provided description
 *
 * @param data pointer to the matrix
 * @param M rows of the matrix
 * @param N columns of the matrix
 * @param description input string for description
 * @param limit printing limit for rows and columns
 */
extern void iris_print_matrix_limited_int8_t(int8_t *data, int M, int N, const char *description, int limit);

/**
  * This function prints the IRIS logo
  * @param string pointer
  * @return Void
  */
extern int iris_logo();

/**
  * This function prints string on console
  * @param string pointer
  * @return Void
  */
extern void iris_println(const char *s);

/**
  * This function returns device context
  * @param Device number
  * @return Context pointer
  */
extern void *iris_dev_get_ctx(int device);

/**
  * This function returns device context
  * @param Device number
  * @param index of stream 
  * @return Stream pointer
  */
extern void *iris_dev_get_stream(int device, int index);

/* Run HPL Mapping algorithm*/
extern void iris_run_hpl_mapping(iris_graph graph);

/* Read IRIS_* bool environment variable
 * @param env_name Environment variable
 * @return This function returns boolean flag
 */
extern int iris_read_bool_env(const char *env_name);

/* Read IRIS_* int environment variable
 * @param env_name Environment variable
 * @return This function returns int flag
 */
extern int iris_read_int_env(const char *env_name);

typedef int32_t (*julia_policy_t)(iris_task task, const char *policy_name, iris_device *devs, int32_t ndevs, int32_t *out_dev); 
// Define a type for the Julia kernel launch function call pointer
typedef int32_t (*julia_kernel_t)(unsigned long task_id, int32_t julia_kernel_type, int32_t target, int32_t devno, void *ctx, int async, int32_t stream_index, void **stream, int32_t nstreams, int32_t *args, void **values, size_t *param_size, size_t *param_dim_size, int32_t nparams, size_t *threads, size_t *blocks, int dim, const char *kernel_name);

/* API to initialize Julia interfacea
 * @param kernel_launch_func Kernel launch Julia function 
 * @param decoupled_init flag to enable decoupled init of worker, devices and scheduler
 * @return This function returns int flag
 */
extern int iris_julia_init(void *julia_launch_func, int decoupled_init);
extern int iris_julia_policy_init(void *julia_policy_func);

/* API to return the Julia kernel launch function
 * @return This function returns Julia kernel launch function pointer
 */
extern julia_kernel_t iris_get_julia_launch_func();

/* API to return the Julia policy launch function
 * @return This function returns Julia policy launch function pointer
 */
extern julia_policy_t iris_get_julia_policy_func();

/* API to return the the status of whether auto parallel macro is on
 * @return This function returns whether AUTO_PAR macro is set or not
 */
extern int iris_is_enabled_auto_par();

extern int iris_vendor_kernel_launch(int dev, void *kernel, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int shared_mem_bytes, void *stream, void **params);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* IRIS_INCLUDE_IRIS_IRIS_RUNTIME_H */


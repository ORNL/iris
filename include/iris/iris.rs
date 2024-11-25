//use libc::{c_char, c_int, c_void, c_ulong, c_double, size_t, malloc, free};
use libc::{size_t, malloc, free};
use std::os::raw::{c_char, c_int, c_uint, c_ulong, c_void, c_float, c_double};
//use core::ffi::{size_t};

pub const IRIS_SUCCESS: i32 = 0;
pub const IRIS_ERROR: i32 = -1;
pub const IRIS_MAX_NPLATFORMS: u32 = 32;
pub const IRIS_MAX_NDEVS: u32 = (1 << 5) - 1;
pub const IRIS_MAX_KERNEL_NARGS: u32 = 64;
pub const IRIS_MAX_DEVICE_NSTREAMS: u32 = 9;
pub const IRIS_MAX_DEVICE_NCOPY_STREAMS: u32 = 3;

pub const iris_default: u32 = 1 << 5;
pub const iris_cpu: u32 = 1 << 6;
pub const iris_nvidia: u32 = 1 << 7;
pub const iris_amd: u32 = 1 << 8;
pub const iris_gpu_intel: u32 = 1 << 9;

pub const iris_gpu: u32 = iris_nvidia | iris_amd | iris_gpu_intel;
pub const iris_phi: u32 = 1 << 10;
pub const iris_fpga: u32 = 1 << 11;
pub const iris_hexagon: u32 = 1 << 12;
pub const iris_dsp: u32 = iris_hexagon;
pub const iris_roundrobin: u32 = 1 << 18;
pub const iris_depend: u32 = 1 << 19;
pub const iris_data: u32 = 1 << 20;
pub const iris_profile: u32 = 1 << 21;
pub const iris_random: u32 = 1 << 22;
pub const iris_pending: u32 = 1 << 23;
pub const iris_sdq: u32 = 1 << 24;
pub const iris_ftf: u32 = 1 << 25;
pub const iris_all: u32 = 1 << 25;
pub const iris_ocl: u32 = 1 << 26;
pub const iris_custom: u32 = 1 << 27;


pub const iris_r: i32 = -1;
pub const iris_w: i32 = -2;
pub const iris_rw: i32 = -3;
pub const iris_xr: i32 = -4;
pub const iris_xw: i32 = -5;
pub const iris_xrw: i32 = -6;

pub const iris_dt_h2d: u32 = 1;
pub const iris_dt_d2o: u32 = 2;
pub const iris_dt_o2d: u32 = 3;
pub const iris_dt_d2h: u32 = 4;
pub const iris_dt_d2d: u32 = 5;
pub const iris_dt_d2h_h2d: u32 = 6;
pub const iris_dt_error: u32 = 0;

pub const iris_int: u32 = (1 << 1) << 16;
pub const iris_uint: u32 = (1 << 1) << 16;
pub const iris_float: u32 = (1 << 2) << 16;
pub const iris_double: u32 = (1 << 3) << 16;
pub const iris_char: u32 = (1 << 4) << 16;
pub const iris_int8: u32 = (1 << 4) << 16;
pub const iris_uint8: u32 = (1 << 4) << 16;
pub const iris_int16: u32 = (1 << 5) << 16;
pub const iris_uint16: u32 = (1 << 5) << 16;
pub const iris_int32: u32 = (1 << 6) << 16;
pub const iris_uint32: u32 = (1 << 6) << 16;
pub const iris_int64: u32 = (1 << 7) << 16;
pub const iris_uint64: u32 = (1 << 7) << 16;
pub const iris_long: u32 = (1 << 8) << 16;
pub const iris_unsigned_long: u32 = (1 << 8) << 16;

pub const iris_normal: u32 = 1 << 10;
pub const iris_reduction: u32 = 1 << 11;
pub const iris_sum: u32 = (1 << 12) | iris_reduction;
pub const iris_max: u32 = (1 << 13) | iris_reduction;
pub const iris_min: u32 = (1 << 14) | iris_reduction;

pub const iris_platform: u32 = 0x1001;
pub const iris_vendor: u32 = 0x1002;
pub const iris_name: u32 = 0x1003;
pub const iris_type: u32 = 0x1004;
pub const iris_backend: u32 = 0x1005;

pub const iris_ncmds: u32 = 1;
pub const iris_ncmds_kernel: u32 = 2;
pub const iris_ncmds_memcpy: u32 = 3;
pub const iris_cmds: u32 = 4;
pub const iris_task_time_submit: u32 = 5;
pub const iris_task_time_start: u32 = 6;
pub const iris_task_time_end: u32 = 7;

pub const iris_event_wait_default: u32 = 0;
pub const iris_event_wait_external: u32 = 1;

pub const iris_event_default: u32 = 0;
pub const iris_event_blocking_sync: u32 = 1;
pub const iris_event_disable_timing: u32 = 2;
pub const iris_event_interprocess: u32 = 3;

pub const iris_stream_default: u32 = 0;
pub const iris_stream_non_blocking: u32 = 1;

#[repr(i32)]
pub enum StreamPolicy {
    StreamPolicyDefault = 0,
    StreamPolicySameForTask = 1,
    StreamPolicyGiveAllStreamsToKernel = 2,
}

#[repr(i32)]
pub enum DeviceModel {
    IrisCuda = 1,
    IrisHip = 3,
    IrisLevelZero = 4,
    IrisOpenCl = 5,
    IrisOpenMp = 6,
    IrisModelAll = (1 << 25), // Same as iris_all
}

#[repr(C)]
pub struct iris_task {
    pub class_obj: *mut c_void,  // Rust equivalent of `void *`
    pub uid: u64,
}

#[repr(C)]
pub struct iris_kernel {
    pub class_obj: *mut c_void,
    pub uid: u64,
}

#[repr(C)]
pub struct iris_mem {
    pub class_obj: *mut c_void,
    pub uid: u64,
}

#[repr(C)]
pub struct iris_graph {
    pub class_obj: *mut c_void,
    pub uid: u64,
}

pub type iris_selector_kernel = extern "C" fn(task: *mut iris_task, params: *mut c_void, params_size: size_t) -> i32;
pub type iris_host_python_task = extern "C" fn(task: *mut iris_task, params_id: i64) -> i32;
pub type iris_host_task = extern "C" fn(task: *mut iris_task, params: *mut c_void) -> i32;
pub type command_handler = extern "C" fn(params: *mut c_void, device: *mut c_void) -> i32;
pub type hook_task = extern "C" fn(task: *mut iris_task) -> i32;
pub type hook_command = extern "C" fn(command: *mut c_void) -> i32;

extern "C" {
    pub fn iris_init(argc: *mut i32, argv: *mut *mut *mut char, sync: i32) -> i32;
    pub fn iris_finalize() -> i32;
    pub fn iris_error_count() -> i32;
    pub fn iris_synchronize() -> i32;
    pub fn iris_overview();
    pub fn iris_platform_count(nplatforms: *mut i32) -> i32;
    pub fn iris_platform_info(platform: i32, param: i32, value: *mut c_void, size: *mut usize) -> i32;
    // Functions related to setting and getting stream policies
    pub fn iris_set_stream_policy(policy: StreamPolicy) -> i32;
    pub fn iris_set_asynchronous(flag: i32) -> i32;
    pub fn iris_set_shared_memory_model(flag: i32) -> i32;
    pub fn iris_mem_enable_usm(mem: *mut iris_mem, device_type: DeviceModel) -> i32;
    pub fn iris_mem_disable_usm(mem: *mut iris_mem, device_type: DeviceModel) -> i32;
    pub fn iris_set_enable_profiler(flag: i32);

    // Device and memory related functions
    pub fn iris_device_count(ndevs: *mut i32) -> i32;
    pub fn iris_ndevices() -> i32;
    pub fn iris_set_nstreams(n: i32) -> i32;
    pub fn iris_set_ncopy_streams(n: i32) -> i32;
    pub fn iris_nstreams() -> i32;
    pub fn iris_ncopy_streams() -> i32;
    pub fn iris_device_info(device: i32, param: i32, value: *mut c_void, size: *mut usize) -> i32;
    pub fn iris_device_set_default(device: i32) -> i32;
    pub fn iris_device_get_default(device: *mut i32) -> i32;
    pub fn iris_device_synchronize(ndevs: i32, devices: *const i32) -> i32;

    // Kernel and task management
    pub fn iris_kernel_create(name: *const c_char, kernel: *mut *mut iris_kernel) -> i32;
    pub fn iris_kernel_get(name: *const c_char, kernel: *mut *mut iris_kernel) -> i32;
    pub fn iris_kernel_setarg(kernel: *mut iris_kernel, idx: c_uint, size: usize, value: *const c_void) -> i32;
    pub fn iris_kernel_setmem(kernel: *mut iris_kernel, idx: c_uint, mem: *mut iris_mem, mode: usize) -> i32;
    pub fn iris_kernel_setmem_off(kernel: *mut iris_kernel, idx: c_uint, mem: *mut iris_mem, off: usize, mode: usize) -> i32;
    pub fn iris_kernel_release(kernel: *mut iris_kernel) -> i32;

    // Task creation and management
    pub fn iris_task_create(task: *mut *mut iris_task) -> i32;
    pub fn iris_task_create_perm(task: *mut *mut iris_task) -> i32;
    pub fn iris_task_create_name(name: *const c_char, task: *mut *mut iris_task) -> i32;
    pub fn iris_task_depend(task: *mut iris_task, ntasks: i32, tasks: *const *mut iris_task) -> i32;
    pub fn iris_task_malloc(task: *mut iris_task, mem: *mut iris_mem) -> i32;
    pub fn iris_task_cmd_reset_mem(task: *mut iris_task, mem: *mut iris_mem, reset: u8) -> i32;
    pub fn iris_task_set_stream_policy(task: *mut iris_task, policy: StreamPolicy) -> i32;

    pub fn iris_task_disable_asynchronous(task: *mut iris_task);
    pub fn iris_task_get_metadata(task: *mut iris_task, index: i32) -> i32;
    pub fn iris_task_set_metadata(task: *mut iris_task, index: i32, metadata: i32) -> i32;
    pub fn iris_task_h2broadcast(task: *mut iris_task, mem: *mut iris_mem, off: size_t, size: size_t, host: *mut c_void) -> i32;
    pub fn iris_task_h2broadcast_offsets(task: *mut iris_task, mem: *mut iris_mem, off: *mut size_t, host_sizes: *mut size_t, dev_sizes: *mut size_t, elem_size: size_t, dim: i32, host: *mut c_void) -> i32;
    pub fn iris_task_h2broadcast_full(task: *mut iris_task, mem: *mut iris_mem, host: *mut c_void) -> i32;
    pub fn iris_task_d2d(task: *mut iris_task, mem: *mut iris_mem, off: size_t, size: size_t, host: *mut c_void, src_dev: i32) -> i32;
    pub fn iris_task_h2d(task: *mut iris_task, mem: *mut iris_mem, off: size_t, size: size_t, host: *mut c_void) -> i32;
    pub fn iris_task_h2d_offsets(task: *mut iris_task, mem: *mut iris_mem, off: *mut size_t, host_sizes: *mut size_t, dev_sizes: *mut size_t, elem_size: size_t, dim: i32, host: *mut c_void) -> i32;
    pub fn iris_task_d2h(task: *mut iris_task, mem: *mut iris_mem, off: size_t, size: size_t, host: *mut c_void) -> i32;
    pub fn iris_task_d2h_offsets(task: *mut iris_task, mem: *mut iris_mem, off: *mut size_t, host_sizes: *mut size_t, dev_sizes: *mut size_t, elem_size: size_t, dim: i32, host: *mut c_void) -> i32;
    pub fn iris_task_dmem_flush_out(task: *mut iris_task, mem: *mut iris_mem) -> i32;
    pub fn iris_task_h2d_full(task: *mut iris_task, mem: *mut iris_mem, host: *mut c_void) -> i32;
    pub fn iris_task_d2h_full(task: *mut iris_task, mem: *mut iris_mem, host: *mut c_void) -> i32;
    pub fn iris_task_kernel_object(task: *mut iris_task, kernel: *mut iris_kernel, dim: i32, off: *mut size_t, gws: *mut size_t, lws: *mut size_t) -> i32;
    pub fn iris_task_kernel(
        task: *mut iris_task,
        kernel: *const c_char,
        dim: i32,
        off: *mut size_t,
        gws: *mut size_t,
        lws: *mut size_t,
        nparams: i32,
        params: *mut *mut c_void,
        params_info: *mut i32,
    ) -> i32;

    pub fn iris_task_kernel_v2(
        task: *mut iris_task,
        kernel: *const c_char,
        dim: i32,
        off: *mut size_t,
        gws: *mut size_t,
        lws: *mut size_t,
        nparams: i32,
        params: *mut *mut c_void,
        params_off: *mut size_t,
        params_info: *mut i32,
    ) -> i32;

    pub fn iris_task_kernel_v3(
        task: *mut iris_task,
        kernel: *const c_char,
        dim: i32,
        off: *mut size_t,
        gws: *mut size_t,
        lws: *mut size_t,
        nparams: i32,
        params: *mut *mut c_void,
        params_off: *mut size_t,
        params_info: *mut i32,
        memranges: *mut size_t,
    ) -> i32;

    pub fn iris_task_kernel_selector(
        task: *mut iris_task,
        func: iris_selector_kernel,
        params: *mut c_void,
        params_size: size_t,
    ) -> i32;
    pub fn iris_task_submit(
        task: *mut iris_task,
        device: i32,
        opt: *const c_char,
        sync: i32
    ) -> i32;
    pub fn iris_task_kernel_launch_disabled(task: *mut iris_task, flag: i32) -> i32;
    pub fn iris_task_python_host(task: *mut iris_task, func: iris_host_python_task, params_id: i64) -> i32;
    pub fn iris_task_host(task: *mut iris_task, func: iris_host_task, params: *mut c_void) -> i32;
    pub fn iris_task_custom(task: *mut iris_task, tag: i32, params: *mut c_void, params_size: size_t) -> i32;
    pub fn iris_task_set_policy(task: *mut iris_task, device: i32) -> i32;
    pub fn iris_task_wait(task: *mut iris_task) -> i32;
    pub fn iris_task_wait_all(ntasks: i32, tasks: *const *mut iris_task) -> i32;
    pub fn iris_task_add_subtask(task: *mut iris_task, subtask: *mut iris_task) -> i32;
    pub fn iris_task_kernel_cmd_only(task: *mut iris_task) -> i32;
    pub fn iris_task_release(task: *mut iris_task) -> i32;
    pub fn iris_task_release_mem(task: *mut iris_task, mem: *mut iris_mem) -> i32;
    pub fn iris_params_map(task: *mut iris_task, params_map: *mut i32) -> i32;
    pub fn iris_task_info(task: *mut iris_task, param: i32, value: *mut c_void, size: *mut size_t) -> i32;
    pub fn iris_register_pin_memory(host: *mut c_void, size: size_t) -> i32;
    pub fn iris_mem_create(size: size_t, mem: *mut *mut iris_mem) -> i32;
    pub fn iris_data_mem_init_reset(mem: *mut iris_mem, reset: i32) -> i32;
    pub fn iris_data_mem_create(mem: *mut *mut iris_mem, host: *mut c_void, size: size_t) -> i32;
    pub fn iris_data_mem_clear(mem: *mut iris_mem) -> i32;
    pub fn iris_data_mem_pin(mem: *mut iris_mem) -> i32;
    pub fn iris_data_mem_update(mem: *mut iris_mem, host: *mut c_void) -> i32;
    pub fn iris_data_mem_create_region(mem: *mut *mut iris_mem, root_mem: *mut iris_mem, region: i32) -> i32;
    pub fn iris_data_mem_enable_outer_dim_regions(mem: *mut iris_mem) -> i32;
    pub fn iris_data_mem_create_tile(
        mem: *mut *mut iris_mem,
        host: *mut c_void,
        off: *mut size_t,
        host_size: *mut size_t,
        dev_size: *mut size_t,
        elem_size: size_t,
        dim: i32
    ) -> i32;
    pub fn iris_data_mem_n_regions(mem: *mut iris_mem) -> i32;
    pub fn iris_data_mem_get_region_uid(mem: *mut iris_mem, region: i32) -> c_ulong;
    pub fn iris_mem_arch(mem: *mut iris_mem, device: i32, arch: *mut *mut c_void) -> i32;
    pub fn iris_mem_reduce(mem: *mut iris_mem, mode: i32, type_: i32) -> i32;
    pub fn iris_mem_release(mem: *mut iris_mem) -> i32;

    pub fn iris_graph_create(graph: *mut *mut iris_graph) -> i32;
    pub fn iris_graph_create_null(graph: *mut *mut iris_graph) -> i32;
    pub fn iris_graph_free(graph: *mut iris_graph) -> i32;
    pub fn iris_graph_tasks_order(graph: *mut iris_graph, order: *mut i32) -> i32;
    pub fn iris_graph_create_json(json: *const c_char, params: *mut *mut c_void, graph: *mut *mut iris_graph) -> i32;
    pub fn iris_graph_task(graph: *mut iris_graph, task: *mut iris_task, device: i32, opt: *const c_char) -> i32;
    pub fn iris_graph_retain(graph: *mut iris_graph, flag: bool) -> i32;
    pub fn iris_graph_release(graph: *mut iris_graph) -> i32;
    pub fn iris_graph_submit(graph: *mut iris_graph, device: i32, sync: i32) -> i32;
    pub fn iris_graph_submit_with_order(graph: *mut iris_graph, order: *mut i32, device: i32, sync: i32) -> i32;
    pub fn iris_graph_submit_with_order_and_time(graph: *mut iris_graph, order: *mut i32, time: *mut c_double, device: i32, sync: i32) -> i32;
    pub fn iris_graph_submit_with_time(graph: *mut iris_graph, time: *mut c_double, device: i32, sync: i32) -> i32;
    pub fn iris_graph_wait(graph: *mut iris_graph) -> i32;
    pub fn iris_graph_wait_all(ngraphs: i32, graphs: *mut *mut iris_graph) -> i32;

    pub fn iris_record_start() -> i32;
    pub fn iris_record_stop() -> i32;
    pub fn iris_timer_now(time: *mut c_double) -> i32;
    pub fn iris_enable_d2d();
    pub fn iris_disable_d2d();
    
    pub fn iris_disable_consistency_check();
    pub fn iris_enable_consistency_check();
    pub fn iris_kernel_get_name(kernel: *mut iris_kernel) -> *mut c_char;
    pub fn iris_task_get_name(task: *mut iris_task) -> *mut c_char;
    pub fn iris_task_set_name(task: *mut iris_task, name: *const c_char);
    pub fn iris_task_get_dependency_count(task: *mut iris_task) -> i32;
    pub fn iris_task_get_dependencies(task: *mut iris_task, tasks: *mut *mut iris_task);
    pub fn iris_task_get_uid(task: *mut iris_task) -> c_ulong;
    pub fn iris_kernel_get_uid(kernel: *mut iris_kernel) -> c_ulong;
    pub fn iris_task_get_kernel(task: *mut iris_task) -> *mut iris_kernel;
    pub fn iris_task_kernel_dmem_fetch_order(task: *mut iris_task, order: *mut i32) -> i32;
    pub fn iris_task_disable_consistency(task: *mut iris_task) -> i32;
    pub fn iris_task_is_cmd_kernel_exists(task: *mut iris_task) -> i32;
    pub fn iris_task_get_cmd_kernel(task: *mut iris_task) -> *mut c_void;

    pub fn iris_mem_get_size(mem: *mut iris_mem) -> size_t;
    pub fn iris_mem_get_type(mem: *mut iris_mem) -> i32;
    pub fn iris_mem_get_uid(mem: *mut iris_mem) -> i32;
    pub fn iris_mem_is_reset(mem: *mut iris_mem) -> i32;
    pub fn iris_get_dmem_for_region(dmem_region_obj: *mut iris_mem) -> *mut iris_mem;

    pub fn iris_cmd_kernel_get_nargs(cmd: *mut c_void) -> i32;
    pub fn iris_cmd_kernel_get_arg_is_mem(cmd: *mut c_void, index: i32) -> i32;
    pub fn iris_cmd_kernel_get_arg_size(cmd: *mut c_void, index: i32) -> size_t;
    pub fn iris_cmd_kernel_get_arg_value(cmd: *mut c_void, index: i32) -> *mut c_void;
    pub fn iris_cmd_kernel_get_arg_mem(cmd: *mut c_void, index: i32) -> *mut iris_mem;
    pub fn iris_cmd_kernel_get_arg_mem_off(cmd: *mut c_void, index: i32) -> size_t;
    pub fn iris_cmd_kernel_get_arg_mem_size(cmd: *mut c_void, index: i32) -> size_t;
    pub fn iris_cmd_kernel_get_arg_off(cmd: *mut c_void, index: i32) -> size_t;
    pub fn iris_cmd_kernel_get_arg_mode(cmd: *mut c_void, index: i32) -> i32;

    // Graph
    pub fn iris_graph_enable_mem_profiling(graph: *mut iris_graph) -> i32;
    pub fn iris_graph_reset_memories(graph: *mut iris_graph) -> i32;
    pub fn iris_graph_get_tasks(graph: *mut iris_graph, tasks: *mut *mut iris_task) -> i32;
    pub fn iris_graph_tasks_count(graph: *mut iris_graph) -> i32;
    pub fn iris_get_graph_max_theoretical_parallelism(graph: *mut iris_graph) -> i32;
    pub fn iris_get_graph_dependency_adj_list(graph: *mut iris_graph, dep_matrix: *mut i8) -> i32;
    pub fn iris_get_graph_dependency_adj_matrix(graph: *mut iris_graph, dep_matrix: *mut i8) -> i32;
    pub fn iris_get_graph_3d_comm_data_size(graph: *mut iris_graph) -> size_t;
    pub fn iris_get_graph_3d_comm_data_ptr(graph: *mut iris_graph) -> *mut c_void;
    pub fn iris_get_graph_tasks_execution_schedule(graph: *mut iris_graph, kernel_profile: i32) -> *mut c_void;
    pub fn iris_get_graph_tasks_execution_schedule_count(graph: *mut iris_graph) -> size_t;
    pub fn iris_get_graph_dataobjects_execution_schedule(graph: *mut iris_graph) -> *mut c_void;
    pub fn iris_get_graph_dataobjects_execution_schedule_count(graph: *mut iris_graph) -> size_t;
    pub fn iris_get_graph_3d_comm_data(graph: *mut iris_graph, comm_data: *mut c_void) -> i32;
    pub fn iris_get_graph_2d_comm_adj_matrix(graph: *mut iris_graph, size_data: *mut size_t) -> i32;
    pub fn iris_calibrate_compute_cost_adj_matrix(graph: *mut iris_graph, comp_data: *mut c_double) -> i32;
    pub fn iris_calibrate_compute_cost_adj_matrix_only_for_types(graph: *mut iris_graph, comp_data: *mut c_double) -> i32;
    pub fn iris_calibrate_communication_cost(data: *mut c_double, data_size: size_t, iterations: i32, pin_memory_flag: i32) -> i32;
    pub fn iris_get_graph_3d_comm_time(graph: *mut iris_graph, comm_time: *mut c_double, mem_ids: *mut i32, iterations: i32, pin_memory_flag: i32) -> i32;
    pub fn iris_count_mems(graph: *mut iris_graph) -> size_t;
    pub fn iris_free_array(ptr: *mut c_void);

    // Utilities
    pub fn iris_allocate_array_int8_t(size: i32, init: i8) -> *mut i8;
    pub fn iris_allocate_array_int16_t(size: i32, init: i16) -> *mut i16;
    pub fn iris_allocate_array_int32_t(size: i32, init: i32) -> *mut i32;
    pub fn iris_allocate_array_int64_t(size: i32, init: i64) -> *mut i64;
    pub fn iris_allocate_array_size_t(size: i32, init: size_t) -> *mut size_t;
    pub fn iris_allocate_array_float(size: i32, init: c_float) -> *mut c_float;
    pub fn iris_allocate_array_double(size: i32, init: c_double) -> *mut c_double;
    pub fn iris_allocate_random_array_int8_t(size: i32) -> *mut i8;
    pub fn iris_allocate_random_array_int16_t(size: i32) -> *mut i16;
    pub fn iris_allocate_random_array_int32_t(size: i32) -> *mut i32;
    pub fn iris_allocate_random_array_int64_t(size: i32) -> *mut i64;
    pub fn iris_allocate_random_array_size_t(size: i32) -> *mut size_t;
    pub fn iris_allocate_random_array_float(size: i32) -> *mut c_float;
    pub fn iris_allocate_random_array_double(size: i32) -> *mut c_double;
    pub fn iris_print_matrix_full_double(data: *const c_double, m: i32, n: i32, description: *const c_char);
    pub fn iris_print_matrix_limited_double(data: *const c_double, m: i32, n: i32, description: *const c_char, limit: i32);
    pub fn iris_print_matrix_full_float(data: *const c_float, m: i32, n: i32, description: *const c_char);
    pub fn iris_print_matrix_limited_float(data: *const c_float, m: i32, n: i32, description: *const c_char, limit: i32);
    pub fn iris_print_matrix_full_int64_t(data: *const i64, m: i32, n: i32, description: *const c_char);
    pub fn iris_print_matrix_limited_int64_t(data: *const i64, m: i32, n: i32, description: *const c_char, limit: i32);
    pub fn iris_print_matrix_full_int32_t(data: *const i32, m: i32, n: i32, description: *const c_char);
    pub fn iris_print_matrix_limited_int32_t(data: *const i32, m: i32, n: i32, description: *const c_char, limit: i32);
    pub fn iris_print_matrix_full_int16_t(data: *const i16, m: i32, n: i32, description: *const c_char);
    pub fn iris_print_matrix_limited_int16_t(data: *const i16, m: i32, n: i32, description: *const c_char, limit: i32);
    pub fn iris_print_matrix_full_int8_t(data: *const i8, m: i32, n: i32, description: *const c_char);
    pub fn iris_print_matrix_limited_int8_t(data: *const i8, m: i32, n: i32, description: *const c_char, limit: i32);
}

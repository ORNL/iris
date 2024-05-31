# iris.jl
import Pkg; Pkg.add("CBinding")
import Pkg; Pkg.add("Libdl")

module Iris

using CBinding
using Libdl

# Define the library name
const libiris = ENV["IRIS"] * "/lib/libiris.so"
Libdl.dlopen(libiris)

# Define enums
@enum StreamPolicy STREAM_POLICY_DEFAULT=0 STREAM_POLICY_SAME_FOR_TASK=1 STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL=2
@enum DeviceModel iris_cuda=1 iris_hip=3 iris_levelzero=4 iris_opencl=5 iris_openmp=6 iris_model_all=(1<<25)

# Define the structs
@cstruct iris_task struct {
    class_obj::Ptr{Cvoid},  # Placeholder for iris::rt::Task*
    uid::Culong
}

@cstruct iris_kernel struct {
    class_obj::Ptr{Cvoid},  # Placeholder for iris::rt::Kernel*
    uid::Culong
}

@cstruct iris_mem struct {
    class_obj::Ptr{Cvoid},  # Placeholder for iris::rt::BaseMem*
    uid::Culong
}

@cstruct iris_graph struct {
    class_obj::Ptr{Cvoid},  # Placeholder for iris::rt::Graph*
    uid::Culong
}

# Define function pointer types
const IrisHostTask = @cfunction(Cint, (Ptr{Cvoid}, Ptr{Cint}))
const IrisHostPythonTask = @cfunction(Cint, (Ptr{Cint64}, Ptr{Cint}))
const CommandHandler = @cfunction(Cint, (Ptr{Cvoid}, Ptr{Cvoid}))
const HookTask = @cfunction(Cint, (Ptr{Cvoid}))
const HookCommand = @cfunction(Cint, (Ptr{Cvoid}))
const IrisSelectorKernel = @cfunction(Cint, (iris_task, Ptr{Cvoid}, Ptr{Cchar}))

export iris_init, iris_error_count, iris_finalize, iris_synchronize, iris_task_retain, 
       iris_env_set, iris_env_get, iris_overview, iris_platform_count, iris_platform_info, 
       iris_set_stream_policy, iris_set_asynchronous, iris_set_shared_memory_model, 
       iris_mem_enable_usm, iris_mem_disable_usm, iris_set_enable_profiler, iris_device_count, 
       iris_ndevices, iris_set_nstreams, iris_set_ncopy_streams, iris_nstreams, iris_ncopy_streams, 
       iris_device_info, iris_device_set_default, iris_device_get_default, iris_device_synchronize, 
       iris_register_policy, iris_register_command, iris_register_hooks_task, iris_register_hooks_command, 
       iris_kernel_create, iris_kernel_get, iris_kernel_setarg, iris_kernel_setmem, iris_kernel_setmem_off, 
       iris_kernel_release, iris_task_create, iris_task_create_perm, iris_task_create_name, iris_task_depend, 
       iris_task_malloc, iris_task_cmd_reset_mem, iris_task_set_stream_policy, iris_task_disable_asynchronous, 
       iris_task_get_metadata, iris_task_set_metadata, iris_task_h2broadcast, iris_task_h2broadcast_offsets, 
       iris_task_h2broadcast_full, iris_task_d2d, iris_task_h2d, iris_task_h2d_offsets, iris_task_d2h, 
       iris_task_d2h_offsets, iris_task_dmem_flush_out, iris_task_h2d_full, iris_task_d2h_full, 
       iris_task_kernel_object, iris_task_kernel, iris_task_kernel_v2, iris_task_kernel_v3, 
       iris_task_kernel_selector, iris_task_kernel_launch_disabled, iris_task_python_host, iris_task_host, 
       iris_task_custom, iris_task_submit, iris_task_set_policy, iris_task_get_policy, iris_task_wait, 
       iris_task_wait_all, iris_task_add_subtask, iris_task_kernel_cmd_only, iris_task_release, 
       iris_task_release_mem, iris_params_map, iris_task_info, iris_register_pin_memory, iris_mem_create, 
       iris_data_mem_init_reset, iris_data_mem_create, iris_data_mem_create_ptr, iris_data_mem_clear, 
       iris_data_mem_pin, iris_data_mem_update, iris_data_mem_create_region, iris_data_mem_create_region_ptr, 
       iris_data_mem_enable_outer_dim_regions, iris_data_mem_create_tile, iris_data_mem_create_tile_ptr, 
       iris_data_mem_update_bc, iris_data_mem_get_rr_bc_dev, iris_data_mem_n_regions, iris_data_mem_get_region_uid, 
       iris_mem_arch, iris_mem_reduce, iris_mem_release, iris_graph_create, iris_graph_create_null, 
       iris_is_graph_null, iris_graph_free, iris_graph_tasks_order, iris_graph_create_json, iris_graph_task, 
       iris_graph_retain, iris_graph_release, iris_graph_submit, iris_graph_submit_with_order, 
       iris_graph_submit_with_order_and_time, iris_graph_submit_with_time, iris_graph_wait, iris_graph_wait_all, 
       iris_record_start, iris_record_stop, iris_timer_now, iris_enable_d2d, iris_disable_d2d, 
       iris_disable_consistency_check, iris_enable_consistency_check, iris_kernel_get_name, iris_task_get_name, 
       iris_task_set_name, iris_task_get_dependency_count, iris_task_get_dependencies, iris_task_get_uid, 
       iris_kernel_get_uid, iris_task_get_kernel, iris_task_kernel_dmem_fetch_order, iris_task_disable_consistency, 
       iris_task_is_cmd_kernel_exists, iris_task_get_cmd_kernel, iris_mem_get_size, iris_mem_get_type, 
       iris_mem_get_uid, iris_mem_is_reset, iris_get_dmem_for_region, iris_cmd_kernel_get_nargs, 
       iris_cmd_kernel_get_arg_is_mem, iris_cmd_kernel_get_arg_size, iris_cmd_kernel_get_arg_value, 
       iris_cmd_kernel_get_arg_mem, iris_cmd_kernel_get_arg_mem_off, iris_cmd_kernel_get_arg_mem_size, 
       iris_cmd_kernel_get_arg_off, iris_cmd_kernel_get_arg_mode, iris_graph_enable_mem_profiling, 
       iris_graph_reset_memories, iris_graph_get_tasks, iris_graph_tasks_count, 
       iris_get_graph_max_theoretical_parallelism, iris_get_graph_dependency_adj_list, 
       iris_get_graph_dependency_adj_matrix, iris_get_graph_3d_comm_data_size, iris_get_graph_3d_comm_data_ptr, 
       iris_get_graph_tasks_execution_schedule, iris_get_graph_tasks_execution_schedule_count, 
       iris_get_graph_dataobjects_execution_schedule, iris_get_graph_dataobjects_execution_schedule_count, 
       iris_get_graph_3d_comm_data, iris_get_graph_2d_comm_adj_matrix, iris_calibrate_compute_cost_adj_matrix, 
       iris_calibrate_compute_cost_adj_matrix_only_for_types, iris_calibrate_communication_cost, 
       iris_get_graph_3d_comm_time, iris_count_mems, iris_free_array, iris_allocate_array_int8_t, 
       iris_allocate_array_int16_t, iris_allocate_array_int32_t, iris_allocate_array_int64_t, 
       iris_allocate_array_size_t, iris_allocate_array_float, iris_allocate_array_double, 
       iris_allocate_random_array_int8_t, iris_allocate_random_array_int16_t, iris_allocate_random_array_int32_t, 
       iris_allocate_random_array_int64_t, iris_allocate_random_array_size_t, iris_allocate_random_array_float, 
       iris_allocate_random_array_double, iris_print_matrix_full_double, iris_print_matrix_limited_double, 
       iris_print_matrix_full_float, iris_print_matrix_limited_float, iris_print_matrix_full_int64_t, 
       iris_print_matrix_limited_int64_t, iris_print_matrix_full_int32_t, iris_print_matrix_limited_int32_t, 
       iris_print_matrix_full_int16_t, iris_print_matrix_limited_int16_t, iris_print_matrix_full_int8_t, 
       iris_print_matrix_limited_int8_t, iris_run_hpl_mapping, iris_read_bool_env, iris_read_int_env

# Bind functions from the IRIS library
iris_init = @cfunction(libiris, iris_init, Cint, (Ref{Cint}, Ref{Ptr{Ptr{Cchar}}}, Cint))
iris_error_count = @cfunction(libiris, iris_error_count, Cint, ())
iris_finalize = @cfunction(libiris, iris_finalize, Cint, ())
iris_synchronize = @cfunction(libiris, iris_synchronize, Cint, ())
iris_task_retain = @cfunction(libiris, iris_task_retain, Cvoid, (iris_task, Cbool))
iris_env_set = @cfunction(libiris, iris_env_set, Cint, (Ptr{Cchar}, Ptr{Cchar}))
iris_env_get = @cfunction(libiris, iris_env_get, Cint, (Ptr{Cchar}, Ref{Ptr{Cchar}}, Ref{Csize_t}))
iris_overview = @cfunction(libiris, iris_overview, Cvoid, ())
iris_platform_count = @cfunction(libiris, iris_platform_count, Cint, (Ref{Cint}))
iris_platform_info = @cfunction(libiris, iris_platform_info, Cint, (Cint, Cint, Ptr{Cvoid}, Ref{Csize_t}))
iris_set_stream_policy = @cfunction(libiris, iris_set_stream_policy, Cint, (StreamPolicy))
iris_set_asynchronous = @cfunction(libiris, iris_set_asynchronous, Cint, (Cint))
iris_set_shared_memory_model = @cfunction(libiris, iris_set_shared_memory_model, Cint, (Cint))
iris_mem_enable_usm = @cfunction(libiris, iris_mem_enable_usm, Cint, (iris_mem, DeviceModel))
iris_mem_disable_usm = @cfunction(libiris, iris_mem_disable_usm, Cint, (iris_mem, DeviceModel))
iris_set_enable_profiler = @cfunction(libiris, iris_set_enable_profiler, Cvoid, (Cint))
iris_device_count = @cfunction(libiris, iris_device_count, Cint, (Ref{Cint}))
iris_ndevices = @cfunction(libiris, iris_ndevices, Cint, ())
iris_set_nstreams = @cfunction(libiris, iris_set_nstreams, Cint, (Cint))
iris_set_ncopy_streams = @cfunction(libiris, iris_set_ncopy_streams, Cint, (Cint))
iris_nstreams = @cfunction(libiris, iris_nstreams, Cint, ())
iris_ncopy_streams = @cfunction(libiris, iris_ncopy_streams, Cint, ())
iris_device_info = @cfunction(libiris, iris_device_info, Cint, (Cint, Cint, Ptr{Cvoid}, Ref{Csize_t}))
iris_device_set_default = @cfunction(libiris, iris_device_set_default, Cint, (Cint))
iris_device_get_default = @cfunction(libiris, iris_device_get_default, Cint, (Ref{Cint}))
iris_device_synchronize = @cfunction(libiris, iris_device_synchronize, Cint, (Cint, Ptr{Cint}))
iris_register_policy = @cfunction(libiris, iris_register_policy, Cint, (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cvoid}))
iris_register_command = @cfunction(libiris, iris_register_command, Cint, (Cint, Cint, CommandHandler))
iris_register_hooks_task = @cfunction(libiris, iris_register_hooks_task, Cint, (HookTask, HookTask))
iris_register_hooks_command = @cfunction(libiris, iris_register_hooks_command, Cint, (HookCommand, HookCommand))
iris_kernel_create = @cfunction(libiris, iris_kernel_create, Cint, (Ptr{Cchar}, Ref{iris_kernel}))
iris_kernel_get = @cfunction(libiris, iris_kernel_get, Cint, (Ptr{Cchar}, Ref{iris_kernel}))
iris_kernel_setarg = @cfunction(libiris, iris_kernel_setarg, Cint, (iris_kernel, Cint, Csize_t, Ptr{Cvoid}))
iris_kernel_setmem = @cfunction(libiris, iris_kernel_setmem, Cint, (iris_kernel, Cint, iris_mem, Csize_t))
iris_kernel_setmem_off = @cfunction(libiris, iris_kernel_setmem_off, Cint, (iris_kernel, Cint, iris_mem, Csize_t, Csize_t))
iris_kernel_release = @cfunction(libiris, iris_kernel_release, Cint, (iris_kernel,))
iris_task_create = @cfunction(libiris, iris_task_create, Cint, (Ref{iris_task}))
iris_task_create_perm = @cfunction(libiris, iris_task_create_perm, Cint, (Ref{iris_task}))
iris_task_create_name = @cfunction(libiris, iris_task_create_name, Cint, (Ptr{Cchar}, Ref{iris_task}))
iris_task_depend = @cfunction(libiris, iris_task_depend, Cint, (iris_task, Cint, Ptr{iris_task}))
iris_task_malloc = @cfunction(libiris, iris_task_malloc, Cint, (iris_task, iris_mem))
iris_task_cmd_reset_mem = @cfunction(libiris, iris_task_cmd_reset_mem, Cint, (iris_task, iris_mem, Cuint8))
iris_task_set_stream_policy = @cfunction(libiris, iris_task_set_stream_policy, Cint, (iris_task, StreamPolicy))
iris_task_disable_asynchronous = @cfunction(libiris, iris_task_disable_asynchronous, Cvoid, (iris_task,))
iris_task_get_metadata = @cfunction(libiris, iris_task_get_metadata, Cint, (iris_task, Cint))
iris_task_set_metadata = @cfunction(libiris, iris_task_set_metadata, Cint, (iris_task, Cint, Cint))
iris_task_h2broadcast = @cfunction(libiris, iris_task_h2broadcast, Cint, (iris_task, iris_mem, Csize_t, Csize_t, Ptr{Cvoid}))
iris_task_h2broadcast_offsets = @cfunction(libiris, iris_task_h2broadcast_offsets, Cint, (iris_task, iris_mem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cint, Ptr{Cvoid}))
iris_task_h2broadcast_full = @cfunction(libiris, iris_task_h2broadcast_full, Cint, (iris_task, iris_mem, Ptr{Cvoid}))
iris_task_d2d = @cfunction(libiris, iris_task_d2d, Cint, (iris_task, iris_mem, Csize_t, Csize_t, Ptr{Cvoid}, Cint))
iris_task_h2d = @cfunction(libiris, iris_task_h2d, Cint, (iris_task, iris_mem, Csize_t, Csize_t, Ptr{Cvoid}))
iris_task_h2d_offsets = @cfunction(libiris, iris_task_h2d_offsets, Cint, (iris_task, iris_mem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cint, Ptr{Cvoid}))
iris_task_d2h = @cfunction(libiris, iris_task_d2h, Cint, (iris_task, iris_mem, Csize_t, Csize_t, Ptr{Cvoid}))
iris_task_d2h_offsets = @cfunction(libiris, iris_task_d2h_offsets, Cint, (iris_task, iris_mem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cint, Ptr{Cvoid}))
iris_task_dmem_flush_out = @cfunction(libiris, iris_task_dmem_flush_out, Cint, (iris_task, iris_mem))
iris_task_h2d_full = @cfunction(libiris, iris_task_h2d_full, Cint, (iris_task, iris_mem, Ptr{Cvoid}))
iris_task_d2h_full = @cfunction(libiris, iris_task_d2h_full, Cint, (iris_task, iris_mem, Ptr{Cvoid}))
iris_task_kernel_object = @cfunction(libiris, iris_task_kernel_object, Cint, (iris_task, iris_kernel, Cint, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}))
iris_task_kernel = @cfunction(libiris, iris_task_kernel, Cint, (iris_task, Ptr{Cchar}, Cint, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cint}))
iris_task_kernel_v2 = @cfunction(libiris, iris_task_kernel_v2, Cint, (iris_task, Ptr{Cchar}, Cint, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Cint, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cint}))
iris_task_kernel_v3 = @cfunction(libiris, iris_task_kernel_v3, Cint, (iris_task, Ptr{Cchar}, Cint, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Cint, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cint}, Ptr{Csize_t}))
iris_task_kernel_selector = @cfunction(libiris, iris_task_kernel_selector, Cint, (iris_task, IrisSelectorKernel, Ptr{Cvoid}, Csize_t))
iris_task_kernel_launch_disabled = @cfunction(libiris, iris_task_kernel_launch_disabled, Cint, (iris_task, Cint))
iris_task_python_host = @cfunction(libiris, iris_task_python_host, Cint, (iris_task, IrisHostPythonTask, Cint64))
iris_task_host = @cfunction(libiris, iris_task_host, Cint, (iris_task, IrisHostTask, Ptr{Cvoid}))
iris_task_custom = @cfunction(libiris, iris_task_custom, Cint, (iris_task, Cint, Ptr{Cvoid}, Csize_t))
iris_task_submit = @cfunction(libiris, iris_task_submit, Cint, (iris_task, Cint, Ptr{Cchar}, Cint))
iris_task_set_policy = @cfunction(libiris, iris_task_set_policy, Cint, (iris_task, Cint))
iris_task_get_policy = @cfunction(libiris, iris_task_get_policy, Cint, (iris_task,))
iris_task_wait = @cfunction(libiris, iris_task_wait, Cint, (iris_task,))
iris_task_wait_all = @cfunction(libiris, iris_task_wait_all, Cint, (Cint, Ptr{iris_task}))
iris_task_add_subtask = @cfunction(libiris, iris_task_add_subtask, Cint, (iris_task, iris_task))
iris_task_kernel_cmd_only = @cfunction(libiris, iris_task_kernel_cmd_only, Cint, (iris_task,))
iris_task_release = @cfunction(libiris, iris_task_release, Cint, (iris_task,))
iris_task_release_mem = @cfunction(libiris, iris_task_release_mem, Cint, (iris_task, iris_mem))
iris_params_map = @cfunction(libiris, iris_params_map, Cint, (iris_task, Ptr{Cint}))
iris_task_info = @cfunction(libiris, iris_task_info, Cint, (iris_task, Cint, Ptr{Cvoid}, Ref{Csize_t}))
iris_register_pin_memory = @cfunction(libiris, iris_register_pin_memory, Cint, (Ptr{Cvoid}, Csize_t))
iris_mem_create = @cfunction(libiris, iris_mem_create, Cint, (Csize_t, Ref{iris_mem}))
iris_data_mem_init_reset = @cfunction(libiris, iris_data_mem_init_reset, Cint, (iris_mem, Cint))
iris_data_mem_create = @cfunction(libiris, iris_data_mem_create, Cint, (Ref{iris_mem}, Ptr{Cvoid}, Csize_t))
iris_data_mem_create_ptr = @cfunction(libiris, iris_data_mem_create_ptr, Ptr{iris_mem}, (Ptr{Cvoid}, Csize_t))
iris_data_mem_clear = @cfunction(libiris, iris_data_mem_clear, Cint, (iris_mem,))
iris_data_mem_pin = @cfunction(libiris, iris_data_mem_pin, Cint, (iris_mem,))
iris_data_mem_update = @cfunction(libiris, iris_data_mem_update, Cint, (iris_mem, Ptr{Cvoid}))
iris_data_mem_create_region = @cfunction(libiris, iris_data_mem_create_region, Cint, (Ref{iris_mem}, iris_mem, Cint))
iris_data_mem_create_region_ptr = @cfunction(libiris, iris_data_mem_create_region_ptr, Ptr{iris_mem}, (iris_mem, Cint))
iris_data_mem_enable_outer_dim_regions = @cfunction(libiris, iris_data_mem_enable_outer_dim_regions, Cint, (iris_mem,))
iris_data_mem_create_tile = @cfunction(libiris, iris_data_mem_create_tile, Cint, (Ref{iris_mem}, Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cint))
iris_data_mem_create_tile_ptr = @cfunction(libiris, iris_data_mem_create_tile_ptr, Ptr{iris_mem}, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Cint))
iris_data_mem_update_bc = @cfunction(libiris, iris_data_mem_update_bc, Cint, (iris_mem, Cbool, Cint, Cint))
iris_data_mem_get_rr_bc_dev = @cfunction(libiris, iris_data_mem_get_rr_bc_dev, Cint, (iris_mem,))
iris_data_mem_n_regions = @cfunction(libiris, iris_data_mem_n_regions, Cint, (iris_mem,))
iris_data_mem_get_region_uid = @cfunction(libiris, iris_data_mem_get_region_uid, Culong, (iris_mem, Cint))
iris_mem_arch = @cfunction(libiris, iris_mem_arch, Cint, (iris_mem, Cint, Ref{Ptr{Cvoid}}))
iris_mem_reduce = @cfunction(libiris, iris_mem_reduce, Cint, (iris_mem, Cint, Cint))
iris_mem_release = @cfunction(libiris, iris_mem_release, Cint, (iris_mem,))
iris_graph_create = @cfunction(libiris, iris_graph_create, Cint, (Ref{iris_graph}))
iris_graph_create_null = @cfunction(libiris, iris_graph_create_null, Cint, (Ref{iris_graph}))
iris_is_graph_null = @cfunction(libiris, iris_is_graph_null, Cbool, (iris_graph,))
iris_graph_free = @cfunction(libiris, iris_graph_free, Cint, (iris_graph,))
iris_graph_tasks_order = @cfunction(libiris, iris_graph_tasks_order, Cint, (iris_graph, Ptr{Cint}))
iris_graph_create_json = @cfunction(libiris, iris_graph_create_json, Cint, (Ptr{Cchar}, Ref{Ptr{Cvoid}}, Ref{iris_graph}))
iris_graph_task = @cfunction(libiris, iris_graph_task, Cint, (iris_graph, iris_task, Cint, Ptr{Cchar}))
iris_graph_retain = @cfunction(libiris, iris_graph_retain, Cint, (iris_graph, Cbool))
iris_graph_release = @cfunction(libiris, iris_graph_release, Cint, (iris_graph,))
iris_graph_submit = @cfunction(libiris, iris_graph_submit, Cint, (iris_graph, Cint, Cint))
iris_graph_submit_with_order = @cfunction(libiris, iris_graph_submit_with_order, Cint, (iris_graph, Ptr{Cint}, Cint, Cint))
iris_graph_submit_with_order_and_time = @cfunction(libiris, iris_graph_submit_with_order_and_time, Cint, (iris_graph, Ptr{Cint}, Ref{Cdouble}, Cint, Cint))
iris_graph_submit_with_time = @cfunction(libiris, iris_graph_submit_with_time, Cint, (iris_graph, Ref{Cdouble}, Cint, Cint))
iris_graph_wait = @cfunction(libiris, iris_graph_wait, Cint, (iris_graph,))
iris_graph_wait_all = @cfunction(libiris, iris_graph_wait_all, Cint, (Cint, Ptr{iris_graph}))
iris_record_start = @cfunction(libiris, iris_record_start, Cint, ())
iris_record_stop = @cfunction(libiris, iris_record_stop, Cint, ())
iris_timer_now = @cfunction(libiris, iris_timer_now, Cint, (Ref{Cdouble}))
iris_enable_d2d = @cfunction(libiris, iris_enable_d2d, Cvoid, ())
iris_disable_d2d = @cfunction(libiris, iris_disable_d2d, Cvoid, ())
iris_disable_consistency_check = @cfunction(libiris, iris_disable_consistency_check, Cvoid, ())
iris_enable_consistency_check = @cfunction(libiris, iris_enable_consistency_check, Cvoid, ())
iris_kernel_get_name = @cfunction(libiris, iris_kernel_get_name, Ptr{Cchar}, (iris_kernel,))
iris_task_get_name = @cfunction(libiris, iris_task_get_name, Ptr{Cchar}, (iris_task,))
iris_task_set_name = @cfunction(libiris, iris_task_set_name, Cvoid, (iris_task, Ptr{Cchar}))
iris_task_get_dependency_count = @cfunction(libiris, iris_task_get_dependency_count, Cint, (iris_task,))
iris_task_get_dependencies = @cfunction(libiris, iris_task_get_dependencies, Cvoid, (iris_task, Ptr{iris_task}))
iris_task_get_uid = @cfunction(libiris, iris_task_get_uid, Culong, (iris_task,))
iris_kernel_get_uid = @cfunction(libiris, iris_kernel_get_uid, Culong, (iris_kernel,))
iris_task_get_kernel = @cfunction(libiris, iris_task_get_kernel, iris_kernel, (iris_task,))
iris_task_kernel_dmem_fetch_order = @cfunction(libiris, iris_task_kernel_dmem_fetch_order, Cint, (iris_task, Ptr{Cint}))
iris_task_disable_consistency = @cfunction(libiris, iris_task_disable_consistency, Cint, (iris_task,))
iris_task_is_cmd_kernel_exists = @cfunction(libiris, iris_task_is_cmd_kernel_exists, Cint, (iris_task,))
iris_task_get_cmd_kernel = @cfunction(libiris, iris_task_get_cmd_kernel, Ptr{Cvoid}, (iris_task,))
iris_mem_get_size = @cfunction(libiris, iris_mem_get_size, Csize_t, (iris_mem,))
iris_mem_get_type = @cfunction(libiris, iris_mem_get_type, Cint, (iris_mem,))
iris_mem_get_uid = @cfunction(libiris, iris_mem_get_uid, Cint, (iris_mem,))
iris_mem_is_reset = @cfunction(libiris, iris_mem_is_reset, Cint, (iris_mem,))
iris_get_dmem_for_region = @cfunction(libiris, iris_get_dmem_for_region, iris_mem, (iris_mem,))
iris_cmd_kernel_get_nargs = @cfunction(libiris, iris_cmd_kernel_get_nargs, Cint, (Ptr{Cvoid},))
iris_cmd_kernel_get_arg_is_mem = @cfunction(libiris, iris_cmd_kernel_get_arg_is_mem, Cint, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_size = @cfunction(libiris, iris_cmd_kernel_get_arg_size, Csize_t, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_value = @cfunction(libiris, iris_cmd_kernel_get_arg_value, Ptr{Cvoid}, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_mem = @cfunction(libiris, iris_cmd_kernel_get_arg_mem, iris_mem, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_mem_off = @cfunction(libiris, iris_cmd_kernel_get_arg_mem_off, Csize_t, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_mem_size = @cfunction(libiris, iris_cmd_kernel_get_arg_mem_size, Csize_t, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_off = @cfunction(libiris, iris_cmd_kernel_get_arg_off, Csize_t, (Ptr{Cvoid}, Cint))
iris_cmd_kernel_get_arg_mode = @cfunction(libiris, iris_cmd_kernel_get_arg_mode, Cint, (Ptr{Cvoid}, Cint))
iris_graph_enable_mem_profiling = @cfunction(libiris, iris_graph_enable_mem_profiling, Cint, (iris_graph,))
iris_graph_reset_memories = @cfunction(libiris, iris_graph_reset_memories, Cint, (iris_graph,))
iris_graph_get_tasks = @cfunction(libiris, iris_graph_get_tasks, Cint, (iris_graph, Ptr{iris_task}))
iris_graph_tasks_count = @cfunction(libiris, iris_graph_tasks_count, Cint, (iris_graph,))
iris_get_graph_max_theoretical_parallelism = @cfunction(libiris, iris_get_graph_max_theoretical_parallelism, Cint, (iris_graph,))
iris_get_graph_dependency_adj_list = @cfunction(libiris, iris_get_graph_dependency_adj_list, Cint, (iris_graph, Ptr{Cint8}))
iris_get_graph_dependency_adj_matrix = @cfunction(libiris, iris_get_graph_dependency_adj_matrix, Cint, (iris_graph, Ptr{Cint8}))
iris_get_graph_3d_comm_data_size = @cfunction(libiris, iris_get_graph_3d_comm_data_size, Csize_t, (iris_graph,))
iris_get_graph_3d_comm_data_ptr = @cfunction(libiris, iris_get_graph_3d_comm_data_ptr, Ptr{Cvoid}, (iris_graph,))
iris_get_graph_tasks_execution_schedule = @cfunction(libiris, iris_get_graph_tasks_execution_schedule, Ptr{Cvoid}, (iris_graph, Cint))
iris_get_graph_tasks_execution_schedule_count = @cfunction(libiris, iris_get_graph_tasks_execution_schedule_count, Csize_t, (iris_graph,))
iris_get_graph_dataobjects_execution_schedule = @cfunction(libiris, iris_get_graph_dataobjects_execution_schedule, Ptr{Cvoid}, (iris_graph,))
iris_get_graph_dataobjects_execution_schedule_count = @cfunction(libiris, iris_get_graph_dataobjects_execution_schedule_count, Csize_t, (iris_graph,))
iris_get_graph_3d_comm_data = @cfunction(libiris, iris_get_graph_3d_comm_data, Cint, (iris_graph, Ptr{Cvoid}))
iris_get_graph_2d_comm_adj_matrix = @cfunction(libiris, iris_get_graph_2d_comm_adj_matrix, Cint, (iris_graph, Ref{Csize_t}))
iris_calibrate_compute_cost_adj_matrix = @cfunction(libiris, iris_calibrate_compute_cost_adj_matrix, Cint, (iris_graph, Ptr{Cdouble}))
iris_calibrate_compute_cost_adj_matrix_only_for_types = @cfunction(libiris, iris_calibrate_compute_cost_adj_matrix_only_for_types, Cint, (iris_graph, Ptr{Cdouble}))
iris_calibrate_communication_cost = @cfunction(libiris, iris_calibrate_communication_cost, Cint, (Ptr{Cdouble}, Csize_t, Cint, Cint))
iris_get_graph_3d_comm_time = @cfunction(libiris, iris_get_graph_3d_comm_time, Cint, (iris_graph, Ptr{Cdouble}, Ptr{Cint}, Cint, Cint))
iris_count_mems = @cfunction(libiris, iris_count_mems, Csize_t, (iris_graph,))
iris_free_array = @cfunction(libiris, iris_free_array, Cvoid, (Ptr{Cvoid},))
iris_allocate_array_int8_t = @cfunction(libiris, iris_allocate_array_int8_t, Ptr{Cint8}, (Cint, Cint8))
iris_allocate_array_int16_t = @cfunction(libiris, iris_allocate_array_int16_t, Ptr{Cint16}, (Cint, Cint16))
iris_allocate_array_int32_t = @cfunction(libiris, iris_allocate_array_int32_t, Ptr{Cint32}, (Cint, Cint32))
iris_allocate_array_int64_t = @cfunction(libiris, iris_allocate_array_int64_t, Ptr{Cint64}, (Cint, Cint64))
iris_allocate_array_size_t = @cfunction(libiris, iris_allocate_array_size_t, Ptr{Csize_t}, (Cint, Csize_t))
iris_allocate_array_float = @cfunction(libiris, iris_allocate_array_float, Ptr{Cfloat}, (Cint, Cfloat))
iris_allocate_array_double = @cfunction(libiris, iris_allocate_array_double, Ptr{Cdouble}, (Cint, Cdouble))
iris_allocate_random_array_int8_t = @cfunction(libiris, iris_allocate_random_array_int8_t, Ptr{Cint8}, (Cint,))
iris_allocate_random_array_int16_t = @cfunction(libiris, iris_allocate_random_array_int16_t, Ptr{Cint16}, (Cint,))
iris_allocate_random_array_int32_t = @cfunction(libiris, iris_allocate_random_array_int32_t, Ptr{Cint32}, (Cint,))
iris_allocate_random_array_int64_t = @cfunction(libiris, iris_allocate_random_array_int64_t, Ptr{Cint64}, (Cint,))
iris_allocate_random_array_size_t = @cfunction(libiris, iris_allocate_random_array_size_t, Ptr{Csize_t}, (Cint,))
iris_allocate_random_array_float = @cfunction(libiris, iris_allocate_random_array_float, Ptr{Cfloat}, (Cint,))
iris_allocate_random_array_double = @cfunction(libiris, iris_allocate_random_array_double, Ptr{Cdouble}, (Cint,))
iris_print_matrix_full_double = @cfunction(libiris, iris_print_matrix_full_double, Cvoid, (Ptr{Cdouble}, Cint, Cint, Ptr{Cchar}))
iris_print_matrix_limited_double = @cfunction(libiris, iris_print_matrix_limited_double, Cvoid, (Ptr{Cdouble}, Cint, Cint, Ptr{Cchar}, Cint))
iris_print_matrix_full_float = @cfunction(libiris, iris_print_matrix_full_float, Cvoid, (Ptr{Cfloat}, Cint, Cint, Ptr{Cchar}))
iris_print_matrix_limited_float = @cfunction(libiris, iris_print_matrix_limited_float, Cvoid, (Ptr{Cfloat}, Cint, Cint, Ptr{Cchar}, Cint))
iris_print_matrix_full_int64_t = @cfunction(libiris, iris_print_matrix_full_int64_t, Cvoid, (Ptr{Cint64}, Cint, Cint, Ptr{Cchar}))
iris_print_matrix_limited_int64_t = @cfunction(libiris, iris_print_matrix_limited_int64_t, Cvoid, (Ptr{Cint64}, Cint, Cint, Ptr{Cchar}, Cint))
iris_print_matrix_full_int32_t = @cfunction(libiris, iris_print_matrix_full_int32_t, Cvoid, (Ptr{Cint32}, Cint, Cint, Ptr{Cchar}))
iris_print_matrix_limited_int32_t = @cfunction(libiris, iris_print_matrix_limited_int32_t, Cvoid, (Ptr{Cint32}, Cint, Cint, Ptr{Cchar}, Cint))
iris_print_matrix_full_int16_t = @cfunction(libiris, iris_print_matrix_full_int16_t, Cvoid, (Ptr{Cint16}, Cint, Cint, Ptr{Cchar}))
iris_print_matrix_limited_int16_t = @cfunction(libiris, iris_print_matrix_limited_int16_t, Cvoid, (Ptr{Cint16}, Cint, Cint, Ptr{Cchar}, Cint))
iris_print_matrix_full_int8_t = @cfunction(libiris, iris_print_matrix_full_int8_t, Cvoid, (Ptr{Cint8}, Cint, Cint, Ptr{Cchar}))
iris_print_matrix_limited_int8_t = @cfunction(libiris, iris_print_matrix_limited_int8_t, Cvoid, (Ptr{Cint8}, Cint, Cint, Ptr{Cchar}, Cint))
iris_run_hpl_mapping = @cfunction(libiris, iris_run_hpl_mapping, Cvoid, (iris_graph,))
iris_read_bool_env = @cfunction(libiris, iris_read_bool_env, Cint, (Ptr{Cchar},))
iris_read_int_env = @cfunction(libiris, iris_read_int_env, Cint, (Ptr{Cchar},))
end

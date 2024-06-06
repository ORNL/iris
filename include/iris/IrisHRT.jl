#####################################################
#   Author: Narasinga Rao Miniskar
#   Date: 06/06/2024
#   File: IrisHRT.jl
#   Contact: miniskarnr@ornl.gov
#   Comment: IRIS Julia interface
#####################################################
module IrisHRT
    using Libdl
    const libiris = "libiris.so"
    # Load the shared library
    lib = Libdl.dlopen(libiris)

    # Define enums
    @enum StreamPolicy STREAM_POLICY_DEFAULT=0 STREAM_POLICY_SAME_FOR_TASK=1 STREAM_POLICY_GIVE_ALL_STREAMS_TO_KERNEL=2
    @enum DeviceModel iris_cuda=1 iris_hip=3 iris_levelzero=4 iris_opencl=5 iris_openmp=6 iris_model_all=(1<<25)

    # Define constants corresponding to the C macros
    const iris_r::Int32 = -1
    const iris_w::Int32 = -2
    const iris_rw::Int32 = -3
    const iris_xr::Int32 = -4
    const iris_xw::Int32 = -5
    const iris_xrw::Int32 = -6

    const iris_default = 1 << 5
    const iris_cpu = 1 << 6
    const iris_nvidia = 1 << 7
    const iris_amd = 1 << 8
    const iris_gpu_intel = 1 << 9
    const iris_gpu = iris_nvidia | iris_amd | iris_gpu_intel
    const iris_phi = 1 << 10
    const iris_fpga = 1 << 11
    const iris_hexagon = 1 << 12
    const iris_dsp = iris_hexagon
    const iris_roundrobin = 1 << 18
    const iris_depend = 1 << 19
    const iris_data = 1 << 20
    const iris_profile = 1 << 21
    const iris_random = 1 << 22
    const iris_pending = 1 << 23
    const iris_sdq = 1 << 24
    const iris_ftf = 1 << 25
    const iris_all = 1 << 25
    const iris_ocl = 1 << 26
    const iris_block_cycle = 1 << 27
    const iris_custom = 1 << 28

    const iris_int = (1 << 1) << 16
    const iris_uint = (1 << 1) << 16
    const iris_float = (1 << 2) << 16
    const iris_double = (1 << 3) << 16
    const iris_char = (1 << 4) << 16
    const iris_int8 = (1 << 4) << 16
    const iris_uint8 = (1 << 4) << 16
    const iris_int16 = (1 << 5) << 16
    const iris_uint16 = (1 << 5) << 16
    const iris_int32 = (1 << 6) << 16
    const iris_uint32 = (1 << 6) << 16
    const iris_int64 = (1 << 7) << 16
    const iris_uint64 = (1 << 7) << 16
    const iris_long = (1 << 8) << 16
    const iris_unsigned_long = (1 << 8) << 16

    const iris_normal = 1 << 10
    const iris_reduction = 1 << 11
    const iris_sum = (1 << 12) | iris_reduction
    const iris_max = (1 << 13) | iris_reduction
    const iris_min = (1 << 14) | iris_reduction

    # Define structs (adjust as necessary based on iris.h definitions)
    struct IrisTask
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    struct IrisKernel
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    struct IrisMem
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    struct IrisGraph
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    # Define function pointer types
    const IrisHostTask = Ptr{Cvoid}
    const IrisHostPythonTask = Ptr{Cvoid}
    const CommandHandler = Ptr{Cvoid}
    const HookTask = Ptr{Cvoid}
    const HookCommand = Ptr{Cvoid}
    const IrisSelectorKernel = Ptr{Cvoid}

    # Bind functions from the IRIS library using ccall
    function iris_init(sync::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_init), Int32, (Ref{Int32}, Ref{Ptr{Ptr{Cchar}}}, Int32), Int32(1), C_NULL, sync)
    end

    function iris_error_count()::Int32
        return ccall(Libdl.dlsym(lib, :iris_error_count), Int32, ())
    end

    function iris_finalize()::Int32
        return ccall(Libdl.dlsym(lib, :iris_finalize), Int32, ())
    end

    function iris_synchronize()::Int32
        return ccall(Libdl.dlsym(lib, :iris_synchronize), Int32, ())
    end

    function iris_task_retain(task::IrisTask, flag::Int8)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_task_retain), Cvoid, (IrisTask, Int8), task, flag)
    end

    function iris_env_set(key::String, value::String)::Int32
        return ccall(Libdl.dlsym(lib, :iris_env_set), Int32, (Ptr{Cchar}, Ptr{Cchar}), pointer(key), pointer(value))
    end

    function iris_env_get(key::String, value::String, vallen::Ref{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_env_get), Int32, (Ptr{Cchar}, Ref{Ptr{Cchar}}, Ref{Csize_t}), pointer(key), pointer(value), vallen)
    end

    function iris_overview()::Cvoid
        ccall(Libdl.dlsym(lib, :iris_overview), Cvoid, ())
    end

    function iris_platform_count(nplatforms::Ref{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_platform_count), Int32, (Ref{Int32},), nplatforms)
    end

    function iris_platform_info(platform::Int32, param::Int32, value::Ptr{Cvoid}, size::Ref{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_platform_info), Int32, (Int32, Int32, Ptr{Cvoid}, Ref{Csize_t}), platform, param, value, size)
    end

    function iris_set_stream_policy(policy::StreamPolicy)::Int32
        return ccall(Libdl.dlsym(lib, :iris_set_stream_policy), Int32, (StreamPolicy,), policy)
    end

    function iris_set_asynchronous(flag::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_set_asynchronous), Int32, (Int32,), flag)
    end

    function iris_set_shared_memory_model(flag::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_set_shared_memory_model), Int32, (Int32,), flag)
    end

    function iris_mem_enable_usm(mem::IrisMem, type::DeviceModel)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_enable_usm), Int32, (IrisMem, DeviceModel), mem, type)
    end

    function iris_mem_disable_usm(mem::IrisMem, type::DeviceModel)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_disable_usm), Int32, (IrisMem, DeviceModel), mem, type)
    end

    function iris_set_enable_profiler(flag::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_set_enable_profiler), Cvoid, (Int32,), flag)
    end

    function iris_device_count(ndevs::Ref{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_device_count), Int32, (Ref{Int32},), ndevs)
    end

    function iris_ndevices()::Int32
        return ccall(Libdl.dlsym(lib, :iris_ndevices), Int32, ())
    end

    function iris_set_nstreams(n::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_set_nstreams), Int32, (Int32,), n)
    end

    function iris_set_ncopy_streams(n::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_set_ncopy_streams), Int32, (Int32,), n)
    end

    function iris_nstreams()::Int32
        return ccall(Libdl.dlsym(lib, :iris_nstreams), Int32, ())
    end

    function iris_ncopy_streams()::Int32
        return ccall(Libdl.dlsym(lib, :iris_ncopy_streams), Int32, ())
    end

    function iris_device_info(device::Int32, param::Int32, value::Ptr{Cvoid}, size::Ref{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_device_info), Int32, (Int32, Int32, Ptr{Cvoid}, Ref{Csize_t}), device, param, value, size)
    end

    function iris_device_set_default(device::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_device_set_default), Int32, (Int32,), device)
    end

    function iris_device_get_default(device::Ref{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_device_get_default), Int32, (Ref{Int32},), device)
    end

    function iris_device_synchronize(ndevs::Int32, devices::Ptr{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_device_synchronize), Int32, (Int32, Ptr{Int32}), ndevs, devices)
    end

    #function iris_register_policy(lib::Ptr{Cchar}, name::Ptr{Cchar}, params::Ptr{Cvoid})::Int32
    #    return ccall(Libdl.dlsym(lib, :iris_register_policy), Int32, (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cvoid}), lib, name, params)
    #end

    function iris_register_command(tag::Int32, device::Int32, handler::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_register_command), Int32, (Int32, Int32, Ptr{Cvoid}), tag, device, handler)
    end

    function iris_register_hooks_task(pre::Ptr{Cvoid}, post::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_register_hooks_task), Int32, (Ptr{Cvoid}, Ptr{Cvoid}), pre, post)
    end

    function iris_register_hooks_command(pre::Ptr{Cvoid}, post::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_register_hooks_command), Int32, (Ptr{Cvoid}, Ptr{Cvoid}), pre, post)
    end

    function iris_kernel_create_struct(name::String)::IrisKernel
        return ccall(Libdl.dlsym(lib, :iris_kernel_create_struct), IrisKernel, (Ptr{Cchar},), pointer(name))
    end

    function iris_kernel_create(name::String, kernel::Ptr{IrisKernel})::Int32
        return ccall(Libdl.dlsym(lib, :iris_kernel_create), Int32, (Ptr{Cchar}, Ptr{IrisKernel}), pointer(name), kernel)
    end

    function iris_kernel_get(name::String, kernel::Ptr{IrisKernel})::Int32
        return ccall(Libdl.dlsym(lib, :iris_kernel_get), Int32, (Ptr{Cchar}, Ptr{IrisKernel}), pointer(name), kernel)
    end

    function iris_kernel_setarg(kernel::IrisKernel, idx::Int32, size::Csize_t, value::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_kernel_setarg), Int32, (IrisKernel, Int32, Csize_t, Ptr{Cvoid}), kernel, idx, size, value)
    end

    function iris_kernel_setmem(kernel::IrisKernel, idx::Int32, mem::IrisMem, mode::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_kernel_setmem), Int32, (IrisKernel, Int32, IrisMem, Csize_t), kernel, idx, mem, mode)
    end

    function iris_kernel_setmem_off(kernel::IrisKernel, idx::Int32, mem::IrisMem, off::Csize_t, mode::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_kernel_setmem_off), Int32, (IrisKernel, Int32, IrisMem, Csize_t, Csize_t), kernel, idx, mem, off, mode)
    end

    function iris_kernel_release(kernel::IrisKernel)::Int32
        return ccall(Libdl.dlsym(lib, :iris_kernel_release), Int32, (IrisKernel,), kernel)
    end

    function iris_task_create(task::Ptr{IrisTask})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_create), Int32, (Ptr{IrisTask},), task)
    end

    function iris_task_create_struct()::IrisTask
        return ccall(Libdl.dlsym(lib, :iris_task_create_struct), IrisTask, ())
    end

    function iris_task_create_perm(task::Ptr{IrisTask})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_create_perm), Int32, (Ptr{IrisTask},), task)
    end

    function iris_task_create_name(name::String, task::Ptr{IrisTask})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_create_name), Int32, (Ptr{Cchar}, Ptr{IrisTask}), pointer(name), task)
    end

    function iris_task_depend(task::IrisTask, ntasks::Int32, tasks::Ptr{IrisTask})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_depend), Int32, (IrisTask, Int32, Ptr{IrisTask}), task, ntasks, tasks)
    end

    function iris_task_malloc(task::IrisTask, mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_malloc), Int32, (IrisTask, IrisMem), task, mem)
    end

    function iris_task_cmd_reset_mem(task::IrisTask, mem::IrisMem, reset::UInt8)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_cmd_reset_mem), Int32, (IrisTask, IrisMem, UInt8), task, mem, reset)
    end

    function iris_task_set_stream_policy(task::IrisTask, policy::StreamPolicy)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_stream_policy), Int32, (IrisTask, StreamPolicy), task, policy)
    end

    function iris_task_disable_asynchronous(task::IrisTask)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_task_disable_asynchronous), Cvoid, (IrisTask,), task)
    end

    function iris_task_get_metadata(task::IrisTask, index::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_get_metadata), Int32, (IrisTask, Int32), task, index)
    end

    function iris_task_set_metadata(task::IrisTask, index::Int32, metadata::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_metadata), Int32, (IrisTask, Int32, Int32), task, index, metadata)
    end

    function iris_task_h2broadcast(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_h2broadcast), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}), task, mem, off, size, host)
    end

    function iris_task_h2broadcast_offsets(task::IrisTask, mem::IrisMem, off::Ptr{Csize_t}, host_sizes::Ptr{Csize_t}, dev_sizes::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_h2broadcast_offsets), Int32, (IrisTask, IrisMem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Ptr{Cvoid}), task, mem, off, host_sizes, dev_sizes, elem_size, dim, host)
    end

    function iris_task_h2broadcast_full(task::IrisTask, mem::IrisMem, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_h2broadcast_full), Int32, (IrisTask, IrisMem, Ptr{Cvoid}), task, mem, host)
    end

    function iris_task_d2d(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid}, src_dev::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_d2d), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}, Int32), task, mem, off, size, host, src_dev)
    end

    function iris_task_h2d(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_h2d), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}), task, mem, off, size, host)
    end

    function iris_task_h2d_offsets(task::IrisTask, mem::IrisMem, off::Ptr{Csize_t}, host_sizes::Ptr{Csize_t}, dev_sizes::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_h2d_offsets), Int32, (IrisTask, IrisMem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Ptr{Cvoid}), task, mem, off, host_sizes, dev_sizes, elem_size, dim, host)
    end

    function iris_task_d2h(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_d2h), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}), task, mem, off, size, host)
    end

    function iris_task_d2h_offsets(task::IrisTask, mem::IrisMem, off::Ptr{Csize_t}, host_sizes::Ptr{Csize_t}, dev_sizes::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_d2h_offsets), Int32, (IrisTask, IrisMem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Ptr{Cvoid}), task, mem, off, host_sizes, dev_sizes, elem_size, dim, host)
    end

    function iris_task_dmem_flush_out(task::IrisTask, mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_dmem_flush_out), Int32, (IrisTask, IrisMem), task, mem)
    end

    function iris_task_h2d_full(task::IrisTask, mem::IrisMem, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_h2d_full), Int32, (IrisTask, IrisMem, Ptr{Cvoid}), task, mem, host)
    end

    function iris_task_d2h_full(task::IrisTask, mem::IrisMem, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_d2h_full), Int32, (IrisTask, IrisMem, Ptr{Cvoid}), task, mem, host)
    end

    function iris_task_kernel_object(task::IrisTask, kernel::IrisKernel, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_object), Int32, (IrisTask, IrisKernel, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}), task, kernel, dim, off, gws, lws)
    end

    function iris_task_spec(kernel::String, dim::Int64, off::Vector{Int64}, gws::Vector{Int64}, lws::Vector{Int64}, nparams::Int64, params::Vector{Any}, params_info::Vector{Int32})::IrisTask
        task = iris_task_create_struct()
        off_c = Ptr{UInt64}(C_NULL)
        if length(off) != 0
            off_c = reinterpret(Ptr{UInt64}, pointer(off))
        end
        gws_c = Ptr{UInt64}(C_NULL)
        if length(gws) != 0
            gws_c = reinterpret(Ptr{UInt64}, pointer(gws))
        end
        lws_c = Ptr{UInt64}(C_NULL)
        if length(lws) != 0
            lws_c = reinterpret(Ptr{UInt64}, pointer(lws))
        end
        c_params = reinterpret(Ptr{Ptr{Cvoid}}, pointer(params))
        ccall(Libdl.dlsym(lib, :iris_task_kernel), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}), task, pointer(kernel), Int32(dim), off_c, gws_c, lws_c, Int32(nparams), c_params, pointer(params_info))
        return task
    end

    function iris_task_kernel(task::IrisTask, kernel::String, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Int32, params::Ptr{Ptr{Cvoid}}, params_info::Ptr{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}), task, pointer(kernel), dim, off, gws, lws, nparams, params, params_info)
    end

    function iris_task_kernel_v2(task::IrisTask, kernel::String, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Int32, params::Ptr{Ptr{Cvoid}}, params_off::Ptr{Csize_t}, params_info::Ptr{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_v2), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Int32}), task, pointer(kernel), dim, off, gws, lws, nparams, params, params_off, params_info)
    end

    function iris_task_kernel_v3(task::IrisTask, kernel::String, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Int32, params::Ptr{Ptr{Cvoid}}, params_off::Ptr{Csize_t}, params_info::Ptr{Int32}, memranges::Ptr{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_v3), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Int32}, Ptr{Csize_t}), task, pointer(kernel), dim, off, gws, lws, nparams, params, params_off, params_info, memranges)
    end

    function iris_task_kernel_selector(task::IrisTask, func::IrisSelectorKernel, params::Ptr{Cvoid}, params_size::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_selector), Int32, (IrisTask, IrisSelectorKernel, Ptr{Cvoid}, Csize_t), task, func, params, params_size)
    end

    function iris_task_kernel_launch_disabled(task::IrisTask, flag::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_launch_disabled), Int32, (IrisTask, Int32), task, flag)
    end

    #function iris_task_python_host(task::IrisTask, func::IrisHostPythonTask, params_id::Int64)::Int32
    #    return ccall(Libdl.dlsym(lib, :iris_task_python_host), Int32, (IrisTask, IrisHostPythonTask, Int64), task, func, params_id)
    #end

    function iris_task_host(task::IrisTask, func::IrisHostTask, params::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_host), Int32, (IrisTask, IrisHostTask, Ptr{Cvoid}), task, func, params)
    end

    function iris_task_custom(task::IrisTask, tag::Int32, params::Ptr{Cvoid}, params_size::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_custom), Int32, (IrisTask, Int32, Ptr{Cvoid}, Csize_t), task, tag, params, params_size)
    end

    function iris_task_submit(task::IrisTask, device::Int64, opt::Ptr{Int8}, sync::Int64)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_submit), Int32, (IrisTask, Int32, Ptr{Int8}, Int32), task, Int32(device), opt, Int32(sync))
    end

    function iris_task_set_policy(task::IrisTask, device::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_policy), Int32, (IrisTask, Int32), task, device)
    end

    function iris_task_get_policy(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_get_policy), Int32, (IrisTask,), task)
    end

    function iris_task_wait(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_wait), Int32, (IrisTask,), task)
    end

    function iris_task_wait_all(ntasks::Int32, tasks::Ptr{IrisTask})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_wait_all), Int32, (Int32, Ptr{IrisTask}), ntasks, tasks)
    end

    #function iris_task_add_subtask(task::IrisTask, subtask::IrisTask)::Int32
    #    return ccall(Libdl.dlsym(lib, :iris_task_add_subtask), Int32, (IrisTask, IrisTask), task, subtask)
    #end

    function iris_task_kernel_cmd_only(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_cmd_only), Int32, (IrisTask,), task)
    end

    function iris_task_release(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_release), Int32, (IrisTask,), task)
    end

    function iris_task_release_mem(task::IrisTask, mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_release_mem), Int32, (IrisTask, IrisMem), task, mem)
    end

    function iris_params_map(task::IrisTask, params_map::Ptr{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_params_map), Int32, (IrisTask, Ptr{Int32}), task, params_map)
    end

    function iris_task_info(task::IrisTask, param::Int32, value::Ptr{Cvoid}, size::Ref{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_info), Int32, (IrisTask, Int32, Ptr{Cvoid}, Ref{Csize_t}), task, param, value, size)
    end

    function iris_register_pin_memory(host::Ptr{Cvoid}, size::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_register_pin_memory), Int32, (Ptr{Cvoid}, Csize_t), host, size)
    end

    function iris_mem_create(size::Csize_t, mem::Ptr{IrisMem})::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_create), Int32, (Csize_t, Ptr{IrisMem}), size, mem)
    end

    function iris_data_mem_init_reset(mem::IrisMem, reset::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_init_reset), Int32, (IrisMem, Int32), mem, reset)
    end

    function iris_data_mem_create(mem::Ptr{IrisMem}, host::Ptr{Cvoid}, size::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create), Int32, (Ptr{IrisMem}, Ptr{Cvoid}, Csize_t), mem, host, size)
    end

    function iris_data_mem_create_ptr(host::Ptr{Cvoid}, size::Csize_t)::Ptr{IrisMem}
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_ptr), Ptr{IrisMem}, (Ptr{Cvoid}, Csize_t), host, size)
    end

    function iris_data_mem(host::Array{T}) where T 
        size = Csize_t(length(host) * sizeof(T))
        host_cptr = reinterpret(Ptr{Cvoid}, pointer(host))
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct), IrisMem, (Ptr{Cvoid}, Csize_t), host_cptr, size)
    end

    function iris_data_mem_create_struct(host::Ptr{Cvoid}, size::Csize_t)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct), IrisMem, (Ptr{Cvoid}, Csize_t), host, size)
    end

    function iris_data_mem_clear(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_clear), Int32, (IrisMem,), mem)
    end

    function iris_data_mem_pin(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_pin), Int32, (IrisMem,), mem)
    end

    function iris_data_mem_update(mem::IrisMem, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_update), Int32, (IrisMem, Ptr{Cvoid}), mem, host)
    end

    function iris_data_mem_create_region(mem::Ptr{IrisMem}, root_mem::IrisMem, region::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_region), Int32, (Ptr{IrisMem}, IrisMem, Int32), mem, root_mem, region)
    end

    function iris_data_mem_create_region_ptr(root_mem::IrisMem, region::Int32)::Ptr{IrisMem}
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_region_ptr), Ptr{IrisMem}, (IrisMem, Int32), root_mem, region)
    end

    function iris_data_mem_enable_outer_dim_regions(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_enable_outer_dim_regions), Int32, (IrisMem,), mem)
    end

    function iris_data_mem_create_tile(mem::Ptr{IrisMem}, host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile), Int32, (Ptr{IrisMem}, Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), mem, host, off, host_size, dev_size, elem_size, dim)
    end

    function iris_data_mem_create_tile_struct(host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_struct), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), host, off, host_size, dev_size, elem_size, dim)
    end

    function iris_data_mem_create_tile_ptr(host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::Ptr{IrisMem}
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_ptr), Ptr{IrisMem}, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), host, off, host_size, dev_size, elem_size, dim)
    end

    function iris_data_mem_update_bc(mem::IrisMem, bc::Int8, row::Int32, col::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_update_bc), Int32, (IrisMem, Int8, Int32, Int32), mem, bc, row, col)
    end

    function iris_data_mem_get_rr_bc_dev(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_get_rr_bc_dev), Int32, (IrisMem,), mem)
    end

    function iris_data_mem_n_regions(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_n_regions), Int32, (IrisMem,), mem)
    end

    function iris_data_mem_get_region_uid(mem::IrisMem, region::Int32)::Culong
        return ccall(Libdl.dlsym(lib, :iris_data_mem_get_region_uid), Culong, (IrisMem, Int32), mem, region)
    end

    function iris_mem_arch(mem::IrisMem, device::Int32, arch::Ref{Ptr{Cvoid}})::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_arch), Int32, (IrisMem, Int32, Ref{Ptr{Cvoid}}), mem, device, arch)
    end

    #function iris_mem_reduce(mem::IrisMem, mode::Int32, type::Int32)::Int32
    #    return ccall(Libdl.dlsym(lib, :iris_mem_reduce), Int32, (IrisMem, Int32, Int32), mem, mode, type)
    #end

    function iris_mem_release(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_release), Int32, (IrisMem,), mem)
    end

    function iris_graph_create(graph::Ptr{IrisGraph})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_create), Int32, (Ptr{IrisGraph},), graph)
    end

    function iris_graph_create_null(graph::Ptr{IrisGraph})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_create_null), Int32, (Ptr{IrisGraph},), graph)
    end

    function iris_is_graph_null(graph::IrisGraph)::Int8
        return ccall(Libdl.dlsym(lib, :iris_is_graph_null), Int8, (IrisGraph,), graph)
    end

    function iris_graph_free(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_free), Int32, (IrisGraph,), graph)
    end

    function iris_graph_tasks_order(graph::IrisGraph, order::Ptr{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_tasks_order), Int32, (IrisGraph, Ptr{Int32}), graph, order)
    end

    function iris_graph_create_json(json::String, params::Ref{Ptr{Cvoid}}, graph::Ptr{IrisGraph})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_create_json), Int32, (Ptr{Cchar}, Ref{Ptr{Cvoid}}, Ptr{IrisGraph}), pointer(json), params, graph)
    end

    function iris_graph_task(graph::IrisGraph, task::IrisTask, device::Int32, opt::Ptr{Cchar})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_task), Int32, (IrisGraph, IrisTask, Int32, Ptr{Cchar}), graph, task, device, opt)
    end

    function iris_graph_retain(graph::IrisGraph, flag::Int8)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_retain), Int32, (IrisGraph, Int8), graph, flag)
    end

    function iris_graph_release(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_release), Int32, (IrisGraph,), graph)
    end

    function iris_graph_submit(graph::IrisGraph, device::Int32, sync::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_submit), Int32, (IrisGraph, Int32, Int32), graph, device, sync)
    end

    function iris_graph_submit_with_order(graph::IrisGraph, order::Ptr{Int32}, device::Int32, sync::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_submit_with_order), Int32, (IrisGraph, Ptr{Int32}, Int32, Int32), graph, order, device, sync)
    end

    function iris_graph_submit_with_order_and_time(graph::IrisGraph, order::Ptr{Int32}, time::Ref{Cdouble}, device::Int32, sync::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_submit_with_order_and_time), Int32, (IrisGraph, Ptr{Int32}, Ref{Cdouble}, Int32, Int32), graph, order, time, device, sync)
    end

    function iris_graph_submit_with_time(graph::IrisGraph, time::Ref{Cdouble}, device::Int32, sync::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_submit_with_time), Int32, (IrisGraph, Ref{Cdouble}, Int32, Int32), graph, time, device, sync)
    end

    function iris_graph_wait(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_wait), Int32, (IrisGraph,), graph)
    end

    function iris_graph_wait_all(ngraphs::Int32, graphs::Ptr{IrisGraph})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_wait_all), Int32, (Int32, Ptr{IrisGraph}), ngraphs, graphs)
    end

    function iris_record_start()::Int32
        return ccall(Libdl.dlsym(lib, :iris_record_start), Int32, ())
    end

    function iris_record_stop()::Int32
        return ccall(Libdl.dlsym(lib, :iris_record_stop), Int32, ())
    end

    function iris_timer_now(time::Ref{Cdouble})::Int32
        return ccall(Libdl.dlsym(lib, :iris_timer_now), Int32, (Ref{Cdouble},), time)
    end

    function iris_enable_d2d()::Cvoid
        ccall(Libdl.dlsym(lib, :iris_enable_d2d), Cvoid, ())
    end

    function iris_disable_d2d()::Cvoid
        ccall(Libdl.dlsym(lib, :iris_disable_d2d), Cvoid, ())
    end

    function iris_disable_consistency_check()::Cvoid
        ccall(Libdl.dlsym(lib, :iris_disable_consistency_check), Cvoid, ())
    end

    function iris_enable_consistency_check()::Cvoid
        ccall(Libdl.dlsym(lib, :iris_enable_consistency_check), Cvoid, ())
    end

    function iris_kernel_get_name(kernel::IrisKernel)::String
        return unsafe_string(ccall(Libdl.dlsym(lib, :iris_kernel_get_name), Ptr{Cchar}, (IrisKernel,), kernel))
    end

    function iris_task_get_name(task::IrisTask)::String
        return unsafe_string(ccall(Libdl.dlsym(lib, :iris_task_get_name), Ptr{Cchar}, (IrisTask,), task))
    end

    function iris_task_set_name(task::IrisTask, name::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_task_set_name), Cvoid, (IrisTask, Ptr{Cchar}), task, pointer(name))
    end

    function iris_task_get_dependency_count(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_get_dependency_count), Int32, (IrisTask,), task)
    end

    function iris_task_get_dependencies(task::IrisTask, tasks::Ptr{IrisTask})::Cvoid
        ccall(Libdl.dlsym(lib, :iris_task_get_dependencies), Cvoid, (IrisTask, Ptr{IrisTask}), task, tasks)
    end

    function iris_task_get_uid(task::IrisTask)::Culong
        return ccall(Libdl.dlsym(lib, :iris_task_get_uid), UInt64, (IrisTask,), task)
    end

    function iris_kernel_get_uid(kernel::IrisKernel)::Culong
        return ccall(Libdl.dlsym(lib, :iris_kernel_get_uid), Culong, (IrisKernel,), kernel)
    end

    function iris_task_get_kernel(task::IrisTask)::IrisKernel
        return ccall(Libdl.dlsym(lib, :iris_task_get_kernel), IrisKernel, (IrisTask,), task)
    end

    function iris_task_kernel_dmem_fetch_order(task::IrisTask, order::Ptr{Int32})::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_dmem_fetch_order), Int32, (IrisTask, Ptr{Int32}), task, order)
    end

    function iris_task_disable_consistency(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_disable_consistency), Int32, (IrisTask,), task)
    end

    function iris_task_is_cmd_kernel_exists(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_is_cmd_kernel_exists), Int32, (IrisTask,), task)
    end

    function iris_task_get_cmd_kernel(task::IrisTask)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_task_get_cmd_kernel), Ptr{Cvoid}, (IrisTask,), task)
    end

    function iris_mem_get_size(mem::IrisMem)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_mem_get_size), Csize_t, (IrisMem,), mem)
    end

    function iris_mem_get_type(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_get_type), Int32, (IrisMem,), mem)
    end

    function iris_mem_get_uid(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_get_uid), Int32, (IrisMem,), mem)
    end

    function iris_mem_is_reset(mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_is_reset), Int32, (IrisMem,), mem)
    end

    function iris_get_dmem_for_region(dmem_region_obj::IrisMem)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_get_dmem_for_region), IrisMem, (IrisMem,), dmem_region_obj)
    end

    function iris_cmd_kernel_get_nargs(cmd::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_nargs), Int32, (Ptr{Cvoid},), cmd)
    end

    function iris_cmd_kernel_get_arg_is_mem(cmd::Ptr{Cvoid}, index::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_is_mem), Int32, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_size(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_size), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_value(cmd::Ptr{Cvoid}, index::Int32)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_value), Ptr{Cvoid}, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_mem(cmd::Ptr{Cvoid}, index::Int32)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mem), IrisMem, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_mem_off(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mem_off), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_mem_size(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mem_size), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_off(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_off), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_cmd_kernel_get_arg_mode(cmd::Ptr{Cvoid}, index::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mode), Int32, (Ptr{Cvoid}, Int32), cmd, index)
    end

    function iris_graph_enable_mem_profiling(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_enable_mem_profiling), Int32, (IrisGraph,), graph)
    end

    function iris_graph_reset_memories(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_reset_memories), Int32, (IrisGraph,), graph)
    end

    function iris_graph_get_tasks(graph::IrisGraph, tasks::Ptr{IrisTask})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_get_tasks), Int32, (IrisGraph, Ptr{IrisTask}), graph, tasks)
    end

    function iris_graph_tasks_count(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_tasks_count), Int32, (IrisGraph,), graph)
    end

    function iris_get_graph_max_theoretical_parallelism(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_get_graph_max_theoretical_parallelism), Int32, (IrisGraph,), graph)
    end

    function iris_get_graph_dependency_adj_list(graph::IrisGraph, dep_matrix::Ptr{Cchar})::Int32
        return ccall(Libdl.dlsym(lib, :iris_get_graph_dependency_adj_list), Int32, (IrisGraph, Ptr{Cchar}), graph, dep_matrix)
    end

    function iris_get_graph_dependency_adj_matrix(graph::IrisGraph, dep_matrix::Ptr{Cchar})::Int32
        return ccall(Libdl.dlsym(lib, :iris_get_graph_dependency_adj_matrix), Int32, (IrisGraph, Ptr{Cchar}), graph, dep_matrix)
    end

    function iris_get_graph_3d_comm_data_size(graph::IrisGraph)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_data_size), Csize_t, (IrisGraph,), graph)
    end

    function iris_get_graph_3d_comm_data_ptr(graph::IrisGraph)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_data_ptr), Ptr{Cvoid}, (IrisGraph,), graph)
    end

    function iris_get_graph_tasks_execution_schedule(graph::IrisGraph, kernel_profile::Int32)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_get_graph_tasks_execution_schedule), Ptr{Cvoid}, (IrisGraph, Int32), graph, kernel_profile)
    end

    function iris_get_graph_tasks_execution_schedule_count(graph::IrisGraph)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_get_graph_tasks_execution_schedule_count), Csize_t, (IrisGraph,), graph)
    end

    function iris_get_graph_dataobjects_execution_schedule(graph::IrisGraph)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_get_graph_dataobjects_execution_schedule), Ptr{Cvoid}, (IrisGraph,), graph)
    end

    function iris_get_graph_dataobjects_execution_schedule_count(graph::IrisGraph)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_get_graph_dataobjects_execution_schedule_count), Csize_t, (IrisGraph,), graph)
    end

    function iris_get_graph_3d_comm_data(graph::IrisGraph, comm_data::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_data), Int32, (IrisGraph, Ptr{Cvoid}), graph, comm_data)
    end

    function iris_get_graph_2d_comm_adj_matrix(graph::IrisGraph, size_data::Ptr{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_get_graph_2d_comm_adj_matrix), Int32, (IrisGraph, Ptr{Csize_t}), graph, size_data)
    end

    function iris_calibrate_compute_cost_adj_matrix(graph::IrisGraph, comp_data::Ptr{Cdouble})::Int32
        return ccall(Libdl.dlsym(lib, :iris_calibrate_compute_cost_adj_matrix), Int32, (IrisGraph, Ptr{Cdouble}), graph, comp_data)
    end

    function iris_calibrate_compute_cost_adj_matrix_only_for_types(graph::IrisGraph, comp_data::Ptr{Cdouble})::Int32
        return ccall(Libdl.dlsym(lib, :iris_calibrate_compute_cost_adj_matrix_only_for_types), Int32, (IrisGraph, Ptr{Cdouble}), graph, comp_data)
    end

    function iris_calibrate_communication_cost(data::Ptr{Cdouble}, data_size::Csize_t, iterations::Int32, pin_memory_flag::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_calibrate_communication_cost), Int32, (Ptr{Cdouble}, Csize_t, Int32, Int32), data, data_size, iterations, pin_memory_flag)
    end

    function iris_get_graph_3d_comm_time(graph::IrisGraph, comm_time::Ptr{Cdouble}, mem_ids::Ptr{Int32}, iterations::Int32, pin_memory_flag::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_time), Int32, (IrisGraph, Ptr{Cdouble}, Ptr{Int32}, Int32, Int32), graph, comm_time, mem_ids, iterations, pin_memory_flag)
    end

    function iris_count_mems(graph::IrisGraph)::Csize_t
        return ccall(Libdl.dlsym(lib, :iris_count_mems), Csize_t, (IrisGraph,), graph)
    end

    function iris_free_array(ptr::Ptr{Cvoid})::Cvoid
        ccall(Libdl.dlsym(lib, :iris_free_array), Cvoid, (Ptr{Cvoid},), ptr)
    end

    function iris_allocate_array_int8_t(SIZE::Int32, init::Int8)::Ptr{Int8}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_int8_t), Ptr{Int8}, (Int32, Int8), SIZE, init)
    end

    function iris_allocate_array_int16_t(SIZE::Int32, init::Int16)::Ptr{Int16}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_int16_t), Ptr{Int16}, (Int32, Int16), SIZE, init)
    end

    function iris_allocate_array_int32_t(SIZE::Int32, init::Int32)::Ptr{Int32}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_int32_t), Ptr{Int32}, (Int32, Int32), SIZE, init)
    end

    function iris_allocate_array_int64_t(SIZE::Int32, init::Int64)::Ptr{Int64}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_int64_t), Ptr{Int64}, (Int32, Int64), SIZE, init)
    end

    function iris_allocate_array_size_t(SIZE::Int32, init::Csize_t)::Ptr{Csize_t}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_size_t), Ptr{Csize_t}, (Int32, Csize_t), SIZE, init)
    end

    function iris_allocate_array_float(SIZE::Int32, init::Float32)::Ptr{Float32}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_float), Ptr{Float32}, (Int32, Float32), SIZE, init)
    end

    function iris_allocate_array_double(SIZE::Int32, init::Cdouble)::Ptr{Cdouble}
        return ccall(Libdl.dlsym(lib, :iris_allocate_array_double), Ptr{Cdouble}, (Int32, Cdouble), SIZE, init)
    end

    function iris_allocate_random_array_int8_t(SIZE::Int32)::Ptr{Int8}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int8_t), Ptr{Int8}, (Int32,), SIZE)
    end

    function iris_allocate_random_array_int16_t(SIZE::Int32)::Ptr{Int16}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int16_t), Ptr{Int16}, (Int32,), SIZE)
    end

    function iris_allocate_random_array_int32_t(SIZE::Int32)::Ptr{Int32}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int32_t), Ptr{Int32}, (Int32,), SIZE)
    end

    function iris_allocate_random_array_int64_t(SIZE::Int32)::Ptr{Int64}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int64_t), Ptr{Int64}, (Int32,), SIZE)
    end

    function iris_allocate_random_array_size_t(SIZE::Int32)::Ptr{Csize_t}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_size_t), Ptr{Csize_t}, (Int32,), SIZE)
    end

    function iris_allocate_random_array_float(SIZE::Int32)::Ptr{Float32}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_float), Ptr{Float32}, (Int32,), SIZE)
    end

    function iris_allocate_random_array_double(SIZE::Int32)::Ptr{Cdouble}
        return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_double), Ptr{Cdouble}, (Int32,), SIZE)
    end

    function iris_print_matrix_full_double(data::Ptr{Cdouble}, M::Int32, N::Int32, description::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_full_double), Cvoid, (Ptr{Cdouble}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
    end

    function iris_print_matrix_limited_double(data::Ptr{Cdouble}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_double), Cvoid, (Ptr{Cdouble}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
    end

    function iris_print_matrix_full_float(data::Ptr{Float32}, M::Int32, N::Int32, description::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_full_float), Cvoid, (Ptr{Float32}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
    end

    function iris_print_matrix_limited_float(data::Ptr{Float32}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_float), Cvoid, (Ptr{Float32}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
    end

    function iris_print_matrix_full_int64_t(data::Ptr{Int64}, M::Int32, N::Int32, description::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int64_t), Cvoid, (Ptr{Int64}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
    end

    function iris_print_matrix_limited_int64_t(data::Ptr{Int64}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int64_t), Cvoid, (Ptr{Int64}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
    end

    function iris_print_matrix_full_int32_t(data::Ptr{Int32}, M::Int32, N::Int32, description::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int32_t), Cvoid, (Ptr{Int32}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
    end

    function iris_print_matrix_limited_int32_t(data::Ptr{Int32}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int32_t), Cvoid, (Ptr{Int32}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
    end

    function iris_print_matrix_full_int16_t(data::Ptr{Int16}, M::Int32, N::Int32, description::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int16_t), Cvoid, (Ptr{Int16}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
    end

    function iris_print_matrix_limited_int16_t(data::Ptr{Int16}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int16_t), Cvoid, (Ptr{Int16}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
    end

    function iris_print_matrix_full_int8_t(data::Ptr{Int8}, M::Int32, N::Int32, description::String)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int8_t), Cvoid, (Ptr{Int8}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
    end

    function iris_print_matrix_limited_int8_t(data::Ptr{Int8}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int8_t), Cvoid, (Ptr{Int8}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
    end

    function iris_run_hpl_mapping(graph::IrisGraph)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_run_hpl_mapping), Cvoid, (IrisGraph,), graph)
    end

    function iris_read_bool_env(env_name::String)::Int32
        return ccall(Libdl.dlsym(lib, :iris_read_bool_env), Int32, (Ptr{Cchar},), pointer(env_name))
    end

    function iris_read_int_env(env_name::String)::Int32
        return ccall(Libdl.dlsym(lib, :iris_read_int_env), Int32, (Ptr{Cchar},), pointer(env_name))
    end
end  # module Iris


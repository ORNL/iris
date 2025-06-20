@inline function iris_init_scheduler(use_pthread::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_init_scheduler), Int32, (Int32,), Int32(use_pthread))
end
@inline function iris_init_worker(dev::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_init_worker), Int32, (Int32,), Int32(dev))
end
@inline function iris_start_worker(dev::Int, use_pthread::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_start_worker), Int32, (Int32, Int32), Int32(dev), Int32(use_pthread))
end
@inline function iris_init_device(dev::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_init_device), Int32, (Int32,), Int32(dev))
end
@inline function iris_init_devices(sync::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_init_devices), Int32, (Int32,), Int32(sync))
end
@inline function iris_init_devices_synchronize(sync::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_init_devices_synchronize), Int32, (Int32,), Int32(sync))
end

@inline function iris_vendor_kernel_launch(dev::Any, kernel::Ptr{Cvoid}, gridx::Any, gridy::Any, gridz::Any, blockx::Any, blocky::Any, blockz::Any, shared_mem_bytes::Any, stream::Ptr{Cvoid}, params::Ptr{Ptr{Cvoid}})::Int32
    return ccall(Libdl.dlsym(lib, :iris_vendor_kernel_launch), Int32, (Int32, Ptr{Cvoid}, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), Int32(dev), kernel, Int32(gridx), Int32(gridy), Int32(gridz), Int32(blockx), Int32(blocky), Int32(blockz), Int32(shared_mem_bytes), stream, params)
end

@inline function iris_error_count()::Int32
    return ccall(Libdl.dlsym(lib, :iris_error_count), Int32, ())
end

@inline function iris_finalize()::Int32
    return ccall(Libdl.dlsym(lib, :iris_finalize), Int32, ())
end

@inline function iris_synchronize()::Int32
    return ccall(Libdl.dlsym(lib, :iris_synchronize), Int32, ())
end

@inline function iris_task_retain(task::IrisTask, flag::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_task_retain), Cvoid, (IrisTask, Int32), task, flag)
end

@inline function iris_task_set_julia_policy(task::IrisTask, name::String)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_set_julia_policy), Int32, (IrisTask, Ptr{Cchar}), task, pointer(name))
end

@inline function iris_println(data::String)::Cvoid
    return ccall(Libdl.dlsym(lib, :iris_println), Cvoid , (Ptr{Cchar},), pointer(data))
end

@inline function iris_env_set(key::String, value::String)::Int32
    return ccall(Libdl.dlsym(lib, :iris_env_set), Int32, (Ptr{Cchar}, Ptr{Cchar}), pointer(key), pointer(value))
end

@inline function iris_env_get(key::String, value::String, vallen::Ref{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_env_get), Int32, (Ptr{Cchar}, Ref{Ptr{Cchar}}, Ref{Csize_t}), pointer(key), pointer(value), vallen)
end

@inline function iris_overview()::Cvoid
    ccall(Libdl.dlsym(lib, :iris_overview), Cvoid, ())
end

@inline function iris_platform_count(nplatforms::Ref{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_platform_count), Int32, (Ref{Int32},), nplatforms)
end

@inline function iris_platform_info(platform::Int32, param::Int32, value::Ptr{Cvoid}, size::Ref{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_platform_info), Int32, (Int32, Int32, Ptr{Cvoid}, Ref{Csize_t}), platform, param, value, size)
end

@inline function iris_set_stream_policy(policy::StreamPolicy)::Int32
    return ccall(Libdl.dlsym(lib, :iris_set_stream_policy), Int32, (StreamPolicy,), policy)
end

@inline function iris_set_asynchronous(flag::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_set_asynchronous), Int32, (Int32,), flag)
end

@inline function iris_set_shared_memory_model(flag::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_set_shared_memory_model), Int32, (Int32,), flag)
end

@inline function iris_mem_enable_usm(mem::IrisMem, type::DeviceModel)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_enable_usm), Int32, (IrisMem, DeviceModel), mem, type)
end

@inline function iris_mem_disable_usm(mem::IrisMem, type::DeviceModel)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_disable_usm), Int32, (IrisMem, DeviceModel), mem, type)
end

@inline function iris_set_enable_profiler(flag::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_set_enable_profiler), Cvoid, (Int32,), flag)
end

@inline function iris_device_count(ndevs::Ref{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_device_count), Int32, (Ref{Int32},), ndevs)
end

@inline function iris_ndevices()::Int32
    return ccall(Libdl.dlsym(lib, :iris_ndevices), Int32, ())
end

@inline function iris_set_nstreams(n::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_set_nstreams), Int32, (Int32,), n)
end

@inline function iris_set_ncopy_streams(n::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_set_ncopy_streams), Int32, (Int32,), n)
end

@inline function iris_nstreams()::Int32
    return ccall(Libdl.dlsym(lib, :iris_nstreams), Int32, ())
end

@inline function iris_ncopy_streams()::Int32
    return ccall(Libdl.dlsym(lib, :iris_ncopy_streams), Int32, ())
end

@inline function iris_device_info(device::Int32, param::Int32, value::Ptr{Cvoid}, size::Ref{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_device_info), Int32, (Int32, Int32, Ptr{Cvoid}, Ref{Csize_t}), device, param, value, size)
end

@inline function iris_device_set_default(device::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_device_set_default), Int32, (Int32,), device)
end

@inline function iris_device_get_default(device::Ref{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_device_get_default), Int32, (Ref{Int32},), device)
end

@inline function iris_device_synchronize(ndevs::Int32, devices::Ptr{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_device_synchronize), Int32, (Int32, Ptr{Int32}), ndevs, devices)
end

#function iris_register_policy(lib::Ptr{Cchar}, name::Ptr{Cchar}, params::Ptr{Cvoid})::Int32
#    return ccall(Libdl.dlsym(lib, :iris_register_policy), Int32, (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cvoid}), lib, name, params)
#end

@inline function iris_register_command(tag::Int32, device::Int32, handler::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_register_command), Int32, (Int32, Int32, Ptr{Cvoid}), tag, device, handler)
end

@inline function iris_register_hooks_task(pre::Ptr{Cvoid}, post::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_register_hooks_task), Int32, (Ptr{Cvoid}, Ptr{Cvoid}), pre, post)
end

@inline function iris_register_hooks_command(pre::Ptr{Cvoid}, post::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_register_hooks_command), Int32, (Ptr{Cvoid}, Ptr{Cvoid}), pre, post)
end

@inline function iris_kernel_create_struct(name::String)::IrisKernel
    return ccall(Libdl.dlsym(lib, :iris_kernel_create_struct), IrisKernel, (Ptr{Cchar},), pointer(name))
end

@inline function iris_kernel_create(name::String, kernel::Ptr{IrisKernel})::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_create), Int32, (Ptr{Cchar}, Ptr{IrisKernel}), pointer(name), kernel)
end

@inline function iris_kernel_get(name::String, kernel::Ptr{IrisKernel})::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_get), Int32, (Ptr{Cchar}, Ptr{IrisKernel}), pointer(name), kernel)
end

@inline function iris_kernel_set_shared_memory_size(kernel::IrisKernel, size::USize)::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_set_shared_memory_size), Int32, (IrisKernel, Csize_t), kernel, Csize_t(size))
end

@inline function iris_kernel_setarg(kernel::IrisKernel, idx::Int32, size::Csize_t, value::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_setarg), Int32, (IrisKernel, Int32, Csize_t, Ptr{Cvoid}), kernel, idx, size, value)
end

@inline function iris_kernel_setmem(kernel::IrisKernel, idx::Int32, mem::IrisMem, mode::Csize_t)::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_setmem), Int32, (IrisKernel, Int32, IrisMem, Csize_t), kernel, idx, mem, mode)
end

@inline function iris_kernel_setmem_off(kernel::IrisKernel, idx::Int32, mem::IrisMem, off::Csize_t, mode::Csize_t)::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_setmem_off), Int32, (IrisKernel, Int32, IrisMem, Csize_t, Csize_t), kernel, idx, mem, off, mode)
end

@inline function iris_kernel_release(kernel::IrisKernel)::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_release), Int32, (IrisKernel,), kernel)
end

@inline function iris_task_create(task::Ptr{IrisTask})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_create), Int32, (Ptr{IrisTask},), task)
end

@inline function iris_task_create_struct()::IrisTask
    return ccall(Libdl.dlsym(lib, :iris_task_create_struct), IrisTask, ())
end

@inline function iris_task_create_perm(task::Ptr{IrisTask})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_create_perm), Int32, (Ptr{IrisTask},), task)
end

@inline function iris_task_create_name(name::String, task::Ptr{IrisTask})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_create_name), Int32, (Ptr{Cchar}, Ptr{IrisTask}), pointer(name), task)
end

@inline function iris_task_depend(task::IrisTask, ntasks::Int32, tasks::Ptr{IrisTask})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_depend), Int32, (IrisTask, Int32, Ptr{IrisTask}), task, ntasks, tasks)
end

@inline function iris_task_malloc(task::IrisTask, mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_malloc), Int32, (IrisTask, IrisMem), task, mem)
end

@inline function iris_task_cmd_reset_mem(task::IrisTask, mem::IrisMem, reset::UInt8)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_cmd_reset_mem), Int32, (IrisTask, IrisMem, UInt8), task, mem, reset)
end

@inline function iris_task_set_stream_policy(task::IrisTask, policy::StreamPolicy)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_set_stream_policy), Int32, (IrisTask, StreamPolicy), task, policy)
end

@inline function iris_task_disable_asynchronous(task::IrisTask)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_task_disable_asynchronous), Cvoid, (IrisTask,), task)
end

@inline function iris_task_get_metadata_all(task::IrisTask)::Ptr{Int32}
    return ccall(Libdl.dlsym(lib, :iris_task_get_metadata_all), Ptr{Int32}, (IrisTask, ), task)
end

@inline function iris_task_set_metadata_all(task::IrisTask, metadata::Ptr{Int32}, n)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_set_metadata_all), Int32, (IrisTask, Ptr{Int32}, Int32), task, metadata, Int32(n))
end

@inline function iris_task_get_metadata(task::IrisTask, index::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_get_metadata), Int32, (IrisTask, Int32), task, Int32(index))
end

@inline function iris_task_set_metadata(task::IrisTask, index::Int, metadata::Int)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_set_metadata), Int32, (IrisTask, Int32, Int32), task, Int32(index), Int32(metadata))
end

@inline function iris_task_get_metadata_count(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_get_metadata_count), Int32, (IrisTask, ), task)
end

@inline function iris_task_h2broadcast(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_h2broadcast), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}), task, mem, off, size, host)
end

@inline function iris_task_h2broadcast_offsets(task::IrisTask, mem::IrisMem, off::Ptr{Csize_t}, host_sizes::Ptr{Csize_t}, dev_sizes::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_h2broadcast_offsets), Int32, (IrisTask, IrisMem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Ptr{Cvoid}), task, mem, off, host_sizes, dev_sizes, elem_size, dim, host)
end

@inline function iris_task_h2broadcast_full(task::IrisTask, mem::IrisMem, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_h2broadcast_full), Int32, (IrisTask, IrisMem, Ptr{Cvoid}), task, mem, host)
end

@inline function iris_task_d2d(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid}, src_dev::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_d2d), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}, Int32), task, mem, off, size, host, src_dev)
end

@inline function iris_task_dmem2dmem(task::IrisTask, src_mem::IrisMem, dst_mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_dmem2dmem), Int32, (IrisTask, IrisMem, IrisMem), task, src_mem, dst_mem)
end

@inline function iris_task_dmem_d2h(task::IrisTask, mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_dmem_d2h), Int32, (IrisTask, IrisMem), task, mem)
end

@inline function iris_task_dmem_h2d(task::IrisTask, mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_dmem_h2d), Int32, (IrisTask, IrisMem), task, mem)
end

@inline function iris_task_h2d(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_h2d), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}), task, mem, off, size, host)
end

@inline function iris_task_h2d_offsets(task::IrisTask, mem::IrisMem, off::Ptr{Csize_t}, host_sizes::Ptr{Csize_t}, dev_sizes::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_h2d_offsets), Int32, (IrisTask, IrisMem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Ptr{Cvoid}), task, mem, off, host_sizes, dev_sizes, elem_size, dim, host)
end

@inline function iris_task_d2h(task::IrisTask, mem::IrisMem, off::Csize_t, size::Csize_t, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_d2h), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Ptr{Cvoid}), task, mem, off, size, host)
end

@inline function iris_task_d2h_offsets(task::IrisTask, mem::IrisMem, off::Ptr{Csize_t}, host_sizes::Ptr{Csize_t}, dev_sizes::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_d2h_offsets), Int32, (IrisTask, IrisMem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Ptr{Cvoid}), task, mem, off, host_sizes, dev_sizes, elem_size, dim, host)
end

@inline function iris_task_dmem_flush_out(task::IrisTask, mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_dmem_flush_out), Int32, (IrisTask, IrisMem), task, mem)
end

@inline function iris_task_h2d_full(task::IrisTask, mem::IrisMem, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_h2d_full), Int32, (IrisTask, IrisMem, Ptr{Cvoid}), task, mem, host)
end

@inline function iris_task_d2h_full(task::IrisTask, mem::IrisMem, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_d2h_full), Int32, (IrisTask, IrisMem, Ptr{Cvoid}), task, mem, host)
end

@inline function iris_task_kernel_object(task::IrisTask, kernel::IrisKernel, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_object), Int32, (IrisTask, IrisKernel, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}), task, kernel, dim, off, gws, lws)
end

@inline function iris_task_kernel(task::IrisTask, kernel::String, dim::Number, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Number, params::Ptr{Ptr{Cvoid}}, params_info::Ptr{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}), task, pointer(kernel), Int32(dim), off, gws, lws, Int32(nparams), params, params_info)
end

@inline function iris_task_kernel_struct(task::IrisTask, kernel::String, dim::Number, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Number, params::Ptr{Ptr{Cvoid}}, params_info::Ptr{Int32})::IrisKernel
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_struct), IrisKernel, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}), task, pointer(kernel), Int32(dim), off, gws, lws, Int32(nparams), params, params_info)
end

@inline function iris_task_kernel_v2(task::IrisTask, kernel::String, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Int32, params::Ptr{Ptr{Cvoid}}, params_off::Ptr{Csize_t}, params_info::Ptr{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_v2), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Int32}), task, pointer(kernel), dim, off, gws, lws, nparams, params, params_off, params_info)
end

@inline function iris_task_kernel_v3(task::IrisTask, kernel::String, dim::Int32, off::Ptr{Csize_t}, gws::Ptr{Csize_t}, lws::Ptr{Csize_t}, nparams::Int32, params::Ptr{Ptr{Cvoid}}, params_off::Ptr{Csize_t}, params_info::Ptr{Int32}, memranges::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_v3), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Int32}, Ptr{Csize_t}), task, pointer(kernel), dim, off, gws, lws, nparams, params, params_off, params_info, memranges)
end

@inline function iris_task_kernel_selector(task::IrisTask, func::IrisSelectorKernel, params::Ptr{Cvoid}, params_size::Csize_t)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_selector), Int32, (IrisTask, IrisSelectorKernel, Ptr{Cvoid}, Csize_t), task, func, params, params_size)
end

@inline function iris_task_kernel_launch_disabled(task::IrisTask, flag::Int64)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_launch_disabled), Int32, (IrisTask, Int32), task, Int32(flag))
end

#function iris_task_python_host(task::IrisTask, func::IrisHostPythonTask, params_id::Int64)::Int32
#    return ccall(Libdl.dlsym(lib, :iris_task_python_host), Int32, (IrisTask, IrisHostPythonTask, Int64), task, func, params_id)
#end

@inline function iris_task_host(task::IrisTask, func::IrisHostTask, params::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_host), Int32, (IrisTask, IrisHostTask, Ptr{Cvoid}), task, func, params)
end

@inline function iris_task_custom(task::IrisTask, tag::Int32, params::Ptr{Cvoid}, params_size::Csize_t)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_custom), Int32, (IrisTask, Int32, Ptr{Cvoid}, Csize_t), task, tag, params, params_size)
end

@inline function iris_task_submit(task::IrisTask, device_policy::Union{Int64,DeviceModel}, opt::Ptr{Int8}, sync::Int64)::Int32
    #gc_state = @ccall(jl_gc_safe_enter()::Int8)
    status = ccall(Libdl.dlsym(lib, :iris_task_submit), Int32, (IrisTask, Int32, Ptr{Int8}, Int32), task, Int32(device_policy), opt, Int32(sync))
    #@ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
    return status
end

@inline function iris_task_set_policy(task::IrisTask, policy::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_set_policy), Int32, (IrisTask, Int32), task, policy)
end

@inline function iris_task_get_policy(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_get_policy), Int32, (IrisTask,), task)
end

@inline function iris_task_wait(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_wait), Int32, (IrisTask,), task)
end

@inline function iris_task_wait_all(ntasks::Int32, tasks::Ptr{IrisTask})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_wait_all), Int32, (Int32, Ptr{IrisTask}), ntasks, tasks)
end

#function iris_task_add_subtask(task::IrisTask, subtask::IrisTask)::Int32
#    return ccall(Libdl.dlsym(lib, :iris_task_add_subtask), Int32, (IrisTask, IrisTask), task, subtask)
#end

@inline function iris_task_kernel_cmd_only(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_cmd_only), Int32, (IrisTask,), task)
end

@inline function iris_task_release(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_release), Int32, (IrisTask,), task)
end

@inline function iris_task_release_mem(task::IrisTask, mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_release_mem), Int32, (IrisTask, IrisMem), task, mem)
end

@inline function iris_params_map(task::IrisTask, params_map::Ptr{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_params_map), Int32, (IrisTask, Ptr{Int32}), task, params_map)
end

@inline function iris_task_info(task::IrisTask, param::Int32, value::Ptr{Cvoid}, size::Ref{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_info), Int32, (IrisTask, Int32, Ptr{Cvoid}, Ref{Csize_t}), task, param, value, size)
end

@inline function iris_register_pin_memory(host::Ptr{Cvoid}, size::Csize_t)::Int32
    return ccall(Libdl.dlsym(lib, :iris_register_pin_memory), Int32, (Ptr{Cvoid}, Csize_t), host, size)
end

@inline function iris_mem_create(size::Csize_t, mem::Ptr{IrisMem})::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_create), Int32, (Csize_t, Ptr{IrisMem}), size, mem)
end

@inline function iris_mem_init_reset(mem::IrisMem, reset::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_init_reset), Int32, (IrisMem, Int32), mem, reset)
end

@inline function iris_kernel_enable_julia_interface(kernel::IrisKernel, flag::Number)::Int32
    return ccall(Libdl.dlsym(lib, :iris_kernel_enable_julia_interface), Int32, (IrisKernel, Int32), kernel, Int32(flag))
end

@inline function iris_task_enable_julia_interface(task::IrisTask, flag::Number)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_enable_julia_interface), Int32, (IrisTask, Int32), task, Int32(flag))
end

@inline function iris_mem_init_reset_assign(mem::IrisMem, element::Any)::Int32
    ivalue = flat_in_buffer(element)
    return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_assign), Int32, (IrisMem, IRISValue), mem, ivalue)
end

@inline function iris_mem_init_reset_random_uniform_seq(mem::IrisMem, seed::Int, min::Any, max::Any)::Int32
    imin = flat_in_buffer(min)
    imax = flat_in_buffer(max)
    #println("Element: ", element, " buffer:", ivalue)
    return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_random_uniform_seq), Int32, (IrisMem, Clonglong, IRISValue, IRISValue), mem, seed, imin, imax)
end

@inline function iris_mem_init_reset_arith_seq(mem::IrisMem, element::Any, step::Any)::Int32
    ivalue = flat_in_buffer(element)
    istep = flat_in_buffer(step)
    #println("Element: ", element, " buffer:", ivalue)
    return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_arith_seq), Int32, (IrisMem, IRISValue, IRISValue), mem, ivalue, istep)
end

@inline function iris_mem_init_reset_geom_seq(mem::IrisMem, element::Any, step::Any)::Int32
    ivalue = flat_in_buffer(element)
    istep = flat_in_buffer(step)
    #println("Element: ", element, " buffer:", ivalue)
    return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_geom_seq), Int32, (IrisMem, IRISValue, IRISValue), mem, ivalue, istep)
end

@inline function iris_data_mem_init_reset(mem::IrisMem, reset::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_init_reset), Int32, (IrisMem, Int32), mem, reset)
end

@inline function iris_data_mem_copy(mem::IrisMem)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_copy), IrisMem, (IrisMem,), mem)
end

@inline function iris_data_mem_print(mem::IrisMem)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_data_mem_print), Cvoid, (IrisMem,), mem)
end

@inline function iris_data_mem_print(mem::IrisMem, tag::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_data_mem_print_with_tag), Cvoid, (IrisMem, Ptr{Cchar}), mem, pointer(tag))
end

@inline function iris_task_cmd_init_reset_assign(task::IrisTask, mem::IrisMem, element::Any)::Int32
    ivalue = flat_in_buffer(element)
    return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_assign), Int32, (IrisTask, IrisMem, IRISValue), task, mem, ivalue)
end

@inline function iris_task_cmd_init_reset_random_uniform_seq(task::IrisTask, mem::IrisMem, seed::Int, min::Any, max::Any)::Int32
    imin = flat_in_buffer(min)
    imax = flat_in_buffer(max)
    #println("Element: ", element, " buffer:", ivalue)
    return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_random_uniform_seq), Int32, (IrisTask, IrisMem, Clonglong, IRISValue, IRISValue), task, mem, seed, imin, imax)
end

@inline function iris_task_cmd_init_reset_arith_seq(task::IrisTask, mem::IrisMem, element::Any, step::Any)::Int32
    ivalue = flat_in_buffer(element)
    istep = flat_in_buffer(step)
    #println("Element: ", element, " buffer:", ivalue)
    return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_arith_seq), Int32, (IrisTask, IrisMem, IRISValue, IRISValue), task, mem, ivalue, istep)
end

@inline function iris_task_cmd_init_reset_geom_seq(task::IrisTask, mem::IrisMem, element::Any, step::Any)::Int32
    ivalue = flat_in_buffer(element)
    istep = flat_in_buffer(step)
    #println("Element: ", element, " buffer:", ivalue)
    return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_geom_seq), Int32, (IrisTask, IrisMem, IRISValue, IRISValue), task, mem, ivalue, istep)
end

@inline function iris_data_mem_create(mem::Ptr{IrisMem}, host::Ptr{Cvoid}, size::Csize_t)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create), Int32, (Ptr{IrisMem}, Ptr{Cvoid}, Csize_t), mem, host, size)
end

@inline function iris_data_mem_create_ptr(host::Ptr{Cvoid}, size::Csize_t)::Ptr{IrisMem}
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_ptr), Ptr{IrisMem}, (Ptr{Cvoid}, Csize_t), host, size)
end

@inline function iris_data_mem_create_tile_struct_with_type(host_cptr::Ptr{Cvoid}, offset::Any, host_size::Any, dev_size::Any, element_size::Union{Csize_t,UInt64,Int64,Int32,UInt32}, dim::Number, element_type::Number)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_struct_with_type), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Int32), host_cptr, offset, host_size, dev_size, element_size, Int32(dim), Int32(element_type))
end

@inline function iris_data_mem_create_struct_nd(host_cptr::Ptr{Cvoid}, host_size::Any, dim::Number, element_size::Union{Csize_t,UInt64,Int64,Int32,UInt32}, element_type::Number)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_nd), IrisMem, (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Csize_t, Int32), host_cptr, host_size, Int32(dim), Csize_t(element_size), Int32(element_type))
end

@inline function iris_data_mem_create_region(mem::Ptr{IrisMem}, root_mem::IrisMem, region::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_region), Int32, (Ptr{IrisMem}, IrisMem, Int32), mem, root_mem, region)
end

@inline function iris_data_mem_create_region_struct(root_mem::IrisMem, region::Int)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_region_struct), IrisMem, (IrisMem, Int32), root_mem, Int32(region))
end

@inline function iris_data_mem_create_region_ptr(root_mem::IrisMem, region::Int32)::Ptr{IrisMem}
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_region_ptr), Ptr{IrisMem}, (IrisMem, Int32), root_mem, region)
end

@inline function iris_data_mem_enable_outer_dim_regions(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_enable_outer_dim_regions), Int32, (IrisMem,), mem)
end

@inline function iris_data_mem_enable_regions(mem::IrisMem, axis::Union{Int32,Int64,UInt32,UInt64})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_enable_regions), Int32, (IrisMem, Int32), mem, Int32(axis-1))
end

@inline function iris_data_mem_set_valid_regions(mem::IrisMem, n_regions::Union{Int32,Int64,UInt32,UInt64})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_set_valid_regions), Int32, (IrisMem, Int32), mem, Int32(n_regions))
end

@inline function iris_data_mem_create_tile(mem::Ptr{IrisMem}, host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile), Int32, (Ptr{IrisMem}, Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), mem, host, off, host_size, dev_size, elem_size, dim)
end

@inline function iris_data_mem_create_tile_struct(host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_struct), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), host, off, host_size, dev_size, elem_size, dim)
end

@inline function iris_data_mem_create_tile_ptr(host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::Ptr{IrisMem}
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_ptr), Ptr{IrisMem}, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), host, off, host_size, dev_size, elem_size, dim)
end

@inline function iris_data_mem_update_bc(mem::IrisMem, bc::Int32, row::Int32, col::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_bc), Int32, (IrisMem, Int32, Int32, Int32), mem, bc, row, col)
end

@inline function iris_data_mem_get_rr_bc_dev(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_get_rr_bc_dev), Int32, (IrisMem,), mem)
end

@inline function iris_data_mem_n_regions(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_n_regions), Int32, (IrisMem,), mem)
end

@inline function iris_data_mem_get_region_uid(mem::IrisMem, region::Int32)::Culong
    return ccall(Libdl.dlsym(lib, :iris_data_mem_get_region_uid), Culong, (IrisMem, Int32), mem, region)
end

@inline function iris_mem_arch(mem::IrisMem, device::Int32, arch::Ref{Ptr{Cvoid}})::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_arch), Int32, (IrisMem, Int32, Ref{Ptr{Cvoid}}), mem, device, arch)
end

@inline function iris_dev_get_ctx(device::Int)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_dev_get_ctx), Ptr{Cvoid}, (Int32, ), Int32(device))
end

@inline function iris_dev_get_stream(device::Int, index::Int)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_dev_get_stream), Ptr{Cvoid}, (Int32, Int32), Int32(device), Int32(index))
end

@inline function iris_mem_arch_ptr(mem::IrisMem, device::Int)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_mem_arch_ptr), Ptr{Cvoid}, (IrisMem, Int32), mem, Int32(device))
end

#function iris_mem_reduce(mem::IrisMem, mode::Int32, type::Int32)::Int32
#    return ccall(Libdl.dlsym(lib, :iris_mem_reduce), Int32, (IrisMem, Int32, Int32), mem, mode, type)
#end

@inline function iris_mem_release(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_release), Int32, (IrisMem,), mem)
end

@inline function iris_graph_create(graph::Ptr{IrisGraph})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_create), Int32, (Ptr{IrisGraph},), graph)
end

@inline function iris_graph_create_empty()::IrisGraph
    return ccall(Libdl.dlsym(lib, :iris_graph_create_empty), IrisGraph, ())
end

@inline function iris_graph_create_null(graph::Ptr{IrisGraph})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_create_null), Int32, (Ptr{IrisGraph},), graph)
end

@inline function iris_is_graph_null(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_is_graph_null), Int32, (IrisGraph,), graph)
end

@inline function iris_graph_free(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_free), Int32, (IrisGraph,), graph)
end

@inline function iris_graph_tasks_order(graph::IrisGraph, order::Ptr{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_tasks_order), Int32, (IrisGraph, Ptr{Int32}), graph, order)
end

@inline function iris_graph_create_json(json::String, params::Ref{Ptr{Cvoid}}, graph::Ptr{IrisGraph})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_create_json), Int32, (Ptr{Cchar}, Ref{Ptr{Cvoid}}, Ptr{IrisGraph}), pointer(json), params, graph)
end

@inline function iris_graph_task(graph::IrisGraph, task::IrisTask, device::Int32, opt::Ptr{Cchar})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_task), Int32, (IrisGraph, IrisTask, Int32, Ptr{Cchar}), graph, task, device, opt)
end

@inline function iris_graph_retain(graph::IrisGraph, flag::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_retain), Int32, (IrisGraph, Int32), graph, flag)
end

@inline function iris_graph_release(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_release), Int32, (IrisGraph,), graph)
end

@inline function iris_graph_submit(graph::IrisGraph, device::Int32, sync::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_submit), Int32, (IrisGraph, Int32, Int32), graph, device, sync)
end

@inline function iris_graph_submit_with_order(graph::IrisGraph, order::Ptr{Int32}, device::Int32, sync::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_submit_with_order), Int32, (IrisGraph, Ptr{Int32}, Int32, Int32), graph, order, device, sync)
end

@inline function iris_graph_submit_with_order_and_time(graph::IrisGraph, order::Ptr{Int32}, time::Ref{Cdouble}, device::Int32, sync::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_submit_with_order_and_time), Int32, (IrisGraph, Ptr{Int32}, Ref{Cdouble}, Int32, Int32), graph, order, time, device, sync)
end

@inline function iris_graph_submit_with_time(graph::IrisGraph, time::Ref{Cdouble}, device::Int32, sync::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_submit_with_time), Int32, (IrisGraph, Ref{Cdouble}, Int32, Int32), graph, time, device, sync)
end

@inline function iris_graph_wait(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_wait), Int32, (IrisGraph,), graph)
end

@inline function iris_graph_wait_all(ngraphs::Int32, graphs::Ptr{IrisGraph})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_wait_all), Int32, (Int32, Ptr{IrisGraph}), ngraphs, graphs)
end

@inline function iris_record_start()::Int32
    return ccall(Libdl.dlsym(lib, :iris_record_start), Int32, ())
end

@inline function iris_record_stop()::Int32
    return ccall(Libdl.dlsym(lib, :iris_record_stop), Int32, ())
end

@inline function iris_timer_now(time::Ref{Cdouble})::Int32
    return ccall(Libdl.dlsym(lib, :iris_timer_now), Int32, (Ref{Cdouble},), time)
end

@inline function iris_enable_d2d()::Cvoid
    ccall(Libdl.dlsym(lib, :iris_enable_d2d), Cvoid, ())
end

@inline function iris_disable_d2d()::Cvoid
    ccall(Libdl.dlsym(lib, :iris_disable_d2d), Cvoid, ())
end

@inline function iris_disable_consistency_check()::Cvoid
    ccall(Libdl.dlsym(lib, :iris_disable_consistency_check), Cvoid, ())
end

@inline function iris_enable_consistency_check()::Cvoid
    ccall(Libdl.dlsym(lib, :iris_enable_consistency_check), Cvoid, ())
end

@inline function iris_kernel_get_name(kernel::IrisKernel)::String
    return unsafe_string(ccall(Libdl.dlsym(lib, :iris_kernel_get_name), Ptr{Cchar}, (IrisKernel,), kernel))
end

@inline function iris_task_get_name(task::IrisTask)::String
    return unsafe_string(ccall(Libdl.dlsym(lib, :iris_task_get_name), Ptr{Cchar}, (IrisTask,), task))
end

@inline function iris_task_set_name(task::IrisTask, name::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_task_set_name), Cvoid, (IrisTask, Ptr{Cchar}), task, pointer(name))
end

@inline function set_name(task::Union{IrisTask,IrisTaskMem}, name::String)
    iris_task_set_name(field_task(task), name)
end

@inline function iris_task_get_dependency_count(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_get_dependency_count), Int32, (IrisTask,), task)
end

@inline function iris_task_get_dependencies(task::IrisTask, tasks::Ptr{IrisTask})::Cvoid
    ccall(Libdl.dlsym(lib, :iris_task_get_dependencies), Cvoid, (IrisTask, Ptr{IrisTask}), task, tasks)
end

@inline function iris_task_get_uid(task::IrisTask)::Culong
    return ccall(Libdl.dlsym(lib, :iris_task_get_uid), UInt64, (IrisTask,), task)
end

@inline function iris_kernel_get_uid(kernel::IrisKernel)::Culong
    return ccall(Libdl.dlsym(lib, :iris_kernel_get_uid), Culong, (IrisKernel,), kernel)
end

@inline function iris_task_get_kernel(task::IrisTask)::IrisKernel
    return ccall(Libdl.dlsym(lib, :iris_task_get_kernel), IrisKernel, (IrisTask,), task)
end

@inline function iris_task_kernel_dmem_fetch_order(task::IrisTask, order::Ptr{Int32})::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_kernel_dmem_fetch_order), Int32, (IrisTask, Ptr{Int32}), task, order)
end

@inline function iris_task_disable_consistency(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_disable_consistency), Int32, (IrisTask,), task)
end

@inline function iris_task_is_cmd_kernel_exists(task::IrisTask)::Int32
    return ccall(Libdl.dlsym(lib, :iris_task_is_cmd_kernel_exists), Int32, (IrisTask,), task)
end

@inline function iris_task_get_cmd_kernel(task::IrisTask)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_task_get_cmd_kernel), Ptr{Cvoid}, (IrisTask,), task)
end

@inline function iris_mem_get_full_size(mem::IrisMem)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_mem_get_full_size), Csize_t, (IrisMem,), mem)
end

@inline function iris_mem_get_size(mem::IrisMem)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_mem_get_size), Csize_t, (IrisMem,), mem)
end

@inline function iris_mem_get_type(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_get_type), Int32, (IrisMem,), mem)
end

@inline function iris_mem_get_uid(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_get_uid), Int32, (IrisMem,), mem)
end

@inline function iris_mem_is_reset(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_mem_is_reset), Int32, (IrisMem,), mem)
end

@inline function iris_get_dmem_for_region(dmem_region_obj::IrisMem)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_get_dmem_for_region), IrisMem, (IrisMem,), dmem_region_obj)
end

@inline function iris_cmd_kernel_get_nargs(cmd::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_nargs), Int32, (Ptr{Cvoid},), cmd)
end

@inline function iris_cmd_kernel_get_arg_is_mem(cmd::Ptr{Cvoid}, index::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_is_mem), Int32, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_size(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_size), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_value(cmd::Ptr{Cvoid}, index::Int32)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_value), Ptr{Cvoid}, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_mem(cmd::Ptr{Cvoid}, index::Int32)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mem), IrisMem, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_mem_off(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mem_off), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_mem_size(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mem_size), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_off(cmd::Ptr{Cvoid}, index::Int32)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_off), Csize_t, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_cmd_kernel_get_arg_mode(cmd::Ptr{Cvoid}, index::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_cmd_kernel_get_arg_mode), Int32, (Ptr{Cvoid}, Int32), cmd, index)
end

@inline function iris_graph_enable_mem_profiling(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_enable_mem_profiling), Int32, (IrisGraph,), graph)
end

@inline function iris_graph_reset_memories(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_reset_memories), Int32, (IrisGraph,), graph)
end

@inline function iris_graph_get_tasks(graph::IrisGraph, tasks::Ptr{IrisTask})::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_get_tasks), Int32, (IrisGraph, Ptr{IrisTask}), graph, tasks)
end

@inline function iris_graph_tasks_count(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_graph_tasks_count), Int32, (IrisGraph,), graph)
end

@inline function iris_get_graph_max_theoretical_parallelism(graph::IrisGraph)::Int32
    return ccall(Libdl.dlsym(lib, :iris_get_graph_max_theoretical_parallelism), Int32, (IrisGraph,), graph)
end

@inline function iris_get_graph_dependency_adj_list(graph::IrisGraph, dep_matrix::Ptr{Cchar})::Int32
    return ccall(Libdl.dlsym(lib, :iris_get_graph_dependency_adj_list), Int32, (IrisGraph, Ptr{Cchar}), graph, dep_matrix)
end

@inline function iris_get_graph_dependency_adj_matrix(graph::IrisGraph, dep_matrix::Ptr{Cchar})::Int32
    return ccall(Libdl.dlsym(lib, :iris_get_graph_dependency_adj_matrix), Int32, (IrisGraph, Ptr{Cchar}), graph, dep_matrix)
end

@inline function iris_get_graph_3d_comm_data_size(graph::IrisGraph)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_data_size), Csize_t, (IrisGraph,), graph)
end

@inline function iris_get_graph_3d_comm_data_ptr(graph::IrisGraph)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_data_ptr), Ptr{Cvoid}, (IrisGraph,), graph)
end

@inline function iris_get_graph_tasks_execution_schedule(graph::IrisGraph, kernel_profile::Int32)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_get_graph_tasks_execution_schedule), Ptr{Cvoid}, (IrisGraph, Int32), graph, kernel_profile)
end

@inline function iris_get_graph_tasks_execution_schedule_count(graph::IrisGraph)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_get_graph_tasks_execution_schedule_count), Csize_t, (IrisGraph,), graph)
end

@inline function iris_get_graph_dataobjects_execution_schedule(graph::IrisGraph)::Ptr{Cvoid}
    return ccall(Libdl.dlsym(lib, :iris_get_graph_dataobjects_execution_schedule), Ptr{Cvoid}, (IrisGraph,), graph)
end

@inline function iris_get_graph_dataobjects_execution_schedule_count(graph::IrisGraph)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_get_graph_dataobjects_execution_schedule_count), Csize_t, (IrisGraph,), graph)
end

@inline function iris_get_graph_3d_comm_data(graph::IrisGraph, comm_data::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_data), Int32, (IrisGraph, Ptr{Cvoid}), graph, comm_data)
end

@inline function iris_get_graph_2d_comm_adj_matrix(graph::IrisGraph, size_data::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_get_graph_2d_comm_adj_matrix), Int32, (IrisGraph, Ptr{Csize_t}), graph, size_data)
end

@inline function iris_calibrate_compute_cost_adj_matrix(graph::IrisGraph, comp_data::Ptr{Cdouble})::Int32
    return ccall(Libdl.dlsym(lib, :iris_calibrate_compute_cost_adj_matrix), Int32, (IrisGraph, Ptr{Cdouble}), graph, comp_data)
end

@inline function iris_calibrate_compute_cost_adj_matrix_only_for_types(graph::IrisGraph, comp_data::Ptr{Cdouble})::Int32
    return ccall(Libdl.dlsym(lib, :iris_calibrate_compute_cost_adj_matrix_only_for_types), Int32, (IrisGraph, Ptr{Cdouble}), graph, comp_data)
end

@inline function iris_calibrate_communication_cost(data::Ptr{Cdouble}, data_size::Csize_t, iterations::Int32, pin_memory_flag::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_calibrate_communication_cost), Int32, (Ptr{Cdouble}, Csize_t, Int32, Int32), data, data_size, iterations, pin_memory_flag)
end

@inline function iris_get_graph_3d_comm_time(graph::IrisGraph, comm_time::Ptr{Cdouble}, mem_ids::Ptr{Int32}, iterations::Int32, pin_memory_flag::Int32)::Int32
    return ccall(Libdl.dlsym(lib, :iris_get_graph_3d_comm_time), Int32, (IrisGraph, Ptr{Cdouble}, Ptr{Int32}, Int32, Int32), graph, comm_time, mem_ids, iterations, pin_memory_flag)
end

@inline function iris_count_mems(graph::IrisGraph)::Csize_t
    return ccall(Libdl.dlsym(lib, :iris_count_mems), Csize_t, (IrisGraph,), graph)
end

@inline function iris_free_array(ptr::Ptr{Cvoid})::Cvoid
    ccall(Libdl.dlsym(lib, :iris_free_array), Cvoid, (Ptr{Cvoid},), ptr)
end

@inline function iris_allocate_array_int8_t(SIZE::Int32, init::Int8)::Ptr{Int8}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_int8_t), Ptr{Int8}, (Int32, Int8), SIZE, init)
end

@inline function iris_allocate_array_int16_t(SIZE::Int32, init::Int16)::Ptr{Int16}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_int16_t), Ptr{Int16}, (Int32, Int16), SIZE, init)
end

@inline function iris_allocate_array_int32_t(SIZE::Int32, init::Int32)::Ptr{Int32}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_int32_t), Ptr{Int32}, (Int32, Int32), SIZE, init)
end

@inline function iris_allocate_array_int64_t(SIZE::Int32, init::Int64)::Ptr{Int64}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_int64_t), Ptr{Int64}, (Int32, Int64), SIZE, init)
end

@inline function iris_allocate_array_size_t(SIZE::Int32, init::Csize_t)::Ptr{Csize_t}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_size_t), Ptr{Csize_t}, (Int32, Csize_t), SIZE, init)
end

@inline function iris_allocate_array_float(SIZE::Int32, init::Float32)::Ptr{Float32}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_float), Ptr{Float32}, (Int32, Float32), SIZE, init)
end

@inline function iris_allocate_array_double(SIZE::Int32, init::Cdouble)::Ptr{Cdouble}
    return ccall(Libdl.dlsym(lib, :iris_allocate_array_double), Ptr{Cdouble}, (Int32, Cdouble), SIZE, init)
end

@inline function iris_allocate_random_array_int8_t(SIZE::Int32)::Ptr{Int8}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int8_t), Ptr{Int8}, (Int32,), SIZE)
end

@inline function iris_allocate_random_array_int16_t(SIZE::Int32)::Ptr{Int16}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int16_t), Ptr{Int16}, (Int32,), SIZE)
end

@inline function iris_allocate_random_array_int32_t(SIZE::Int32)::Ptr{Int32}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int32_t), Ptr{Int32}, (Int32,), SIZE)
end

@inline function iris_allocate_random_array_int64_t(SIZE::Int32)::Ptr{Int64}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_int64_t), Ptr{Int64}, (Int32,), SIZE)
end

@inline function iris_allocate_random_array_size_t(SIZE::Int32)::Ptr{Csize_t}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_size_t), Ptr{Csize_t}, (Int32,), SIZE)
end

@inline function iris_allocate_random_array_float(SIZE::Int32)::Ptr{Float32}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_float), Ptr{Float32}, (Int32,), SIZE)
end

@inline function iris_allocate_random_array_double(SIZE::Int32)::Ptr{Cdouble}
    return ccall(Libdl.dlsym(lib, :iris_allocate_random_array_double), Ptr{Cdouble}, (Int32,), SIZE)
end

@inline function iris_print_matrix_full_double(data::Ptr{Cdouble}, M::Int32, N::Int32, description::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_full_double), Cvoid, (Ptr{Cdouble}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
end

@inline function iris_print_matrix_limited_double(data::Ptr{Cdouble}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_double), Cvoid, (Ptr{Cdouble}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
end

@inline function iris_print_matrix_full_float(data::Ptr{Float32}, M::Int32, N::Int32, description::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_full_float), Cvoid, (Ptr{Float32}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
end

@inline function iris_print_matrix_limited_float(data::Ptr{Float32}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_float), Cvoid, (Ptr{Float32}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
end

@inline function iris_print_matrix_full_int64_t(data::Ptr{Int64}, M::Int32, N::Int32, description::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int64_t), Cvoid, (Ptr{Int64}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
end

@inline function iris_print_matrix_limited_int64_t(data::Ptr{Int64}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int64_t), Cvoid, (Ptr{Int64}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
end

@inline function iris_print_matrix_full_int32_t(data::Ptr{Int32}, M::Int32, N::Int32, description::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int32_t), Cvoid, (Ptr{Int32}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
end

@inline function iris_print_matrix_limited_int32_t(data::Ptr{Int32}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int32_t), Cvoid, (Ptr{Int32}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
end

@inline function iris_print_matrix_full_int16_t(data::Ptr{Int16}, M::Int32, N::Int32, description::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int16_t), Cvoid, (Ptr{Int16}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
end

@inline function iris_print_matrix_limited_int16_t(data::Ptr{Int16}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int16_t), Cvoid, (Ptr{Int16}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
end

@inline function iris_print_matrix_full_int8_t(data::Ptr{Int8}, M::Int32, N::Int32, description::String)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_full_int8_t), Cvoid, (Ptr{Int8}, Int32, Int32, Ptr{Cchar}), data, M, N, pointer(description))
end

@inline function iris_print_matrix_limited_int8_t(data::Ptr{Int8}, M::Int32, N::Int32, description::String, limit::Int32)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_print_matrix_limited_int8_t), Cvoid, (Ptr{Int8}, Int32, Int32, Ptr{Cchar}, Int32), data, M, N, pointer(description), limit)
end

@inline function iris_run_hpl_mapping(graph::IrisGraph)::Cvoid
    ccall(Libdl.dlsym(lib, :iris_run_hpl_mapping), Cvoid, (IrisGraph,), graph)
end

@inline function iris_read_bool_env(env_name::String)::Int32
    return ccall(Libdl.dlsym(lib, :iris_read_bool_env), Int32, (Ptr{Cchar},), pointer(env_name))
end

@inline function iris_read_int_env(env_name::String)::Int32
    return ccall(Libdl.dlsym(lib, :iris_read_int_env), Int32, (Ptr{Cchar},), pointer(env_name))
end

@inline function iris_dmem_set_source(brs_mem::IrisMem, source::IrisMem)::Ptr{Cint}
    jfunc = Libdl.dlsym(lib, :iris_dmem_set_source)
    return ccall(jfunc, Ptr{Cint}, (IrisMem,IrisMem), brs_mem, source)
end

@inline function iris_dmem_get_source(brs_mem::IrisMem)::IrisMem
    jfunc = Libdl.dlsym(lib, :iris_dmem_get_source)
    return ccall(jfunc, IrisMem, (IrisMem,), brs_mem)
end

# Function: void* iris_get_dmem_host(iris_mem brs_mem);
@inline function iris_get_dmem_host(brs_mem::IrisMem)::Ptr{Cvoid}
    func = Libdl.dlsym(lib, :iris_get_dmem_host)
    return ccall(func, Ptr{Cvoid}, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_elem_size(brs_mem::IrisMem)::Cint
    func = Libdl.dlsym(lib, :iris_dmem_get_elem_size)
    return ccall(func, Cint, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_elem_type(brs_mem::IrisMem)::Cint
    func = Libdl.dlsym(lib, :iris_dmem_get_elem_type)
    return ccall(func, Cint, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_dev_dim(brs_mem::IrisMem)::Cint
    func = Libdl.dlsym(lib, :iris_dmem_get_dev_dim)
    return ccall(func, Cint, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_dim(brs_mem::IrisMem)::Cint
    func = Libdl.dlsym(lib, :iris_dmem_get_dim)
    return ccall(func, Cint, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_host_dim(brs_mem::IrisMem)::Cint
    func = Libdl.dlsym(lib, :iris_dmem_get_host_dim)
    return ccall(func, Cint, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_host_size(brs_mem::IrisMem)::Ptr{Csize_t}
    func = Libdl.dlsym(lib, :iris_dmem_get_host_size)
    return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_host_tile_size(brs_mem::IrisMem)::Ptr{Csize_t}
    func = Libdl.dlsym(lib, :iris_dmem_get_host_tile_size)
    return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_dev_tile_size(brs_mem::IrisMem)::Ptr{Csize_t}
    func = Libdl.dlsym(lib, :iris_dmem_get_dev_tile_size)
    return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_dev_size(brs_mem::IrisMem)::Ptr{Csize_t}
    func = Libdl.dlsym(lib, :iris_dmem_get_dev_size)
    return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_offsets(brs_mem::IrisMem)::Ptr{Csize_t}
    func = Libdl.dlsym(lib, :iris_dmem_get_offsets)
    return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
end

@inline function iris_dmem_get_offsets_from_host(brs_mem::IrisMem)::Ptr{Csize_t}
    func = Libdl.dlsym(lib, :iris_dmem_get_offsets_from_host)
    return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
end

@inline function iris_get_mem_element_type(brs_mem::IrisMem)::Cint
    func = Libdl.dlsym(lib, :iris_get_mem_element_type)
    return ccall(func, Cint, (IrisMem,), brs_mem)
end

# Function: void* iris_get_dmem_valid_host(iris_mem brs_mem);
@inline function iris_get_dmem_valid_host(brs_mem::IrisMem)::Ptr{Cvoid}
    func = Libdl.dlsym(lib, :iris_get_dmem_valid_host)
    return ccall(func, Ptr{Cvoid}, (IrisMem,), brs_mem)
end

# Function: void* iris_get_dmem_host_fetch(iris_mem brs_mem);
@inline function iris_get_dmem_host_fetch(brs_mem::IrisMem)::Ptr{Cvoid}
    func = Libdl.dlsym(lib, :iris_get_dmem_host_fetch)
    return ccall(func, Ptr{Cvoid}, (IrisMem,), brs_mem)
end

# Function: void* iris_get_dmem_host_fetch_with_size(iris_mem brs_mem, size_t size);
@inline function iris_get_dmem_host_fetch_with_size(brs_mem::IrisMem, size::Csize_t)::Ptr{Cvoid}
    func = Libdl.dlsym(lib, :iris_get_dmem_host_fetch_with_size)
    return ccall(func, Ptr{Cvoid}, (IrisMem, Csize_t), brs_mem, size)
end

# Function: int iris_fetch_dmem_data(iris_mem brs_mem, void* host_ptr);
@inline function iris_fetch_dmem_data(brs_mem::IrisMem, host_ptr::Ptr{Cvoid})::Cint
    func = Libdl.dlsym(lib, :iris_fetch_dmem_data)
    return ccall(func, Cint, (IrisMem, Ptr{Cvoid}), brs_mem, host_ptr)
end

# Function: int iris_fetch_dmem_data_with_size(iris_mem brs_mem, void* host_ptr, size_t size);
@inline function iris_fetch_dmem_data_with_size(brs_mem::IrisMem, host_ptr::Ptr{Cvoid}, size::Csize_t)::Cint
    func = Libdl.dlsym(lib, :iris_fetch_dmem_data_with_size)
    return ccall(func, Cint, (IrisMem, Ptr{Cvoid}, Csize_t), brs_mem, host_ptr, size)
end

@inline function iris_data_mem_create_struct(host::Ptr{Cvoid}, size::Csize_t)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct), IrisMem, (Ptr{Cvoid}, Csize_t), host, size)
end

@inline function iris_data_mem_create_struct_with_type(host::Ptr{Cvoid}, size::Csize_t, element_type::Int32)::IrisMem
    return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_with_type), IrisMem, (Ptr{Cvoid}, Csize_t, Int32), host, size, element_type)
end

@inline function iris_data_mem_clear(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_clear), Int32, (IrisMem,), mem)
end

@inline function iris_data_mem_pin(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_pin), Int32, (IrisMem,), mem)
end

@inline function iris_data_mem_set_pin_flag(flag::Bool)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_set_pin_flag), Int32, (Int32,), Int32(flag))
end

@inline function iris_data_mem_update(mem::IrisMem, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update), Int32, (IrisMem, Ptr{Cvoid}), mem, host)
end

@inline function iris_data_mem_copyto_host(mem::IrisMem, host::Ptr{Cvoid})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_copyto_host), Int32, (IrisMem, Ptr{Cvoid}), mem, host)
end

@inline function iris_data_mem_refresh(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_refresh), Int32, (IrisMem, ), mem)
end

@inline function iris_data_mem_fresh_device_data(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_fresh_device_data), Int32, (IrisMem, ), mem)
end

@inline function iris_data_mem_update_host_dim(mem::IrisMem, d::Number)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_host_dim), Int32, (IrisMem, Int32,), mem, Int32(d))
end

@inline function iris_data_mem_update_dim(mem::IrisMem, d::Number)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_dim), Int32, (IrisMem, Int32,), mem, Int32(d))
end

@inline function iris_data_mem_update_host_offs(mem::IrisMem, offs::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_host_offs), Int32, (IrisMem, Ptr{Csize_t}), mem, offs)
end

@inline function iris_data_mem_update_dev_tile_size(mem::IrisMem, host::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_dev_tile_size), Int32, (IrisMem, Ptr{Csize_t}), mem, host)
end

@inline function iris_data_mem_update_host_tile_size(mem::IrisMem, host::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_host_tile_size), Int32, (IrisMem, Ptr{Csize_t}), mem, host)
end

@inline function iris_data_mem_update_host_size(mem::IrisMem, host::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_host_size), Int32, (IrisMem, Ptr{Csize_t}), mem, host)
end

@inline function iris_data_mem_update_host_size_no_strides(mem::IrisMem, host::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_host_size_no_strides), Int32, (IrisMem, Ptr{Csize_t}), mem, host)
end

@inline function iris_data_mem_update_dev_size_no_strides(mem::IrisMem, dev_size::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_dev_size_no_strides), Int32, (IrisMem, Ptr{Csize_t}), mem, dev_size)
end

@inline function iris_data_mem_update_dev_size(mem::IrisMem, dev_size::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_dev_size), Int32, (IrisMem, Ptr{Csize_t}), mem, dev_size)
end

@inline function iris_data_mem_update_off(mem::IrisMem, dev_size::Ptr{Csize_t})::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_update_off), Int32, (IrisMem, Ptr{Csize_t}), mem, dev_size)
end

@inline function iris_data_mem_is_size_preserved(mem::IrisMem)::Int32
    return ccall(Libdl.dlsym(lib, :iris_data_mem_is_size_preserved), Int32, (IrisMem, ), mem)
end

@inline function iris_julia_init(func_ptr::Ptr{Cvoid}, flag::Number)::Int32
    return ccall(Libdl.dlsym(lib, :iris_julia_init), Int32, (Ptr{Cvoid}, Int32), func_ptr, Int32(flag))
end

@inline function iris_init_c(sync::Number)
    return ccall(Libdl.dlsym(lib, :iris_init), Int32, (Ref{Int32}, Ref{Ptr{Ptr{Cchar}}}, Int32), Int32(1), C_NULL, Int32(sync))
end

@inline function iris_julia_policy_init(policy_ptr::Ptr{Cvoid})
    return ccall(Libdl.dlsym(lib, :iris_julia_policy_init), Int32, (Ptr{Cvoid},), policy_ptr)
end


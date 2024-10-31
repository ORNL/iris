#####################################################
#   Author: Narasinga Rao Miniskar
#   Date: 06/06/2024
#   File: IrisHRT.jl
#   Contact: miniskarnr@ornl.gov
#   Comment: IRIS Julia interface
#####################################################
#import Pkg
#import Pkg.add("CUDA")
#import Pkg.add("AMDGPU")
module IrisHRT
    using Requires
    using Base.Threads: @spawn
    #__precompile__(false)
    const cuda_available = try
        using CUDA
        true
    catch 
        false
    end
    const hip_available = try
        using AMDGPU
        AMDGPU.has_rocm_gpu()
    catch
        false
    end
    using Libdl
    cwd = pwd()
    const iris_kernel_jl = cwd * "/kernel.jl"
    println("Kernel file: $iris_kernel_jl")
    include(iris_kernel_jl)
    using .IrisKernelImpl
    libiris = ENV["IRIS"] * "/lib/" * "libiris.so"
    if !isfile(libiris)
        libiris = "libiris.so"
    end
    # Load the shared library
    if !haskey(ENV, "IRIS_ARCHS")
        ENV["IRIS_ARCHS"] = "cuda:hip:openmp"
    end
    lib = Libdl.dlopen(libiris)
    println(Core.stdout, "PWD: $cwd")

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

    const iris_unknown       = 0 << 16
    const iris_int           = 1 << 16
    const iris_uint          = 2 << 16
    const iris_float         = 3 << 16
    const iris_double        = 4 << 16
    const iris_char          = 5 << 16
    const iris_int8          = 6 << 16
    const iris_uint8         = 7 << 16
    const iris_int16         = 8 << 16
    const iris_uint16        = 9 << 16
    const iris_int32         = 10 << 16
    const iris_uint32        = 11 << 16
    const iris_int64         = 12 << 16
    const iris_uint64        = 13 << 16
    const iris_long          = 14 << 16
    const iris_unsigned_long = 15 << 16
    const iris_pointer       = 16384 << 16

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

    macro maybethreads(ex)
       if Threads.nthreads() > 1
            esc(:(Threads.@threads $ex))
       else
            esc(ex)
       end
    end
    # Define function pointer types
    const IrisHostTask = Ptr{Cvoid}
    const IrisHostPythonTask = Ptr{Cvoid}
    const CommandHandler = Ptr{Cvoid}
    const HookTask = Ptr{Cvoid}
    const HookCommand = Ptr{Cvoid}
    const IrisSelectorKernel = Ptr{Cvoid}

    export iris_init
    export kernel_julia_wrapper
    export convert_args

    function convert_args(args_type::Ptr{Cint}, args::Ptr{Ptr{Cvoid}}, nparams::Cint, start_index::Cint)
        julia_args = []
        return julia_args
    end 
    function iris_print_array_size_t(data::Ptr{Csize_t}, index::Int)::Cvoid
        ldata = UInt64(unsafe_load(data + index*sizeof(Cint)))
        iris_println("Array:$index Data:$ldata")
    end
    function iris_print_array_int(data::Ptr{Cint}, index::Int)::Cvoid
        ldata = Int(unsafe_load(data + index*sizeof(Cint)))
        iris_println("Array:$index Data:$ldata")
    end
    function kernel_julia_wrapper(target::Cint, devno::Cint, ctx::Ptr{Cvoid}, async_flag::Cchar, stream_index::Cint, stream::Ptr{Ptr{Cvoid}}, nstreams::Cint, args_type::Ptr{Cint}, args::Ptr{Ptr{Cvoid}}, param_size::Ptr{Csize_t}, param_dim_size::Ptr{Csize_t}, nparams::Cint, c_threads::Ptr{Csize_t}, c_blocks::Ptr{Csize_t}, dim::Cint, kernel_name::Ptr{Cchar})::Cint
        # Array to hold the converted Julia values
        #julia_args = convert_args(args_type, args, nparams, 2)
        #iris_println("nstreams: $nstreams, steram:$stream ctx:$ctx stream_index:$stream_index")
        b_async_flag = Bool(async_flag)
        julia_args = []
        julia_sizes = []
        julia_dims = []
        start_index = 2
        DMEM_MAX_DIM = 6
        #iris_println("args_type:$args_type args:$args nparams:$nparams")
        #iris_print_array(param_size, Int(nparams))
        # Iterate through each argument
        dev_ptr_index = []
        for i in start_index:(nparams - 1)
            single_arg_type = unsafe_load(args_type + i*sizeof(Cint))
            # Load the type of the current argument
            current_type = Int(single_arg_type & 0x3FFF0000)
            is_pointer = Int(single_arg_type & 0x40000000)
            element_size = Int((single_arg_type >> 8) & 0xFF)
            dim = Int(single_arg_type & 0xFF)
            full_size = UInt64(unsafe_load(param_size + i*sizeof(Csize_t)))
            size_dims = [] 
            for d in 0:(dim - 1)
                value = unsafe_load(param_dim_size + i*sizeof(Csize_t)*DMEM_MAX_DIM + d * sizeof(Csize_t)) + 0
                push!(size_dims, Int64(value))
            end
            size_dims = Tuple(size_dims)
            # Load the pointer to the actual argument
            arg_ptr = unsafe_load(Ptr{Ptr{Cvoid}}(args + i*sizeof(Ptr{Cvoid})))
            #iris_println("I:$i type:$current_type ptr:$arg_ptr is_pointer:$is_pointer element_size:$element_size dim:$dim Size:$full_size SizeDim:$size_dims")
            # Convert based on the type
            if is_pointer != Int(iris_pointer)
                #iris_println("I is not pointer")
                if current_type == Int(iris_int64)  # Assuming 1 is for Int
                    push!(julia_args, Int64(unsafe_load(Ptr{Clonglong}(arg_ptr))))
                elseif current_type == Int(iris_int32)  # Assuming 1 is for Int
                    push!(julia_args, Int32(unsafe_load(Ptr{Cint}(arg_ptr))))
                elseif current_type == Int(iris_int16)  # Assuming 1 is for Int
                    push!(julia_args, Int16(unsafe_load(Ptr{Cshort}(arg_ptr))))
                elseif current_type == Int(iris_int8)  # Assuming 1 is for Int
                    push!(julia_args, Int8(unsafe_load(Ptr{Cchar}(arg_ptr))))
                elseif current_type == Int(iris_uint64)  # Assuming 1 is for Int
                    push!(julia_args, UInt64(unsafe_load(Ptr{Culonglong}(arg_ptr))))
                elseif current_type == Int(iris_uint32)  # Assuming 1 is for Int
                    push!(julia_args, UInt32(unsafe_load(Ptr{Cuint}(arg_ptr))))
                elseif current_type == Int(iris_uint16)  # Assuming 1 is for Int
                    push!(julia_args, UInt16(unsafe_load(Ptr{Cushort}(arg_ptr))))
                elseif current_type == Int(iris_uint8)  # Assuming 1 is for Int
                    push!(julia_args, UInt8(unsafe_load(Ptr{Cuchar}(arg_ptr))))
                elseif current_type == Int(iris_float)  # Assuming 2 is for Float
                    push!(julia_args, Float32(unsafe_load(Ptr{Cfloat}(arg_ptr))))
                elseif current_type == Int(iris_double) # Assuming 2 is for Float
                    push!(julia_args, Float64(unsafe_load(Ptr{Cdouble}(arg_ptr))))
                elseif current_type == Int(iris_char)  # Assuming 3 is for Char
                    push!(julia_args, Char(unsafe_load(Ptr{Cchar}(arg_ptr))))
                else
                    error("Unsupported type")
                end
            else # Assuming 3 is for Char
                push!(dev_ptr_index, i+1-start_index)
                if current_type == Int(iris_float)
                    arg_ptr = Ptr{Cfloat}(arg_ptr)
                elseif current_type == Int(iris_double)
                    arg_ptr = Ptr{Cdouble}(arg_ptr)
                elseif current_type == Int(iris_int64)
                    arg_ptr = Ptr{Clonglong}(arg_ptr)
                elseif current_type == Int(iris_int32)
                    arg_ptr = Ptr{Cint}(arg_ptr)
                elseif current_type == Int(iris_int16)
                    arg_ptr = Ptr{Cshort}(arg_ptr)
                elseif current_type == Int(iris_int8)
                    arg_ptr = Ptr{Cchar}(arg_ptr)
                elseif current_type == Int(iris_char)
                    arg_ptr = Ptr{Cchar}(arg_ptr)
                elseif current_type == Int(iris_uint64)
                    arg_ptr = Ptr{Culonglong}(arg_ptr)
                elseif current_type == Int(iris_uint32)
                    arg_ptr = Ptr{Cuint}(arg_ptr)
                elseif current_type == Int(iris_uint16)
                    arg_ptr = Ptr{Cushort}(arg_ptr)
                elseif current_type == Int(iris_uint8)
                    arg_ptr = Ptr{Cuchar}(arg_ptr)
                else
                    iris_println("[Julia-IrisHRT-Error] No type present")
                end
                try 
                    #jptr = unsafe_wrap(CUDA.Array, arg_ptr, size_dims, own=false)
                    #jptr = unsafe_wrap(CuDeviceArray{Float32, 1}, arg_ptr, size_dims, own=false)
                    #iris_println("After Pointer $arg_ptr $jptr")
                    #push!(julia_args, jptr)
                    push!(julia_args, (arg_ptr, size_dims, current_type, full_size, element_size))
                catch e 
                    iris_println("[Julia-IrisHRT-Error] exception raised during push of arguments to julia_args")
                    rethrow(e)
                end
            end
        end
        #iris_println("-----Hello:$target Devno:$devno nparams:$nparams" * vec_string(julia_args))
        for i in dev_ptr_index
            (arg_ptr, size_dims, current_type, full_size, element_size)  = julia_args[i]
            #iris_println("Index: $i arg:$arg_ptr ---")
            try 
                #type_name = CuPtr{Float32}
                #cu_arg_ptr = reinterpret(type_name, arg_ptr)
                if Int(target) == Int(iris_cuda)
                    arg_ptr = cuda_reinterpret(arg_ptr, current_type)
                    #iris_println("Index: $i target arg:$arg_ptr size:$size_dims")
                    arg_ptr = unsafe_wrap(CuArray, arg_ptr, size_dims, own=false)
                    #iris_println("After Pointer $arg_ptr")
                elseif Int(target) == Int(iris_hip)
                    arg_ptr = ptr_reinterpret(arg_ptr, current_type)
                    #iris_println("Index: $i target arg:$arg_ptr size:$size_dims")
                    arg_ptr = unsafe_wrap(ROCArray, arg_ptr, size_dims, lock=false)
                    #iris_println("After Pointer $arg_ptr")
                elseif Int(target) == Int(iris_openmp)
                    arg_ptr = ptr_reinterpret(arg_ptr, current_type)
                    #iris_println("Index: $i target arg:$arg_ptr size:$size_dims")
                    arg_ptr = unsafe_wrap(Array, arg_ptr, size_dims, own=false)
                    #iris_println("After Pointer $arg_ptr")
                else
                    iris_println("[Julia-IrisHRT-Error] Unknown target to handle arguments")
                end
                julia_args[i] = arg_ptr
                #push!(julia_args, arg_ptr)
            catch e 
                iris_println("[Julia-IrisHRT-Error] exception raised during conversion to device specific type")
                rethrow(e)
            end
        end
        #iris_println("Hello:$target Devno:$devno nparams:$nparams" * vec_string(julia_args))
        #iris_println("Now calling CUDA")
        j_threads = []
        j_blocks = []
        # Iterate through each argument
        for i in 0:(dim - 1)
            # Load the type of the current argument
            push!(j_threads, UInt64(unsafe_load(c_threads+i*sizeof(Csize_t)))+0)
            push!(j_blocks, UInt64(unsafe_load(c_blocks+i*sizeof(Csize_t)))+0)
        end
        j_threads = Tuple(j_threads)
        j_blocks = Tuple(j_blocks)
        if Int(target) == Int(iris_cuda)
            #cu_ctx = unsafe_load(Ref{CuContext}(ctx))
            #GC.gc()
            cu_ctx = unsafe_load(reinterpret(Ptr{CuContext}, ctx))
            cu_stream = nothing
            if b_async_flag 
                cu_stream = unsafe_load(reinterpret(Ptr{CuStream}, stream))
            end
            #iris_println("Stream: $cu_stream dev:$devno")
            #iris_println("Ctx: $cu_ctx dev:$devno")
            func_name = unsafe_string(kernel_name)
            CUDA.device!(Int(devno))
            CUDA.context!(cu_ctx)
            iris_println("Calling CUDA kernel")
            call_cuda_kernel(func_name, j_threads, j_blocks, cu_stream, julia_args, b_async_flag)
            #CUDA.@sync @cuda threads=j_threads blocks=j_blocks Main.saxpy_cuda(julia_args[1], julia_args[2], julia_args[3], julia_args[4])
            ##############################################################
            iris_println("Completed CUDA")
            return target*11
        elseif Int(target) == Int(iris_hip)
            #GC.gc()
            hip_ctx = unsafe_load(reinterpret(Ptr{HIPContext}, ctx))
            hip_stream = nothing
            if b_async_flag 
                hip_stream = unsafe_load(reinterpret(Ptr{HIPStream}, stream))
            end
            #iris_println("----Ctx: $hip_ctx dev:$devno")
            func_name = unsafe_string(kernel_name)
            AMDGPU.device!(AMDGPU.devices()[devno+1])
            AMDGPU.context!(hip_ctx)
            iris_println("Calling HIP kernel")
            call_hip_kernel(func_name, j_threads, j_blocks, hip_stream, julia_args, b_async_flag)
            ##############################################################
            iris_println("Completed HIP")
            return target*12
        elseif Int(target) == Int(iris_openmp)
            #cu_ctx = unsafe_load(Ref{CuContext}(ctx))
            #GC.gc()
            #iris_println("dev:$devno")
            func_name = unsafe_string(kernel_name)
            #iris_println("Calling OpenMP kernel")
            call_openmp_kernel(func_name, j_threads, j_blocks, julia_args, b_async_flag)
            ##############################################################
            #iris_println("Completed OpenMP")
            return target*13
        else
            return target*10
        end
    end

    function cuda_reinterpret(arg_ptr::Any, current_type::Any)
        if current_type == Int(iris_float)
            arg_ptr = reinterpret(CuPtr{Float32}, arg_ptr)
        elseif current_type == Int(iris_double)
            arg_ptr = reinterpret(CuPtr{Float64}, arg_ptr)
        elseif current_type == Int(iris_int64)
            arg_ptr = reinterpret(CuPtr{Int64}, arg_ptr)
        elseif current_type == Int(iris_int32)
            arg_ptr = reinterpret(CuPtr{Int32}, arg_ptr)
        elseif current_type == Int(iris_int16)
            arg_ptr = reinterpret(CuPtr{Int16}, arg_ptr)
        elseif current_type == Int(iris_int8)
            arg_ptr = reinterpret(CuPtr{Int8}, arg_ptr)
        elseif current_type == Int(iris_char)
            arg_ptr = reinterpret(CuPtr{Char}, arg_ptr)
        elseif current_type == Int(iris_uint64)
            arg_ptr = reinterpret(CuPtr{UInt64}, arg_ptr)
        elseif current_type == Int(iris_uint32)
            arg_ptr = reinterpret(CuPtr{UInt32}, arg_ptr)
        elseif current_type == Int(iris_uint16)
            arg_ptr = reinterpret(CuPtr{UInt16}, arg_ptr)
        elseif current_type == Int(iris_uint8)
            arg_ptr = reinterpret(CuPtr{UInt8}, arg_ptr)
        else
            iris_println("[Julia-IrisHRT-Error][cuda_reinterpret] Unknown type")
        end
        return arg_ptr
    end

    function ptr_reinterpret(arg_ptr::Any, current_type::Any)
        if current_type == Int(iris_float)
            arg_ptr = reinterpret(Ptr{Float32}, arg_ptr)
        elseif current_type == Int(iris_double)
            arg_ptr = reinterpret(Ptr{Float64}, arg_ptr)
        elseif current_type == Int(iris_int64)
            arg_ptr = reinterpret(Ptr{Int64}, arg_ptr)
        elseif current_type == Int(iris_int32)
            arg_ptr = reinterpret(Ptr{Int32}, arg_ptr)
        elseif current_type == Int(iris_int16)
            arg_ptr = reinterpret(Ptr{Int16}, arg_ptr)
        elseif current_type == Int(iris_int8)
            arg_ptr = reinterpret(Ptr{Int8}, arg_ptr)
        elseif current_type == Int(iris_char)
            arg_ptr = reinterpret(Ptr{Char}, arg_ptr)
        elseif current_type == Int(iris_uint64)
            arg_ptr = reinterpret(Ptr{UInt64}, arg_ptr)
        elseif current_type == Int(iris_uint32)
            arg_ptr = reinterpret(Ptr{UInt32}, arg_ptr)
        elseif current_type == Int(iris_uint16)
            arg_ptr = reinterpret(Ptr{UInt16}, arg_ptr)
        elseif current_type == Int(iris_uint8)
            arg_ptr = reinterpret(Ptr{UInt8}, arg_ptr)
        else
            iris_println("[Julia-IrisHRT-Error][ptr_reinterpret] Unknown type")
        end
        return arg_ptr
    end

    function iris_print_array(data::Ptr{T}, N::Int) where T
        str = []
        arr = unsafe_wrap(Array, data, N)
        for i in 1:N
            value = string.(arr[i])
            #iris_println("Data Value: $value")
            push!(str, value)
        end
        str_data = "[" * join(str, ",") * "]"
        iris_println("Data: $str_data")
    end
    function vec_string(arr::AbstractArray, sep::String = ", ")
        return "[" * join(string.(arr), sep) * "]"
    end

    # Bind functions from the IRIS library using ccall
    function iris_init(sync)::Int32
        func_ptr = @cfunction(kernel_julia_wrapper, Cint, (Cint,        Cint,        Ptr{Cvoid}, Cchar, Cint,               Ptr{Ptr{Cvoid}},         Cint,           Ptr{Cint},       Ptr{Ptr{Cvoid}},         Ptr{Csize_t},  Ptr{Csize_t}, Cint,          Ptr{Csize_t},          Ptr{Csize_t},         Cint,      Ptr{Cchar}))
        global stored_func_ptr = func_ptr
        use_julia_threads = false
        if use_julia_threads
            flag = ccall(Libdl.dlsym(lib, :iris_julia_init), Int32, (Ptr{Cvoid}, Int32), func_ptr, Int32(1))
            flag = flag & ccall(Libdl.dlsym(lib, :iris_init), Int32, (Ref{Int32}, Ref{Ptr{Ptr{Cchar}}}, Int32), Int32(1), C_NULL, Int32(sync))
            ndevs = iris_ndevices()
            t = @spawn begin 
                iris_init_scheduler(0)
            end
            println(Core.stdout, "Julia: N devices $ndevs")
            for i in 0:(ndevs- 1)
                println(Core.stdout, "Julia: Initializing Worker $i")
                flag = flag & iris_init_worker(i)
            end
            t = nothing
            for i in 0:(ndevs- 1)
                println(Core.stdout, "Julia: Starting Worker $i")
                t = @spawn begin
                    iris_start_worker(i, 0)
                end
                println(Core.stdout, t)
            end
            for i in 0:(ndevs- 1)
                println(Core.stdout, "Julia: Initializing device $i")
                flag = flag & iris_init_device(i)
            end
            println(Core.stdout, "Julia: Doing device synchronize")
            println(Core.stdout, t)
            flush(stdout)
            flag = flag & iris_init_devices_synchronize(1)
            println(Core.stdout, "Julia: Completed device synchronize")
        else
            flag = ccall(Libdl.dlsym(lib, :iris_julia_init), Int32, (Ptr{Cvoid}, Int32), func_ptr, Int32(0))
            flag = ccall(Libdl.dlsym(lib, :iris_init), Int32, (Ref{Int32}, Ref{Ptr{Ptr{Cchar}}}, Int32), Int32(1), C_NULL, Int32(sync))
        end
        return flag
    end

    function iris_init_scheduler(use_pthread::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_init_scheduler), Int32, (Int32,), Int32(use_pthread))
    end
    function iris_init_worker(dev::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_init_worker), Int32, (Int32,), Int32(dev))
    end
    function iris_start_worker(dev::Int, use_pthread::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_start_worker), Int32, (Int32, Int32), Int32(dev), Int32(use_pthread))
    end
    function iris_init_device(dev::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_init_device), Int32, (Int32,), Int32(dev))
    end
    function iris_init_devices(sync::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_init_devices), Int32, (Int32,), Int32(sync))
    end
    function iris_init_devices_synchronize(sync::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_init_devices_synchronize), Int32, (Int32,), Int32(sync))
    end
    # Call Julia kernel 
    # CUDA kernel wrapper
    function call_cuda_kernel(func_name::String, threads::Any, blocks::Any, stream::Any, args::Any, async_flag::Bool=false)
        if !cuda_available
            error("CUDA device not available.")
        end

        func_name_target = func_name * "_cuda"
        iris_println("CUDA func name: $func_name_target")
        # Initialize CUDA
        #CUDA.allowscalar(false)  # Disable scalar operations on the GPU
        func = getfield(Main, Symbol(func_name_target))
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #println(Core.stdout, "Args_tuple: $args_tuple")
        println(Core.stdout, "Async $async_flag")
        if async_flag
            iris_println("Asynchronous CUDA execution")
            CUDA.@async @cuda threads=threads blocks=blocks stream=stream func(args_tuple...)
        else
            iris_println("---------Synchronous CUDA execution $blocks $threads func:$func----------")
            gc_state = @ccall(jl_gc_safe_enter()::Int8)
            CUDA.@sync begin
                @cuda threads=threads blocks=blocks saxpy_cuda(args_tuple...)
            end
            @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)

            #CUDA.@sync @cuda threads=threads blocks=blocks func(args_tuple...)
            #func1(threads, blocks, args_tuple)
        end
        #synchronize(blocking = true)
    end
    function saxpy_cuda(Z, A, X, Y)
        # Calculate global index
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        @inbounds Z[i] = A * X[i] + Y[i]
        return nothing
    end

    # HIP kernel wrapper
    function call_hip_kernel(func_name::String, threads::Any, blocks::Any, stream::Any, args::Any, async_flag::Bool=false)
        if !hip_available
            error("HIP device not available.")
        end

        func_name_target = func_name * "_hip"
        # Initialize CUDA
        #AMDGPU.allowscalar(false)  # Disable scalar operations on the GPU
        func = getfield(Main, Symbol(func_name_target))
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #@hip threads=threads blocks=blocks add_kernel(a,b,c,N)
        #println(Core.stdout, "Args_tuple: $args_tuple")
        #println(Core.stdout, "Threads: $threads blocks:$blocks")
        if async_flag
            AMDGPU.@async @roc groupsize=threads gridsize=blocks stream=stream func(args_tuple...)
        else
            AMDGPU.@sync @roc groupsize=threads gridsize=blocks func(args_tuple...)
        end
        #AMDGPU.synchronize()
    end

    # OpenMP kernel wrapper
    function call_openmp_kernel(func_name::String, threads::Any, blocks::Any, args::Any, async_flag::Bool=false)
        func_name_target = func_name * "_openmp"
        func = getfield(Main, Symbol(func_name_target))
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        func(args_tuple...)
        #AMDGPU.synchronize()
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

    function iris_println(data::String)::Cvoid
        return ccall(Libdl.dlsym(lib, :iris_println), Cvoid , (Ptr{Cchar},), pointer(data))
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

    function iris_task_dmem2dmem(task::IrisTask, src_mem::IrisMem, dst_mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_dmem2dmem), Int32, (IrisTask, IrisMem, IrisMem), task, src_mem, dst_mem)
    end

    function iris_task_dmem_d2h(task::IrisTask, mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_dmem_d2h), Int32, (IrisTask, IrisMem), task, mem)
    end

    function iris_task_dmem_h2d(task::IrisTask, mem::IrisMem)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_dmem_h2d), Int32, (IrisTask, IrisMem), task, mem)
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

    function print_element_info(vec::Vector{Any})
        for element in vec
            println("Type: ", typeof(element), " Value:", element)
            if !isa(element, Base.RefValue{IrisHRT.IrisMem})
                println("Size1: ", sizeof(element))
            else
                println("Size2: ", sizeof(element))
            end
        end
    end

    function iris_task_julia(kernel::String, dim::Int64, off::Vector{Int64}, gws::Vector{Int64}, lws::Vector{Int64}, jparams::Any)::IrisTask
        params_info = Int32[]
        params = []
        for (index,element) in enumerate(jparams)
            #println("Type: ", typeof(element), " Value:", element)
            if isa(element, Tuple)
                push!(params_info, element[2])
                if isa(element[1], Base.RefValue{IrisHRT.IrisMem})
                    push!(params, element[1])
                elseif isa(element[1], IrisHRT.IrisMem)
                    push!(params, Ref(element[1]))
                elseif isa(element[1], Vector{Any})
                    mem_X = IrisHRT.iris_data_mem(element[1])
                    push!(params, Ref(mem_X))
                    jparams[index] = (mem_X, element[2])
                    # TODO: Release this memory object 
                end
            elseif isa(element, Base.RefValue{IrisHRT.IrisMem})
                push!(params_info, Int32(IrisHRT.iris_rw))
                push!(params, element)
            elseif isa(element, IrisHRT.IrisMem)
                push!(params_info, Int32(IrisHRT.iris_rw))
                push!(params, Ref(element))
            elseif isa(element[1], Vector{Any})
                push!(params_info, Int32(IrisHRT.iris_rw))
                mem_X = IrisHRT.iris_data_mem(element[1])
                push!(params, Ref(mem_X))
                jparams[index] = (element[0], mem_X)
                # TODO: Release this memory object 
            else
                type_data = iris_unknown
                if typeof(element) == Float32
                    type_data = iris_float
                elseif typeof(element) == Float64
                    type_data = iris_double
                elseif typeof(element) == Int64
                    type_data = iris_int64
                elseif typeof(element) == Int32
                    type_data = iris_int32
                elseif typeof(element) == Int16
                    type_data = iris_int16
                elseif typeof(element) == Int8
                    type_data = iris_int8
                elseif typeof(element) == UInt64
                    type_data = iris_uint64
                elseif typeof(element) == UInt32
                    type_data = iris_uint32
                elseif typeof(element) == UInt16
                    type_data = iris_uint16
                elseif typeof(element) == UInt8
                    type_data = iris_uint8
                elseif typeof(element) == Char
                    type_data = iris_char
                end
                push!(params_info, type_data | sizeof(element))
                push!(params, Ref(element))
            end
        end
        nparams = Int64(length(params))
        #println("Params: ", params)
        #println("NParams : $nparams")
        task = iris_task_create_struct()
        ccall(Libdl.dlsym(lib, :iris_task_enable_julia_interface), Int32, (IrisTask,), task)
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

    function iris_task_native(kernel::String, dim::Int64, off::Vector{Int64}, gws::Vector{Int64}, lws::Vector{Int64}, nparams::Int64, params::Any, params_info::Vector{Int32})::IrisTask
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

    function iris_task_kernel_launch_disabled(task::IrisTask, flag::Int64)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_kernel_launch_disabled), Int32, (IrisTask, Int32), task, Int32(flag))
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

    function iris_data_mem(T, dims...) 
        #size = Csize_t(length(host) * sizeof(T))
        host_size = prod(dims) * sizeof(T)
        dim = length(dims)
        element_size = Int32(sizeof(T))
        host_cptr = C_NULL
        #println("Type of element: ", T, " Size:", size(host), " Element size:", element_size)
        element_type = iris_pointer
        if T == Float32
            element_type = iris_float 
        elseif T == Float64
            element_type = iris_double
        elseif T == Int64
            element_type = iris_int64
        elseif T == Int32
            element_type = iris_int32
        elseif T == Int16
            element_type = iris_int16
        elseif T == Int8
            element_type = iris_int8
        elseif T == UInt64
            element_type = iris_uint64
        elseif T == UInt32
            element_type = iris_uint32
        elseif T == UInt16
            element_type = iris_uint16
        elseif T == UInt8
            element_type = iris_uint8
        elseif T == Char
            element_type = iris_char
        else
            element_type = iris_unknown
        end
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_nd), IrisMem, (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Csize_t, Int32), host_cptr, host_size, dim, element_size, Int32(element_type))
    end

    function iris_data_mem(host::Array{T}) where T 
        #size = Csize_t(length(host) * sizeof(T))
        host_size = collect(size(host))
        dim = length(host_size)
        element_size = Int32(sizeof(T))
        host_cptr = reinterpret(Ptr{Cvoid}, pointer(host))
        #println("Type of element: ", T, " Size:", size(host), " Element size:", element_size)
        element_type = iris_pointer
        if T == Float32
            element_type = iris_float 
        elseif T == Float64
            element_type = iris_double
        elseif T == Int64
            element_type = iris_int64
        elseif T == Int32
            element_type = iris_int32
        elseif T == Int16
            element_type = iris_int16
        elseif T == Int8
            element_type = iris_int8
        elseif T == UInt64
            element_type = iris_uint64
        elseif T == UInt32
            element_type = iris_uint32
        elseif T == UInt16
            element_type = iris_uint16
        elseif T == UInt8
            element_type = iris_uint8
        elseif T == Char
            element_type = iris_char
        else
            element_type = iris_unknown
        end
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_nd), IrisMem, (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Csize_t, Int32), host_cptr, host_size, dim, element_size, Int32(element_type))
    end

    function iris_data_mem_create_struct(host::Ptr{Cvoid}, size::Csize_t)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct), IrisMem, (Ptr{Cvoid}, Csize_t), host, size)
    end

    function iris_data_mem_create_struct_nd(host::Ptr{Cvoid}, host_size::Ptr{Csize_t}, dim::Int32, elem_size::Csize_t, element_type::Int32)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_nd), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Int32, Csize_t, Int32), host, host_size, dim, elem_size, element_type)
    end

    function iris_data_mem_create_struct_with_type(host::Ptr{Cvoid}, size::Csize_t, element_type::Int32)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_with_type), IrisMem, (Ptr{Cvoid}, Csize_t, Int32), host, size, element_type)
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

    function iris_data_mem_create_region_struct(root_mem::IrisMem, region::Int32)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_region_struct), IrisMem, (IrisMem, Int32), root_mem, region)
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

    function iris_dev_get_ctx(device::Int)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_dev_get_ctx), Ptr{Cvoid}, (Int32, ), Int32(device))
    end

    function iris_mem_arch_ptr(mem::IrisMem, device::Int)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_mem_arch_ptr), Ptr{Cvoid}, (IrisMem, Int32), mem, Int32(device))
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


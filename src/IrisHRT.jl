#####################################################
#   Author: Narasinga Rao Miniskar
#   Date: 01/28/2025
#   File: IrisHRT.jl
#   Contact: miniskarnr@ornl.gov
#   Comment: IRIS Julia interface
#####################################################
#import Pkg
#import Pkg.add("CUDA")
#import Pkg.add("AMDGPU")
Base.Experimental.make_io_thread()
if !haskey(ENV, "IRIS_ARCHS")
    ENV["IRIS_ARCHS"] = "cuda:hip:openmp"
end
const ARRAY_TO_DMEM_MAP = Dict{Any, Any}()
const __iris_dmem_map = Dict{Any, Any}()
const __iris_cuda_devno_stream = Dict{Any, Any}()
const __iris_hip_devno_stream = Dict{Any, Any}()
const __iris_dmem_custom_type = Dict{Any, Any}()
const __iris_taskid_paramid_custom_type = Dict{Any, Any}()
module DummyAMDGPU
    macro sync(args...) end
    macro async(args...) end
    macro roc(args...) end  # Use a varargs definition instead of keyword arguments
    # Mocked function to simulate the existence of a GPU
    function has_rocm_gpu()
        return false  # Simulate no available GPU
    end
    function ROCBackend(args...) return nothing end
    function device!(dev) end
    function context!(ctx) end
    function stream!(str) end
    # Mockup for the status variable
    const status = "No ROCm GPU available"
    module HIP
    struct hipStream_t
        id::Int64
    end
    struct hipContext_t
        id::Int64
    end
    end
    struct HIPDevice
        id::Int64
    end
    struct HIPContext
        id::Int64
    end
end

module DummyCUDA
    struct CUctx_st
        id::Int
    end
    struct CuStream
        id::Int
    end
    struct CuArray
        id::Int
    end
    struct CUstream
        id::Int
    end
    struct CuContext
        id::Int
    end
    const cudaStream_t = Ptr{CUstream}
    macro sync() end
    macro async() end
    function CuContext(ctx) end
    function context!(ctx) end
    function stream!(str) end
    function device!(dev) end
    macro cuda(args...) end  # Use a varargs definition instead of keyword arguments
end
const cuda_available = try
    using CUDA
    const MyCUDA = CUDA
    true
catch e
    println(Core.stdout, "CUDA is not defined")
    const MyCUDA = DummyCUDA
    false
end

#__precompile__(false)
const hip_available = try
    using AMDGPU
    if AMDGPU.has_rocm_gpu()
        #println(Core.stdout, "AMDGPU is defined and a ROCm GPU is available!")
        const MyAMDGPU = AMDGPU
        true  # Assign true if AMDGPU is available
    else
        #const AMDGPU = DummyAMDGPU
        const MyAMDGPU = DummyAMDGPU
        println(Core.stdout, "AMDGPU is defined but no ROCm GPU available.")
        false  # Assign false if AMDGPU is defined but no GPU
    end
catch e
    const MyAMDGPU = DummyAMDGPU
    #const MyAMDGPU = AMDGPU
    println(Core.stdout, "AMDGPU is not defined!")
    false  # Assign false if AMDGPU is not available
end
mutable struct IRISCuStream
    handle::MyCUDA.CUstream
    Base.@atomic valid::Bool
    ctx::Union{Nothing,MyCUDA.CuContext}
end
mutable struct IRISHIPStream
    stream::MyAMDGPU.HIP.hipStream_t
    priority::Symbol
    device::MyAMDGPU.HIPDevice
    ctx::MyAMDGPU.HIPContext
end
module IrisHRT

    using InteractiveUtils
    using Requires
    using Base.Threads: @spawn
    #__precompile__(false)

    using Libdl
    using JACC
    cwd = pwd()
    const cuda_available = Main.cuda_available
    const MyCUDA = Main.MyCUDA
    const MyAMDGPU = Main.MyAMDGPU
    const hip_available = Main.hip_available
    const iris_kernel_jl = cwd * "/kernel.jl"
    #println("Kernel file: $iris_kernel_jl")
    #include(iris_kernel_jl)
    #using .IrisKernelImpl
    libiris = ENV["IRIS"] * "/lib/" * "libiris.so"
    if !isfile(libiris)
        libiris = "libiris.so"
    end
    # Load the shared library
    if !haskey(ENV, "IRIS_ARCHS")
        ENV["IRIS_ARCHS"] = "cuda:hip:openmp"
    end
    lib = Libdl.dlopen(libiris)
    #println(Core.stdout, "PWD: $cwd")
    #println(Core.stdout, "CUDA availability: $cuda_available")
    #println(Core.stdout, "HIP availability: $hip_available")

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

    const iris_julia_native              = 0
    const iris_julia_native_host         = (1 << 16) | 0
    const iris_core_native               = 1
    const iris_julia_kernel_abstraction  = 2
    const iris_julia_jacc                = 3
    const iris_julia_jacc_parallel_for   = (0 << 16) | 3
    const iris_julia_jacc_parallel_reduce= (1 << 16) | 3


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
    const iris_julia_policy = 1 << 29

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
    const iris_bool          = 16 << 16
    const iris_custom_type   = 17 << 16
    const iris_pointer       = 16384 << 16

    const iris_normal = 1 << 10
    const iris_reduction = 1 << 11
    const iris_sum = (1 << 12) | iris_reduction
    const iris_max = (1 << 13) | iris_reduction
    const iris_min = (1 << 14) | iris_reduction

    const iris_reset_memset       = 0
    const iris_reset_assign       = 1 
    const iris_reset_arith_seq    = 2
    const iris_reset_geom_seq     = 3


    # Define structs (adjust as necessary based on iris.h definitions)
    struct IrisTask
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    struct IrisKernel
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    struct IRISValue
        value_buffer::NTuple{8, UInt8} 
    end

    struct IrisDevice
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

    # Define the struct for IRISBackend
    struct IRISBackend
        backend_name::String
    end

    macro maybethreads(ex)
       nthreads = Threads.nthreads()
       #println(Core.stdout, "Nthreads: $nthreads") 
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
    export task_policy_handler
    export convert_args

    function convert_args(args_type::Ptr{Cint}, args::Ptr{Ptr{Cvoid}}, nparams::Cint, start_index::Cint)
        julia_args = []
        return julia_args
    end 
    function iris_print_array_size_t(data::Ptr{Csize_t}, index::Int)::Cvoid
        ldata = UInt64(unsafe_load(data + index*sizeof(Cint)))
        #iris_println("Array:$index Data:$ldata")
    end
    function iris_print_array_int(data::Ptr{Cint}, index::Int)::Cvoid
        ldata = Int(unsafe_load(data + index*sizeof(Cint)))
        #iris_println("Array:$index Data:$ldata")
    end
    function set_cuda_context!(c_ctx::Ptr{Cvoid})
        # Load the pointer to CUcontext
        cu_ctx_ptr = unsafe_load(reinterpret(Ptr{Ptr{MyCUDA.CUctx_st}}, c_ctx))

        # Convert the pointer to CuContext
        cu_ctx_conv = MyCUDA.CuContext(cu_ctx_ptr)

        # Set the CUDA context
        MyCUDA.context!(cu_ctx_conv)
        return cu_ctx_conv
    end

    function set_hip_context!(c_ctx::Ptr{Cvoid})
        # Load the pointer to CUcontext
        hip_ctx_ptr = unsafe_load(reinterpret(Ptr{Ptr{MyAMDGPU.HIP.hipContext_t}}, c_ctx))

        # Convert the pointer to CuContext
        hip_ctx_conv = MyAMDGPU.HIPContext(hip_ctx_ptr, true)

        # Set the CUDA context
        MyAMDGPU.context!(hip_ctx_conv)
        return hip_ctx_conv
    end

    function preserve_cuda_iris_stream(devno, cu_stream_ptr, cu_ctx)
        iris_stream_cuda = Main.IRISCuStream(cu_stream_ptr, true, cu_ctx)
        #println(Core.stdout, "CUDA devno: ", devno)
        cu_stream = unsafe_load(reinterpret(Ptr{MyCUDA.CuStream}, Base.unsafe_convert(Ptr{Main.IRISCuStream}, Ref(iris_stream_cuda))))
        Main.__iris_cuda_devno_stream[devno] = iris_stream_cuda
        return cu_stream
    end

    function push_dmem_custom_type(mem_id, type_info)
        Main.__iris_dmem_custom_type[mem_id] = type_info
    end

    function get_dmem_custom_type(mem)
        mem_id = mem.uid
        return Main.__iris_dmem_custom_type[mem_id] 
    end

    function push_custom_type(task_id, param_id, type_info)
        Main.__iris_dmem_custom_type[task_id * 1000000 + param_id] = type_info
        return nothing
    end

    function get_custom_type(task_id, param_id)
        return Main.__iris_dmem_custom_type[task_id*1000000 + param_id]
    end

    function preserve_hip_iris_stream(devno, hip_dev, hip_stream_ptr, hip_ctx)
        iris_stream = Main.IRISHIPStream(hip_stream_ptr, Symbol(0), hip_dev, hip_ctx)
        #println(Core.stdout, "HIP devno: ", devno)
        hip_stream = unsafe_load(reinterpret(Ptr{MyAMDGPU.HIP.HIPStream}, Base.unsafe_convert(Ptr{Main.IRISHIPStream}, Ref(iris_stream))))
        Main.__iris_hip_devno_stream[devno] = iris_stream
        #println(Core.stdout, "CUDA Stream: $cu_stream ctx: $cu_ctx")
        return hip_stream
    end

    function task_policy_handler(brs_task::IrisTask, policy_name::Ptr{Cchar}, devs::Ptr{IrisDevice}, ndevs::Int32, out_devs::Ptr{Int32})::Cint
        func_name = unsafe_string(policy_name)
        policy_fn = getfield(Main, Symbol(func_name))
        j_devs = unsafe_wrap(Array, devs, Int(ndevs), own=false)
        j_out_devs = unsafe_wrap(Array, out_devs, Int(ndevs), own=false)
        out = policy_fn(brs_task, j_devs, Int(ndevs), j_out_devs)
        #println(" JPolicy Fn: ", policy_fn, " out:", out)
        return Int32(out)
    end

    function parse_function(s::AbstractString)
        parts = split(s, ":")
        if length(parts) == 3
            include_file, mod, func = parts
            if mod == ""
                mod = nothing
            end
        elseif length(parts) == 2
            # Assume missing include-file; first part is the module
            include_file = nothing
            mod, func = parts
        elseif length(parts) == 1
            include_file = nothing
            mod = nothing
            func = parts[1]
        else
            error("Unexpected format: too many dots")
        end
        return (include_file, mod, func)
    end
    function getFunction(func_name)
        (include_file, mod, func) = parse_function(func_name)
        func_s = nothing
        if include_file != nothing
            #println(Core.stdout, "Including file:", include_file)
            include(include_file * ".jl")
            if mod != nothing
                mod_s = getfield(@__MODULE__, Symbol(mod))
                func_s = getfield(mod_s, Symbol(func))
            else
                #println(Core.stdout, "Get symbol: ", func)
                func_s = getfield(@__MODULE__, Symbol(func))
                #println(Core.stdout, "Get symbol: ", func_s)
            end
        elseif mod != nothing
            mod_s = getfield(Main, Symbol(mod))
            func_s = getfield(mod_s, Symbol(func))
        else
            func_s = getfield(Main, Symbol(func))
        end
        return func_s
    end

    function kernel_julia_wrapper(task_id::Culong, julia_kernel_type::Cint, target::Cint, devno::Cint, ctx::Ptr{Cvoid}, async_flag_int::Cint, stream_index::Cint, stream::Ptr{Ptr{Cvoid}}, nstreams::Cint, args_type::Ptr{Cint}, args::Ptr{Ptr{Cvoid}}, param_size::Ptr{Csize_t}, param_dim_size::Ptr{Csize_t}, nparams::Cint, c_blocks::Ptr{Csize_t}, c_threads::Ptr{Csize_t}, dim::Cint, kernel_name::Ptr{Cchar})::Cint
        # Array to hold the converted Julia values
        #julia_args = convert_args(args_type, args, nparams, 2)
        #iris_println("nstreams: $nstreams, steram:$stream ctx:$ctx stream_index:$stream_index")
        #println("Kernel type:", julia_kernel_type)
        b_async_flag = Bool(async_flag_int)
        julia_args = []
        julia_sizes = []
        julia_dims = []
        start_index = 2
        DMEM_MAX_DIM = 6
        cu_ctx = nothing
        hip_ctx = nothing
        #println(Core.stdout, " dim--:", dim)
        if Int(target) == Int(iris_cuda)
            MyCUDA.device!(Int(devno))
            cu_ctx = set_cuda_context!(ctx)
            MyCUDA.context!(cu_ctx)
        elseif Int(target) == Int(iris_hip)
            MyAMDGPU.device!(MyAMDGPU.devices()[devno+1])
            hip_ctx = set_hip_context!(ctx)
            MyAMDGPU.context!(hip_ctx)
            #GC.gc()
        end
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
            param_dim = Int(single_arg_type & 0xFF)
            full_size = UInt64(unsafe_load(param_size + i*sizeof(Csize_t)))
            size_dims = [] 
            for d in 0:(param_dim - 1)
                value = unsafe_load(param_dim_size + i*sizeof(Csize_t)*DMEM_MAX_DIM + d * sizeof(Csize_t)) + 0
                push!(size_dims, Int64(value))
            end
            size_dims = Tuple(size_dims)
            # Load the pointer to the actual argument
            arg_ptr = unsafe_load(Ptr{Ptr{Cvoid}}(args + i*sizeof(Ptr{Cvoid})))
            #iris_println("I:$i type:$current_type ptr:$arg_ptr is_pointer:$is_pointer element_size:$element_size dim:$dim Size:$full_size SizeDim:$size_dims")
            # Convert based on the type
            if is_pointer != Int(iris_pointer)
                #println(Core.stdout, "I is not pointer:", i, " current_type:", current_type)
                if current_type == Int(iris_int64)  # Assuming 1 is for Int
                    push!(julia_args, Int64(unsafe_load(Ptr{Clonglong}(arg_ptr))))
                    #println(Core.stdout, "I is not pointer:", i, " current_type: Int64-", current_type, " value:", julia_args[end])
                elseif current_type == Int(iris_int32)  # Assuming 1 is for Int
                    push!(julia_args, Int32(unsafe_load(Ptr{Cint}(arg_ptr))))
                    #println(Core.stdout, "I is not pointer:", i, " current_type: Int32-", current_type, " value:", julia_args[end])
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
                    println(Core.stdout, "I is not pointer:", i, " current_type:", current_type)
                    error("Unsupported type")
                end
            else # Assuming 3 is for Char
                #println(Core.stdout, "I is a pointer:", i)
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
                elseif current_type == Int(iris_bool)
                    arg_ptr = Ptr{Cuchar}(arg_ptr)
                elseif current_type == Int(iris_custom_type)
                    cust_type = get_custom_type(task_id, i-start_index+1)
                    arg_ptr = Ptr{cust_type}(arg_ptr)
                else
                    println(Core.stdout, "I is not pointer index:", i-start_index+1, " current_type:", current_type)
                    iris_println("[Julia-IrisHRT-Error] No type present")
                end
                try 
                    #jptr = unsafe_wrap(CUDA.Array, arg_ptr, size_dims, own=false)
                    #jptr = unsafe_wrap(CuDeviceArray{Float32, 1}, arg_ptr, size_dims, own=false)
                    #iris_println("After Pointer $arg_ptr $jptr")
                    #push!(julia_args, jptr)
                    if Int(target) == Int(iris_cuda)
                        arg_ptr = cuda_reinterpret(task_id, i-start_index+1, arg_ptr, current_type)
                        #println(Core.stdout, "Index: $i target arg:$arg_ptr size:$size_dims")
                        arg_ptr = unsafe_wrap(MyCUDA.CuArray, arg_ptr, size_dims, own=false)
                        #iris_println("After Pointer $arg_ptr")
                    elseif Int(target) == Int(iris_hip)
                        arg_ptr = ptr_reinterpret(task_id, i-start_index+1, 0, arg_ptr, current_type)
                        #iris_println("Index: $i target arg:$arg_ptr size:$size_dims")
                        arg_ptr = unsafe_wrap(MyAMDGPU.ROCArray, arg_ptr, size_dims, lock=false)
                        #iris_println("After Pointer $arg_ptr")
                    elseif Int(target) == Int(iris_openmp)
                        arg_ptr = ptr_reinterpret(task_id, i-start_index+1, 0, arg_ptr, current_type)
                        #iris_println("Index: $i target arg:$arg_ptr size:$size_dims")
                        arg_ptr = unsafe_wrap(Array, arg_ptr, size_dims, own=false)
                        #iris_println("After Pointer $arg_ptr")
                    else
                        iris_println("[Julia-IrisHRT-Error] Unknown target to handle arguments")
                    end
                    push!(julia_args, arg_ptr)
                catch e 
                    iris_println("[Julia-IrisHRT-Error] exception raised during push of arguments to julia_args")
                    rethrow(e)
                end
            end
        end
        #iris_println("Hello:$target Devno:$devno nparams:$nparams" * vec_string(julia_args))
        #iris_println("Now calling CUDA")
        j_threads = []
        j_blocks = []
        # Iterate through each argument
        #println(Core.stdout, " dim:", dim)
        for i in 0:(dim - 1)
            # Load the type of the current argument
            push!(j_threads, UInt64(unsafe_load(c_threads+i*sizeof(Csize_t)))+0)
            push!(j_blocks, UInt64(unsafe_load(c_blocks+i*sizeof(Csize_t)))+0)
        end
        j_threads = Tuple(j_threads)
        j_blocks = Tuple(j_blocks)

        #println(Core.stdout, " j_threads: ", j_threads)
        #println(Core.stdout, " j_blocks: ", j_blocks)
        all_streams = nothing
        if Int(target) == Int(iris_cuda)
            #cu_ctx = unsafe_load(Ref{CuContext}(ctx))
            #GC.gc()
            func_name = unsafe_string(kernel_name)
            cu_stream = nothing
            if !b_async_flag
                cu_stream = nothing
                #iris_println("Ctx: $cu_ctx dev:$devno")
            else
                #println(Core.stdout, "CUDA stream: ", stream, " type: ", typeof(stream))
                cu_stream_ptr = unsafe_load(reinterpret(Ptr{MyCUDA.CUstream}, stream))
                cus_ptr = reinterpret(Ptr{MyCUDA.cudaStream_t}, stream)
                all_streams = unsafe_wrap(Vector{MyCUDA.cudaStream_t}, cus_ptr, nstreams)
                cu_stream = preserve_cuda_iris_stream(devno, cu_stream_ptr, cu_ctx)
                #iris_stream = Main.IRISCuStream(cu_stream_ptr, true, cu_ctx)
                #cu_stream = unsafe_load(reinterpret(Ptr{MyCUDA.CuStream}, Base.unsafe_convert(Ptr{Main.IRISCuStream}, Ref(iris_stream))))
                # Explicitly keep a reference
                #cu_ctx = unsafe_load(ctx)  # CUcontext
                #iris_println("Stream: $cu_stream dev:$devno")
                #iris_println("Ctx: $cu_ctx dev:$devno")
                #println(Core.stdout, "IRIS Stream: $iris_stream")
                #println(Core.stdout, "CU Stream: $cu_stream ctx:$cu_ctx")
                #println("Size of CuContext type: ", sizeof(CuContext), " bytes")
                #println("Size of CuStream type: ", sizeof(MyCUDA.CuStream), " bytes")
                #println("Size of CUstream type: ", sizeof(Main.CUDA.CUstream), " bytes")
                #println("Size of IRISCuStream type: ", sizeof(Main.IRISCuStream), " bytes")
            end
            #iris_println("Calling CUDA kernel")
            if julia_kernel_type == iris_julia_kernel_abstraction 
                call_cuda_kernel_ka(julia_kernel_type, devno, func_name, j_threads, j_blocks, cu_ctx, cu_stream, julia_args, b_async_flag)
            elseif (julia_kernel_type & 0xFFFF) == iris_julia_jacc
                call_cuda_kernel_jacc(julia_kernel_type, devno, func_name, j_threads, j_blocks, cu_ctx, cu_stream, julia_args, b_async_flag)
            else
                call_cuda_kernel(julia_kernel_type, devno, func_name, j_threads, j_blocks, cu_ctx, cu_stream, all_streams, julia_args, b_async_flag)
            end
            #CUDA.@sync @cuda threads=j_threads blocks=j_blocks Main.saxpy_cuda(julia_args[1], julia_args[2], julia_args[3], julia_args[4])
            ##############################################################
            #iris_println("Completed CUDA")
            return target*11
        elseif Int(target) == Int(iris_hip)
            #GC.gc()
            hip_ctx = unsafe_load(reinterpret(Ptr{MyAMDGPU.HIPContext}, ctx))
            hip_stream = nothing
            all_streams = nothing
            if b_async_flag 
                hip_dev = MyAMDGPU.device!(MyAMDGPU.devices()[devno+1])
                #hip_stream = unsafe_load(reinterpret(Ptr{HIPStream}, stream))
                hip_stream_ptr = unsafe_load(reinterpret(Ptr{MyAMDGPU.HIP.HIP.hipStream_t}, stream))
                hip_ptr = reinterpret(Ptr{MyAMDGPU.HIP.hipStream_t}, stream)
                all_streams = unsafe_wrap(Vector{MyAMDGPU.HIP.hipStream_t}, hip_ptr, nstreams)
                #println(Core.stdout, "HIP stream ptr", hip_stream_ptr, " type:", typeof(hip_stream_ptr))
                #println(Core.stdout, "HIP dev", hip_dev, " type:", typeof(hip_dev))
                #println(Core.stdout, "HIP ctx", hip_ctx, " type:", typeof(hip_ctx))
                hip_stream = preserve_hip_iris_stream(devno, hip_dev, hip_stream_ptr, hip_ctx)
                #println(Core.stdout, "HIP stream", hip_stream, " type:", typeof(hip_stream))
            end
            #iris_println("----Ctx: $hip_ctx dev:$devno")
            func_name = unsafe_string(kernel_name)
            #Main.AMDGPU.device!(Main.AMDGPU.devices()[devno+1])
            #Main.AMDGPU.context!(hip_ctx)
            #iris_println("Calling HIP kernel")
            if julia_kernel_type == iris_julia_kernel_abstraction 
                call_hip_kernel_ka(julia_kernel_type, devno, func_name, j_threads, j_blocks, hip_ctx, hip_stream, julia_args, b_async_flag)
            elseif (julia_kernel_type & 0xFFFF) == iris_julia_jacc
                call_hip_kernel_jacc(julia_kernel_type, devno, func_name, j_threads, j_blocks, hip_ctx, hip_stream, julia_args, b_async_flag)
            else
                call_hip_kernel(julia_kernel_type, devno, func_name, j_threads, j_blocks, hip_ctx, hip_stream, all_streams, julia_args, b_async_flag)
            end
            ##############################################################
            #iris_println("Completed HIP")
            return target*12
        elseif Int(target) == Int(iris_openmp)
            #cu_ctx = unsafe_load(Ref{CuContext}(ctx))
            #GC.gc()
            #iris_println("dev:$devno")
            func_name = unsafe_string(kernel_name)
            #iris_println("Calling OpenMP kernel")
            if julia_kernel_type == iris_julia_kernel_abstraction 
                call_openmp_kernel_ka(julia_kernel_type, func_name, j_threads, j_blocks, julia_args, b_async_flag)
            elseif (julia_kernel_type & 0xFFFF) == iris_julia_jacc
                call_openmp_kernel_jacc(julia_kernel_type, func_name, j_threads, j_blocks, julia_args, b_async_flag)
            else
                call_openmp_kernel(julia_kernel_type, devno, func_name, j_threads, j_blocks, julia_args, b_async_flag)
            end
            ##############################################################
            #iris_println("Completed OpenMP")
            return target*13
        else
            return target*10
        end
    end

    function cuda_reinterpret(task_id, param_id, arg_ptr::Any, current_type::Any)
        if current_type == Int(iris_float)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Float32}, arg_ptr)
        elseif current_type == Int(iris_double)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Float64}, arg_ptr)
        elseif current_type == Int(iris_int64)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Int64}, arg_ptr)
        elseif current_type == Int(iris_int32)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Int32}, arg_ptr)
        elseif current_type == Int(iris_int16)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Int16}, arg_ptr)
        elseif current_type == Int(iris_int8)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Int8}, arg_ptr)
        elseif current_type == Int(iris_char)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Char}, arg_ptr)
        elseif current_type == Int(iris_uint64)
            arg_ptr = reinterpret(MyCUDA.CuPtr{UInt64}, arg_ptr)
        elseif current_type == Int(iris_uint32)
            arg_ptr = reinterpret(MyCUDA.CuPtr{UInt32}, arg_ptr)
        elseif current_type == Int(iris_uint16)
            arg_ptr = reinterpret(MyCUDA.CuPtr{UInt16}, arg_ptr)
        elseif current_type == Int(iris_uint8)
            arg_ptr = reinterpret(MyCUDA.CuPtr{UInt8}, arg_ptr)
        elseif current_type == Int(iris_bool)
            arg_ptr = reinterpret(MyCUDA.CuPtr{Bool}, arg_ptr)
        elseif current_type == Int(iris_custom_type)
            data_type = get_custom_type(task_id, param_id)
            #println(Core.stdout, "dtype2: ", data_type)
            arg_ptr = reinterpret(MyCUDA.CuPtr{data_type}, arg_ptr)
        else
            iris_println("[Julia-IrisHRT-Error][cuda_reinterpret] Unknown type")
        end
        return arg_ptr
    end

    function ptr_reinterpret(task_id, param_id, mem_custom, arg_ptr::Any, current_type::Any)
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
        elseif current_type == Int(iris_bool)
            arg_ptr = reinterpret(Ptr{Bool}, arg_ptr)
        elseif current_type == Int(iris_custom_type) && mem_custom == 0
            data_type = get_custom_type(task_id, param_id)
            #println(Core.stdout, "dtype: ", data_type)
            arg_ptr = reinterpret(Ptr{data_type}, arg_ptr)
        elseif current_type == Int(iris_custom_type) && mem_custom == 1
            data_type = Main.__iris_dmem_custom_type[task_id] 
            #println(Core.stdout, "dtype1: ", data_type)
            arg_ptr = reinterpret(Ptr{data_type}, arg_ptr)
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
        policy_ptr = @cfunction(task_policy_handler, Cint, (IrisTask, Ptr{Cchar}, Ptr{IrisDevice}, Int32, Ptr{Int32}))
        global stored_policy_ptr = policy_ptr
        func_ptr = @cfunction(kernel_julia_wrapper, Cint, (Culong, Cint, Cint,        Cint,        Ptr{Cvoid}, Cint, Cint,               Ptr{Ptr{Cvoid}},         Cint,           Ptr{Cint},       Ptr{Ptr{Cvoid}},         Ptr{Csize_t},  Ptr{Csize_t}, Cint,          Ptr{Csize_t},          Ptr{Csize_t},         Cint,      Ptr{Cchar}))
        global stored_func_ptr = func_ptr
        use_julia_threads = false
        if use_julia_threads
            flag = ccall(Libdl.dlsym(lib, :iris_julia_init), Int32, (Ptr{Cvoid}, Int32), func_ptr, Int32(1))
            flag = flag & ccall(Libdl.dlsym(lib, :iris_init), Int32, (Ref{Int32}, Ref{Ptr{Ptr{Cchar}}}, Int32), Int32(1), C_NULL, Int32(sync))
            flag = flag & ccall(Libdl.dlsym(lib, :iris_julia_policy_init), Int32, (Ptr{Cvoid},), policy_ptr)
            ndevs = iris_ndevices()
            t = @spawn begin 
                iris_init_scheduler(0)
            end
            println(Core.stdout, "Julia: N devices $ndevs")
            for i in 0:(ndevs- 1)
                #println(Core.stdout, "Julia: Initializing Worker $i")
                flag = flag & iris_init_worker(i)
            end
            t = nothing
            for i in 0:(ndevs- 1)
                #println(Core.stdout, "Julia: Starting Worker $i")
                t = @spawn begin
                    iris_start_worker(i, 0)
                end
                #println(Core.stdout, t)
            end
            for i in 0:(ndevs- 1)
                #println(Core.stdout, "Julia: Initializing device $i")
                flag = flag & iris_init_device(i)
            end
            #println(Core.stdout, "Julia: Doing device synchronize")
            #println(Core.stdout, t)
            flush(stdout)
            flag = flag & iris_init_devices_synchronize(1)
            #println(Core.stdout, "Julia: Completed device synchronize")
        else
            flag = ccall(Libdl.dlsym(lib, :iris_julia_init), Int32, (Ptr{Cvoid}, Int32), func_ptr, Int32(0))
            flag = flag & ccall(Libdl.dlsym(lib, :iris_init), Int32, (Ref{Int32}, Ref{Ptr{Ptr{Cchar}}}, Int32), Int32(1), C_NULL, Int32(sync))
            flag = flag & ccall(Libdl.dlsym(lib, :iris_julia_policy_init), Int32, (Ptr{Cvoid},), policy_ptr)
        end
        ndevs = iris_ndevices()
        for i in 1:ndevs
            Main.__iris_cuda_devno_stream[i] = nothing
            Main.__iris_hip_devno_stream[i] = nothing
        end
        return flag
    end

    function init(sync; gc_enter=true)::Int32
        status = iris_init(sync)
        if gc_enter
            global gc_state = gc_safe_enter()
        end
        return status
    end

    function init(; gc_enter=true)::Int32
        status = iris_init(1)
        if gc_enter
            global gc_state = gc_safe_enter()
        end
        return status
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
    # CUDA kernel wrapper (KernelAbstraction kernels)
    function call_cuda_kernel_jacc(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, ctx::Any, stream::Any, args::Any, async_flag::Bool=false)
        if !Main.cuda_available
            error("CUDA device not available.")
        end

        func_name_target = func_name
        # Initialize CUDA
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        gws = map(*, threads, blocks)
        if length(gws)==1
            gws = gws[1]
        end
        if async_flag
            MyCUDA.device!(Int(devno))
            MyCUDA.context!(ctx)
            MyCUDA.stream!(stream)
            backend = MyCUDA.CUDAKernels.CUDABackend(false, false)
            #println(Core.stdout, "Calling JACC async:")
            JACC.parallel_for(backend, gws, func, args_tuple...)
        else
            backend = MyCUDA.CUDAKernels.CUDABackend(false, false)
            #println(Core.stdout, "Calling JACC:", gws, " func:", func)
            JACC.parallel_for(backend, gws, func, args_tuple...)
        end
        #synchronize(blocking = true)
    end
    using KernelAbstractions
    function call_cuda_kernel_ka(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, ctx::Any, stream::Any, args::Any, async_flag::Bool=false)
        if !Main.cuda_available
            error("CUDA device not available.")
        end

        #iris_println("Func $func_name")
        func_name_target = func_name
        # Initialize CUDA
        #CUDA.allowscalar(false)  # Disable scalar operations on the GPU
        func = getFunction(func_name_target)
        #func = getfield(Main, Symbol(func_name_target))
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #println(Core.stdout, "Args_tuple: $args_tuple")
        #println(Core.stdout, "Async $async_flag")
        gws = map(*, threads, blocks)
        if async_flag
            #iris_println("---------ASynchronous CUDA:$devno execution $blocks $threads func:$func stream:$stream ctx:$ctx----------")
            MyCUDA.device!(Int(devno))
            MyCUDA.context!(ctx)
            MyCUDA.stream!(stream)
            backend = MyCUDA.CUDAKernels.CUDABackend(false, false)
            #println(Core.stdout, "func:$func_name")
            #new_func = assign_cuda_ka_kernel(Symbol(func_name), func)
            #println(Core.stdout, "cuda_func: $new_func")
            #new_func(backend)(args_tuple...; ndrange=gws, workgroupsize=threads)
            func(backend)(args_tuple...; ndrange=gws, workgroupsize=threads)
        else
            #iris_println("---------Synchronous CUDA execution $blocks $threads func:$func----------")
            backend = MyCUDA.CUDAKernels.CUDABackend(false, false)
            func(backend)(args_tuple...; ndrange=gws, workgroupsize=threads)
            KernelAbstractions.synchronize(backend)
        end
        #synchronize(blocking = true)
    end
    
    function assign_cuda_ka_kernel(base_name::Symbol, kernel_function)
        new_name = Symbol(base_name, "____cuda")  
        @eval const $(new_name) = $kernel_function
        return getfield(@__MODULE__, new_name) 
    end

    function assign_hip_ka_kernel(base_name::Symbol, kernel_function)
        new_name = Symbol(base_name, "____hip")  
        @eval const $(new_name) = $kernel_function
        return getfield(@__MODULE__, new_name) 
    end

    function create_and_return_kernel(base_name::Symbol, kernel_function)
        new_name = Symbol(base_name, "_obj")  
        @eval const $(new_name) = $kernel_function
        return getfield(Main, new_name)  
    end


    function call_cuda_kernel(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, ctx::Any, stream::Any, all_streams::Any, args::Any, async_flag::Bool=false)
        if !Main.cuda_available
            error("CUDA device not available.")
        end

        #iris_println("Func $func_name")
        func_name_target = func_name * "_cuda"
        # Initialize CUDA
        #CUDA.allowscalar(false)  # Disable scalar operations on the GPU
        func = getFunction(func_name_target)
        #func = getfield(Main, Symbol(func_name_target))
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #println(Core.stdout, "Args_tuple: $args_tuple")
        #println(Core.stdout, "Async $async_flag")
        if async_flag
            #iris_println("---------ASynchronous CUDA execution $blocks $threads func:$func stream:$stream----------")
            #Main.CUDA.@async begin
                #Main.CUDA.context!(ctx)
                if (julia_kernel_type >> 16) == 0
                    MyCUDA.@cuda threads=threads blocks=blocks stream=stream func(args_tuple...)
                else
                    invokelatest(func, devno, all_streams, args_tuple...)
                end
            #end
            # Ensure all tasks and CUDA operations complete
        else
            #println(Core.stdout, "---------Synchronous CUDA execution blocks=$blocks threads=$threads func:$func----------")
            if (julia_kernel_type >> 16) == 0
            MyCUDA.@sync begin
                MyCUDA.context!(ctx)
                MyCUDA.@cuda threads=threads blocks=blocks func(args_tuple...)
            end
            else
                invokelatest(func, devno, all_streams, args_tuple...)
            end

            #CUDA.@sync @cuda threads=threads blocks=blocks func(args_tuple...)
            #func1(threads, blocks, args_tuple)
        end
        #synchronize(blocking = true)
    end

    function call_hip_kernel_jacc(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, ctx::Any, stream::Any, args::Any, async_flag::Bool=false)
        #iris_println("Func $func_name")
        # Initialize CUDA
        #func = getfield(Main, Symbol(func_name))
        func = getFunction(func_name)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        gws = map(*, threads, blocks)
        if length(gws)==1
            gws = gws[1]
        end
        if async_flag
                #iris_println("---------ASynchronous HIP:$devno execution $blocks $threads func:$func stream:$stream ctx:$ctx gws:$gws----------")
                MyAMDGPU.context!(ctx)
                MyAMDGPU.stream!(stream)
                backend = MyAMDGPU.ROCBackend()
                JACC.parallel_for(backend, gws, func, args_tuple...)
        else
            #println(Core.stdout, "---------Synchronous HIP execution $blocks $threads $gws func:$func----------", ctx)
            #MyAMDGPU.context!(ctx)
            backend = MyAMDGPU.ROCBackend()
            JACC.parallel_for(backend, gws, func, args_tuple...)
        end
        #AMDGPU.synchronize()
    end
    function call_hip_kernel_ka(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, ctx::Any, stream::Any, args::Any, async_flag::Bool=false)
        if !Main.hip_available
            error("HIP device not available.")
        end

        #iris_println("Func $func_name")
        func_name_target = func_name 
        # Initialize CUDA
        #AMDGPU.allowscalar(false)  # Disable scalar operations on the GPU
        #func = getfield(IrisKernelImpl, Symbol(func_name_target))
        #func = getfield(Main, Symbol(func_name_target))
        func = getFunction(func_name_target)
        #func = getfield(Main, Symbol(func_name_target, "_ka_hip"))
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #@hip threads=threads blocks=blocks add_kernel(a,b,c,N)
        #println(Core.stdout, "Args_tuple: $args_tuple")
        #println(Core.stdout, "Threads: $threads blocks:$blocks")
        #iris_println("Async $async_flag")
        gws = map(*, threads, blocks)
        if async_flag
                #iris_println("---------ASynchronous HIP:$devno execution $blocks $threads func:$func stream:$stream ctx:$ctx----------")
                MyAMDGPU.context!(ctx)
                MyAMDGPU.stream!(stream)
                backend = MyAMDGPU.ROCBackend()
                #new_func = assign_hip_ka_kernel(Symbol(func_name), func)
                #new_func(backend)(args_tuple...; ndrange=gws, workgroupsize=threads)
                func(backend)(args_tuple...; ndrange=gws, workgroupsize=threads)
        else
            #iris_println("---------Synchronous HIP execution $blocks $threads func:$func----------")
            backend = MyAMDGPU.ROCBackend()
            func(backend)(args_tuple...; ndrange=gws, workgroupsize=threads)
            KernelAbstractions.synchronize(backend)
        end
        #AMDGPU.synchronize()
    end

    # HIP kernel wrapper
    function call_hip_kernel(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, ctx::Any, stream::Any, all_streams::Any, args::Any, async_flag::Bool=false)
        if !Main.hip_available
            error("HIP device not available.")
        end

        #iris_println("Func $func_name")
        func_name_target = func_name * "_hip"
        # Initialize CUDA
        #AMDGPU.allowscalar(false)  # Disable scalar operations on the GPU
        #func = getfield(IrisKernelImpl, Symbol(func_name_target))
        #func = getfield(Main, Symbol(func_name_target))
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #@hip threads=threads blocks=blocks add_kernel(a,b,c,N)
        #println(Core.stdout, "Args_tuple: $args_tuple")
        #println(Core.stdout, "Threads: $threads blocks:$blocks")
        #iris_println("Async $async_flag")
        if async_flag
            #Main.AMDGPU.@async begin
                if (julia_kernel_type >> 16) == 0
                    MyAMDGPU.@roc groupsize=threads gridsize=blocks stream=stream func(args_tuple...)
                else
                    invokelatest(func, devno, all_streams, args_tuple...)
                end
            #end
        else
            #iris_println("---------Synchronous HIP execution $blocks $threads func:$func----------")
            if (julia_kernel_type >> 16) == 0
            MyAMDGPU.@sync begin
                MyAMDGPU.context!(ctx)
                MyAMDGPU.@roc groupsize=threads gridsize=blocks func(args_tuple...)
            end
            else
                invokelatest(func, devno, all_streams, args_tuple...)
            end
        end
        #AMDGPU.synchronize()
    end

    # OpenMP kernel wrapper
    function call_openmp_kernel(julia_kernel_type::Any, devno::Any, func_name::String, threads::Any, blocks::Any, args::Any, async_flag::Bool=false)
        func_name_target = func_name * "_openmp"
        #func = getfield(IrisKernelImpl, Symbol(func_name_target))
        #func = getfield(Main, Symbol(func_name_target))
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        if (julia_kernel_type >> 16) == 0
            func(args_tuple...)
        else
            invokelatest(func, devno, args_tuple...)
        end
    end

    function call_openmp_kernel_jacc(julia_kernel_type::Any, func_name::String, threads::Any, blocks::Any, args::Any, async_flag::Bool=false)
        func_name_target = func_name
        #func = getfield(IrisKernelImpl, Symbol(func_name_target))
        #func = getfield(Main, Symbol(func_name_target))
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        gws = threads
        if length(gws)==1
            gws = gws[1]
        end
        backend = JACC.ThreadsBackend()
        nthreads = Threads.nthreads()
        num_cpus = Sys.CPU_THREADS
        JACC.parallel_for(backend, gws, func, args_tuple...)
    end
    function call_openmp_kernel_ka(julia_kernel_type::Any, func_name::String, threads::Any, blocks::Any, args::Any, async_flag::Bool=false)
        func_name_target = func_name
        #func = getfield(IrisKernelImpl, Symbol(func_name_target))
        #func = getfield(Main, Symbol(func_name_target))
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        gws = threads
        backend = CPU()
        func(backend)(args_tuple...; ndrange=gws)
        KernelAbstractions.synchronize(backend)
        #AMDGPU.synchronize()
    end

    function iris_vendor_kernel_launch(dev::Any, kernel::Ptr{Cvoid}, gridx::Any, gridy::Any, gridz::Any, blockx::Any, blocky::Any, blockz::Any, shared_mem_bytes::Any, stream::Ptr{Cvoid}, params::Ptr{Ptr{Cvoid}})::Int32
        return ccall(Libdl.dlsym(lib, :iris_vendor_kernel_launch), Int32, (Int32, Ptr{Cvoid}, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), Int32(dev), kernel, Int32(gridx), Int32(gridy), Int32(gridz), Int32(blockx), Int32(blocky), Int32(blockz), Int32(shared_mem_bytes), stream, params)
    end


    function iris_error_count()::Int32
        return ccall(Libdl.dlsym(lib, :iris_error_count), Int32, ())
    end

    function iris_finalize()::Int32
        return ccall(Libdl.dlsym(lib, :iris_finalize), Int32, ())
    end

    function finalize(;gc_leave=true)::Int32
        if gc_leave
            IrisHRT.gc_leave(gc_state)
        end
        return iris_finalize()
    end

    function synchronize()::Int32
        return ccall(Libdl.dlsym(lib, :iris_synchronize), Int32, ())
    end

    function iris_synchronize()::Int32
        return ccall(Libdl.dlsym(lib, :iris_synchronize), Int32, ())
    end

    function iris_task_retain(task::IrisTask, flag::Int32)::Cvoid
        ccall(Libdl.dlsym(lib, :iris_task_retain), Cvoid, (IrisTask, Int32), task, flag)
    end

    function iris_task_set_julia_policy(task::IrisTask, name::String)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_julia_policy), Int32, (IrisTask, Ptr{Cchar}), task, pointer(name))
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

    function ndevs()::Int
        return Int(iris_ndevices())
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

    function empty_task()
        return iris_task_create_struct()
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

    function iris_task_depend(task::IrisTask, tasks::Vector{IrisTask})::Int32
        return iris_task_depend(task, Int32(length(tasks)), pointer(tasks)) 
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

    function iris_task_get_metadata_all(task::IrisTask)::Ptr{Int32}
        return ccall(Libdl.dlsym(lib, :iris_task_get_metadata_all), Ptr{Int32}, (IrisTask, ), task)
    end

    function iris_task_set_metadata_all(task::IrisTask, metadata::Ptr{Int32}, n)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_metadata_all), Int32, (IrisTask, Ptr{Int32}, Int32), task, metadata, Int32(n))
    end

    function iris_task_get_metadata(task::IrisTask, index::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_get_metadata), Int32, (IrisTask, Int32), task, Int32(index))
    end

    function iris_task_set_metadata(task::IrisTask, index::Int, metadata::Int)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_metadata), Int32, (IrisTask, Int32, Int32), task, Int32(index), Int32(metadata))
    end

    function iris_task_get_metadata_count(task::IrisTask)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_get_metadata_count), Int32, (IrisTask, ), task)
    end

    function metadata(task::IrisTask, index=nothing)
        if index != nothing
            return iris_task_get_metadata(task, index)
        end
        n = iris_task_get_metadata_count(task)
        out = iris_task_get_metadata_all(task)
        j_out = unsafe_wrap(Array, out, Int(n), own=false)
        j_out64 = map(Int64, j_out)
        return j_out64
    end

    function set_metadata(task::IrisTask, metadata, index=nothing)
        if index != nothing
            return iris_task_set_metadata(task, index, metadata)
        end
        if isa(metadata, Tuple)
            metadata = collect(metadata)
            metadata32 = map(Int32, metadata)
            return iris_task_set_metadata_all(task, pointer(metadata32), Int32(length(metadata)))
        elseif isa(metadata, Array)
            metadata32 = map(Int32, metadata)
            return iris_task_set_metadata_all(task, pointer(metadata32), Int32(length(metadata)))
        elseif isa(metadata, Number)
            return iris_task_set_metadata(task, 0, metadata)
        else
            error("Unknown metadata type for task: ", task, " metadata:", metadata, " metadata type: ", typeof(metadata))
            return IrisHRT.IRIS_ERROR
        end
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

    function flush(task::IrisTask, mem::IrisMem)::Int32
        return iris_task_dmem_flush_out(task, mem)
    end

    function flush(task::IrisTask, host::Array{T})::Int32 where T
        p_host = pointer(host)
        mem = Main.__iris_dmem_map[p_host]
        return iris_task_dmem_flush_out(task, mem)
    end

    function flush(mem; submit=true, wait=true, dependencies=[], policy=IrisHRT.iris_roundrobin)::IrisTask
        task0 = iris_task_create_struct()
        if isa(mem, IrisMem)
            IrisHRT.iris_task_dmem_flush_out(task0, mem)
        else
            p_mem = pointer(mem)
            IrisHRT.iris_task_dmem_flush_out(task0, Main.__iris_dmem_map[p_mem])
        end
        dependencies = [x for x in dependencies if x !== nothing]
        if length(dependencies) > 0
            iris_task_depend(task0, dependencies)
        end
        if submit
            IrisHRT.iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        else
            IrisHRT.iris_task_set_policy(task0, Int32(policy))
        end
        return task0
    end
    function release(array_obj::Array{T}) where T 
        dmem_finalizer(array_obj)
    end

    function dmem_finalizer(array_obj)
        p_array = pointer(array_obj)
        if haskey(Main.__iris_dmem_map, p_array)
            println(Core.stdout, ">>>>>>> releasing memory: $p_array <<<<<<<<<<<")
            IrisHRT.iris_mem_release(Main.__iris_dmem_map[p_array])
            delete!(Main.__iris_dmem_map, p_array)       
        end
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
                println("Size: ", sizeof(element))
            else
                println("Size: ", sizeof(element))
            end
        end
    end

    function iris_task_julia(kernel::String, dim::Int64, off::Any, gws::Any, lws::Any, jparams::Any)::IrisTask
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
                type_data = get_iris_type(typeof(element))
                push!(params_info, type_data | sizeof(element))
                push!(params, Ref(element))
            end
        end
        nparams = Int64(length(params))
        #println("Params: ", params)
        #println("NParams : $nparams")
        task = iris_task_create_struct()
        ccall(Libdl.dlsym(lib, :iris_task_enable_julia_interface), Int32, (IrisTask, Int32), task, Int32(IrisHRT.iris_julia_native))
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

    function iris_task_submit(task::IrisTask, device_policy::Int64, opt::Ptr{Int8}, sync::Int64)::Int32
        #gc_state = @ccall(jl_gc_safe_enter()::Int8)
        status = ccall(Libdl.dlsym(lib, :iris_task_submit), Int32, (IrisTask, Int32, Ptr{Int8}, Int32), task, Int32(device_policy), opt, Int32(sync))
        #@ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
        return status
    end

    function submit(task::IrisTask, device_policy::Int64, opt::Ptr{Int8}, sync::Int64)::Int32
        return iris_task_submit(task, device_policy, opt, sync)
    end

    function submit(task::IrisTask, device_policy::Int64, sync::Int64)::Int32
        return iris_task_submit(task, device_policy, Ptr{Int8}(C_NULL), sync)
    end

    function submit(task::IrisTask, device_policy::Int64)::Int32
        return iris_task_submit(task, device_policy, Ptr{Int8}(C_NULL), 1)
    end

    function submit(task::IrisTask)::Int32
        return iris_task_submit(task, iris_default, Ptr{Int8}(C_NULL), 1)
    end

    function iris_task_set_policy(task::IrisTask, policy::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_task_set_policy), Int32, (IrisTask, Int32), task, policy)
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

    function wait(task::IrisTask)::Int32
        return iris_task_wait(task)
    end

    function wait(tasks::Vector{IrisTask})::Int32
        return iris_task_wait_all(Int32(length(tasks)), pointer(tasks))
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

    function release(task::IrisTask)::Int32
        return iris_task_release(task)
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

    function iris_mem_init_reset(mem::IrisMem, reset::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_mem_init_reset), Int32, (IrisMem, Int32), mem, reset)
    end

    const VALUE_SIZE = 8
    function flat_in_buffer(value::Any)::IRISValue
        value_buffer = Base.zeros(UInt8, VALUE_SIZE)
        if typeof(value) == Float32
            value_buffer[1:4] .= reinterpret(UInt8, [value])
        elseif typeof(value) == Float64
            value_buffer[1:8] .= reinterpret(UInt8, [value])
        elseif typeof(value) == Int64
            value_buffer[1:8] .= reinterpret(UInt8, [value])
        elseif typeof(value) == Int32
            value_buffer[1:4] .= reinterpret(UInt8, [value])
        elseif typeof(value) == Int16
            value_buffer[1:2] .= reinterpret(UInt8, [value])
        elseif typeof(value) == Int8
            value_buffer[1:1] .= reinterpret(UInt8, [value])
        elseif typeof(value) == UInt64
            value_buffer[1:8] .= reinterpret(UInt8, [value])
        elseif typeof(value) == UInt32
            value_buffer[1:4] .= reinterpret(UInt8, [value])
        elseif typeof(value) == UInt16
            value_buffer[1:2] .= reinterpret(UInt8, [value])
        elseif typeof(value) == UInt8
            value_buffer[1:1] .= reinterpret(UInt8, [value])
        elseif typeof(value) == Bool 
            value_buffer[1:1] .= reinterpret(UInt8, [value])
        else
            println(Core.stdout, "Unknown type of $value")
        end
        ivalue = IRISValue(Tuple(value_buffer))
        #println("Element: ", value, " buffer:", ivalue)
        return ivalue 
    end

    function iris_mem_init_reset_assign(mem::IrisMem, element::Any)::Int32
        ivalue = flat_in_buffer(element)
        return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_assign), Int32, (IrisMem, IRISValue), mem, ivalue)
    end

    function iris_mem_init_reset_random_uniform_seq(mem::IrisMem, seed::Int, min::Any, max::Any)::Int32
        imin = flat_in_buffer(min)
        imax = flat_in_buffer(max)
        #println("Element: ", element, " buffer:", ivalue)
        return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_random_uniform_seq), Int32, (IrisMem, Clonglong, IRISValue, IRISValue), mem, seed, imin, imax)
    end

    function iris_mem_init_reset_arith_seq(mem::IrisMem, element::Any, step::Any)::Int32
        ivalue = flat_in_buffer(element)
        istep = flat_in_buffer(step)
        #println("Element: ", element, " buffer:", ivalue)
        return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_arith_seq), Int32, (IrisMem, IRISValue, IRISValue), mem, ivalue, istep)
    end

    function iris_mem_init_reset_geom_seq(mem::IrisMem, element::Any, step::Any)::Int32
        ivalue = flat_in_buffer(element)
        istep = flat_in_buffer(step)
        #println("Element: ", element, " buffer:", ivalue)
        return ccall(Libdl.dlsym(lib, :iris_mem_init_reset_geom_seq), Int32, (IrisMem, IRISValue, IRISValue), mem, ivalue, istep)
    end

    function iris_data_mem_init_reset(mem::IrisMem, reset::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_init_reset), Int32, (IrisMem, Int32), mem, reset)
    end
    
    function iris_task_cmd_init_reset_assign(task::IrisTask, mem::IrisMem, element::Any)::Int32
        ivalue = flat_in_buffer(element)
        return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_assign), Int32, (IrisTask, IrisMem, IRISValue), task, mem, ivalue)
    end

    function iris_task_cmd_init_reset_random_uniform_seq(task::IrisTask, mem::IrisMem, seed::Int, min::Any, max::Any)::Int32
        imin = flat_in_buffer(min)
        imax = flat_in_buffer(max)
        #println("Element: ", element, " buffer:", ivalue)
        return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_random_uniform_seq), Int32, (IrisTask, IrisMem, Clonglong, IRISValue, IRISValue), task, mem, seed, imin, imax)
    end

    function iris_task_cmd_init_reset_arith_seq(task::IrisTask, mem::IrisMem, element::Any, step::Any)::Int32
        ivalue = flat_in_buffer(element)
        istep = flat_in_buffer(step)
        #println("Element: ", element, " buffer:", ivalue)
        return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_arith_seq), Int32, (IrisTask, IrisMem, IRISValue, IRISValue), task, mem, ivalue, istep)
    end

    function iris_task_cmd_init_reset_geom_seq(task::IrisTask, mem::IrisMem, element::Any, step::Any)::Int32
        ivalue = flat_in_buffer(element)
        istep = flat_in_buffer(step)
        #println("Element: ", element, " buffer:", ivalue)
        return ccall(Libdl.dlsym(lib, :iris_task_cmd_init_reset_geom_seq), Int32, (IrisTask, IrisMem, IRISValue, IRISValue), task, mem, ivalue, istep)
    end

    function rand(T, seed, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_random_uniform_seq(dmem, seed, T(0), T(1))
        return dmem
    end

    function zeros(T, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_assign(dmem, T(0))
        return dmem
    end

    function ones(T, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_assign(dmem, T(1))
        return dmem
    end

    function arange(T, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_arith_seq(dmem, T(0), T(1))
        return dmem
    end

    function linspace(T, start, step, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_arith_seq(dmem, T(start), T(step))
        return dmem
    end

    function geomspace(T, start, step, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_geom_seq(dmem, T(start), T(step))
        return dmem
    end

    function iris_data_mem_create(mem::Ptr{IrisMem}, host::Ptr{Cvoid}, size::Csize_t)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create), Int32, (Ptr{IrisMem}, Ptr{Cvoid}, Csize_t), mem, host, size)
    end

    function iris_data_mem_create_ptr(host::Ptr{Cvoid}, size::Csize_t)::Ptr{IrisMem}
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_ptr), Ptr{IrisMem}, (Ptr{Cvoid}, Csize_t), host, size)
    end

    function get_iris_type(T, default=iris_unknown)
        element_type = default
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
        elseif T == Bool
            element_type = iris_bool
        elseif isstructtype(T) 
            element_type = iris_custom_type
        elseif default == iris_pointer
            element_type = iris_unknown
        end
        return element_type
    end

    function iris_data_mem(::Type{T}, dims...; dev_size=nothing, offset=[])  where T
        #size = Csize_t(length(host) * sizeof(T))
        dim_size = dims
        #println(Core.stdout, "Type of element: ", T, " dim:", dims)
        if length(dims) == 1 && isa(dims[1], Tuple)
            dim_size = dims[1]
        end
        host_size = prod(dim_size) * sizeof(T)
        dim = length(dim_size)
        dim_size_v = collect(dim_size)
        element_size = Int32(sizeof(T))
        host_cptr = C_NULL
        #println(Core.stdout, "Type of element: ", T, " dim:", dim, " Size:", host_size, " Element size:", element_size)
        element_type = get_iris_type(T)
        iris_mem = nothing
        if dev_size != nothing 
            if length(offset) == 0
                offset = Base.zeros(Csize_t, dim)
            end
            iris_mem = ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_struct_with_type), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Int32), host_cptr, pointer(offset), pointer(dim_size_v), pointer(dev_size), element_size, dim, Int32(element_type))
        else
            iris_mem = ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_nd), IrisMem, (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Csize_t, Int32), host_cptr, pointer(dim_size_v), dim, element_size, Int32(element_type))
        end
        if isstructtype(T) 
            push_dmem_custom_type(iris_mem.uid, T)
        end
        return iris_mem
    end

    function iris_data_mem(host::Array{T}) where T 
        #size = Csize_t(length(host) * sizeof(T))
        host_size = collect(size(host))
        dim = length(host_size)
        element_size = Int32(sizeof(T))
        host_cptr = reinterpret(Ptr{Cvoid}, pointer(host))
        #println(Core.stdout, "Type of element: ", T, " Size:", size(host), " Element size:", element_size)
        element_type = get_iris_type(T, iris_pointer)
        iris_mem = ccall(Libdl.dlsym(lib, :iris_data_mem_create_struct_nd), IrisMem, (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Csize_t, Int32), host_cptr, host_size, dim, element_size, Int32(element_type))
        if isstructtype(T) 
            push_dmem_custom_type(iris_mem.uid, T)
        end
        return iris_mem
    end

    function iris_data_mem(host::AbstractArray{T,N}, dev_size_array, offset) where {T,N} 
        #size = Csize_t(length(host) * sizeof(T))
        host_size = collect(size(host))
        dev_size = pointer(dev_size_array)
        dim = length(host_size)
        element_size = Int32(sizeof(T))
        ptr = pointer(host)
        host_cptr = reinterpret(Ptr{Cvoid}, ptr)
        #println(Core.stdout, "Type of element: ", T, " Size:", size(host), " Element size:", element_size)
        element_type = get_iris_type(T, iris_pointer)
        if length(offset) == 0
            offset = Base.zeros(Csize_t, dim)
        end
        iris_mem = ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_struct_with_type), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Int32), host_cptr, pointer(offset), host_size, dev_size, element_size, dim, Int32(element_type))
        if isstructtype(T) 
            push_dmem_custom_type(iris_mem.uid, T)
        end
        return iris_mem
    end

    function dmem(::Type{T}, dims...) where T
        return iris_data_mem(T, dims...)
    end

    function dmem(host::Array{T}) where T
        return iris_data_mem(host)
    end

    function dmem_offset(host, dev_size, offset=[]) 
        return iris_data_mem(host, dev_size, offset)
    end

    # Function: void* iris_get_dmem_host(iris_mem brs_mem);
    function iris_get_dmem_host(brs_mem::IrisMem)::Ptr{Cvoid}
        func = Libdl.dlsym(lib, :iris_get_dmem_host)
        return ccall(func, Ptr{Cvoid}, (IrisMem,), brs_mem)
    end

    function host(mem::IrisMem)
        arg_ptr = iris_get_dmem_host(mem)
        if arg_ptr == C_NULL
            return nothing
        end
        size = host_size(mem)
        j_ptr = ptr_reinterpret(mem.uid, 0, 1, arg_ptr, get_type(mem))
        return unsafe_wrap(Array, j_ptr, size, own=false)
    end

    function valid_host(mem::IrisMem)
        arg_ptr = iris_get_dmem_valid_host(mem)
        if arg_ptr == C_NULL
            return nothing
        end
        size = host_size(mem)
        j_ptr = ptr_reinterpret(mem.uid, 0, 1, arg_ptr, get_type(mem))
        return unsafe_wrap(Array, j_ptr, size, own=false)
    end

    function get_type(mem::IrisMem)
        return iris_get_mem_element_type(mem)
    end

    function ndim(mem::IrisMem)
        return iris_dmem_get_dim(mem)
    end

    function host_size(mem::IrisMem)
        size_ptr = iris_dmem_get_host_size(mem)
        dim = iris_dmem_get_dim(mem)
        dims_array = unsafe_wrap(Array, size_ptr, dim)
        return Tuple(dims_array)
    end

    function iris_dmem_get_dim(brs_mem::IrisMem)::Cint
        func = Libdl.dlsym(lib, :iris_dmem_get_dim)
        return ccall(func, Cint, (IrisMem,), brs_mem)
    end

    function iris_dmem_get_host_size(brs_mem::IrisMem)::Ptr{Csize_t}
        func = Libdl.dlsym(lib, :iris_dmem_get_host_size)
        return ccall(func, Ptr{Csize_t}, (IrisMem,), brs_mem)
    end

    function iris_get_mem_element_type(brs_mem::IrisMem)::Cint
        func = Libdl.dlsym(lib, :iris_get_mem_element_type)
        return ccall(func, Cint, (IrisMem,), brs_mem)
    end

    # Function: void* iris_get_dmem_valid_host(iris_mem brs_mem);
    function iris_get_dmem_valid_host(brs_mem::IrisMem)::Ptr{Cvoid}
        func = Libdl.dlsym(lib, :iris_get_dmem_valid_host)
        return ccall(func, Ptr{Cvoid}, (IrisMem,), brs_mem)
    end

    # Function: void* iris_get_dmem_host_fetch(iris_mem brs_mem);
    function iris_get_dmem_host_fetch(brs_mem::IrisMem)::Ptr{Cvoid}
        func = Libdl.dlsym(lib, :iris_get_dmem_host_fetch)
        return ccall(func, Ptr{Cvoid}, (IrisMem,), brs_mem)
    end

    # Function: void* iris_get_dmem_host_fetch_with_size(iris_mem brs_mem, size_t size);
    function iris_get_dmem_host_fetch_with_size(brs_mem::IrisMem, size::Csize_t)::Ptr{Cvoid}
        func = Libdl.dlsym(lib, :iris_get_dmem_host_fetch_with_size)
        return ccall(func, Ptr{Cvoid}, (IrisMem, Csize_t), brs_mem, size)
    end

    # Function: int iris_fetch_dmem_data(iris_mem brs_mem, void* host_ptr);
    function iris_fetch_dmem_data(brs_mem::IrisMem, host_ptr::Ptr{Cvoid})::Cint
        func = Libdl.dlsym(lib, :iris_fetch_dmem_data)
        return ccall(func, Cint, (IrisMem, Ptr{Cvoid}), brs_mem, host_ptr)
    end

    # Function: int iris_fetch_dmem_data_with_size(iris_mem brs_mem, void* host_ptr, size_t size);
    function iris_fetch_dmem_data_with_size(brs_mem::IrisMem, host_ptr::Ptr{Cvoid}, size::Csize_t)::Cint
        func = Libdl.dlsym(lib, :iris_fetch_dmem_data_with_size)
        return ccall(func, Cint, (IrisMem, Ptr{Cvoid}, Csize_t), brs_mem, host_ptr, size)
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

    function iris_data_mem_set_pin_flag(flag::Bool)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_set_pin_flag), Int32, (Int32,), Int32(flag))
    end

    function enable_pin()
        return iris_data_mem_set_pin_flag(true)
    end

    function disable_pin()
        return iris_data_mem_set_pin_flag(false)
    end

    function iris_data_mem_update(mem::IrisMem, host::Ptr{Cvoid})::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_update), Int32, (IrisMem, Ptr{Cvoid}), mem, host)
    end

    function iris_data_mem_update_host_size(mem::IrisMem, host::Ptr{Csize_t})::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_update_host_size), Int32, (IrisMem, Ptr{Csize_t}), mem, host)
    end

    function dmem_update_host_size(mem::IrisMem, host_size)
        dims = collect(Csize_t, host_size)
        iris_data_mem_update_host_size(mem, pointer(dims))
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

    function iris_data_mem_create_tile_struct_with_type(host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32, element_type::Int32)::IrisMem
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_struct_with_type), IrisMem, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32, Int32), host, off, host_size, dev_size, elem_size, dim, element_type)
    end

    function iris_data_mem_create_tile_ptr(host::Ptr{Cvoid}, off::Ptr{Csize_t}, host_size::Ptr{Csize_t}, dev_size::Ptr{Csize_t}, elem_size::Csize_t, dim::Int32)::Ptr{IrisMem}
        return ccall(Libdl.dlsym(lib, :iris_data_mem_create_tile_ptr), Ptr{IrisMem}, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Int32), host, off, host_size, dev_size, elem_size, dim)
    end

    function iris_data_mem_update_bc(mem::IrisMem, bc::Int32, row::Int32, col::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_data_mem_update_bc), Int32, (IrisMem, Int32, Int32, Int32), mem, bc, row, col)
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

    function iris_dev_get_stream(device::Int, index::Int)::Ptr{Cvoid}
        return ccall(Libdl.dlsym(lib, :iris_dev_get_stream), Ptr{Cvoid}, (Int32, Int32), Int32(device), Int32(index))
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

    function release(mem::IrisMem)::Int32
        return iris_mem_release(mem)
    end

    function iris_graph_create(graph::Ptr{IrisGraph})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_create), Int32, (Ptr{IrisGraph},), graph)
    end

    function iris_graph_create_null(graph::Ptr{IrisGraph})::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_create_null), Int32, (Ptr{IrisGraph},), graph)
    end

    function iris_is_graph_null(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_is_graph_null), Int32, (IrisGraph,), graph)
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

    function iris_graph_retain(graph::IrisGraph, flag::Int32)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_retain), Int32, (IrisGraph, Int32), graph, flag)
    end

    function iris_graph_release(graph::IrisGraph)::Int32
        return ccall(Libdl.dlsym(lib, :iris_graph_release), Int32, (IrisGraph,), graph)
    end

    function release(graph::IrisGraph)::Int32
        return iris_graph_release(graph)
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

    function wait(graph::IrisGraph)::Int32
        return iris_graph_wait(graph)
    end

    function wait(graphs::Vector{IrisGraph})::Int32
        return iris_graph_wait_all(Int32(length(graphs)), pointer(graphs))
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

    function gc_safe_enter()
        gc_state = @ccall(jl_gc_safe_enter()::Int8)
        return gc_state
    end

    function gc_leave(gc_state)
        clear_map()
        @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
        return gc_state
    end

    function clear_map() 
        for (key, value) in Main.__iris_dmem_map 
            IrisHRT.iris_mem_release(value)
        end
        # Clear all elements in the dictionary
        empty!(Main.__iris_dmem_map)
        empty!(Main.__iris_dmem_custom_type)
        empty!(Main.__iris_taskid_paramid_custom_type)
        empty!(Main.__iris_cuda_devno_stream)
        empty!(Main.__iris_hip_devno_stream)
    end

    function check_update_dmem_host_size(mem, larray)
        hsize = host_size(mem)
        total_elements = prod(hsize) 
        array_elements = length(larray)
        if total_elements != array_elements
            #println(Core.stdout, " hsize", hsize)
            #println(Core.stdout, " prev elems: ", total_elements)
            #println(Core.stdout, " current elems: ", array_elements)
            dmem_update_host_size(mem, size(larray))
        end
    end

    function get_init_task_dmem(::Type{T}, dims...; task=nothing, dmem=nothing) where T
        task0 = task
        if task0 == nothing
            task0 = iris_task_create_struct()
        end
        dmem0 = dmem
        if dmem0 == nothing
            dmem0 = iris_data_mem(T, dims...)
        end
        return (task=task0, dmem=dmem0)
    end
    
    function get_init_task_dmem(host::Array{T}; task=nothing, dmem=nothing) where T
        task0 = task
        if task0 == nothing
            task0 = iris_task_create_struct()
        end
        dmem0 = dmem
        if dmem0 == nothing
            p_array = pointer(host)
            if !haskey(Main.__iris_dmem_map, p_array)
                dmem0 = iris_data_mem(host)
                Main.__iris_dmem_map[p_array] = dmem0
            else
                dmem0 = Main.__iris_dmem_map[p_array]
            end
        end
        return (task=task0, dmem=dmem0)
    end
    
    function rand_task(::Type{T}, seed, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem)
        iris_task_cmd_init_reset_random_uniform_seq(task0, dmem0, seed, T(0), T(1))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function zeros_task(::Type{T}, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem) 
        iris_task_cmd_init_reset_assign(task0, dmem0, T(0))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function ones_task(::Type{T}, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem)
        iris_task_cmd_init_reset_assign(task0, dmem0, T(1))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function arange_task(::Type{T}, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem)
        iris_task_cmd_init_reset_arith_seq(task0, dmem0, T(0), T(1))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function linspace_task(::Type{T}, start, step, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem)
        iris_task_cmd_init_reset_arith_seq(task0, dmem0, T(start), T(step))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function geomspace_task(::Type{T}, start, step, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem)
        iris_task_cmd_init_reset_geom_seq(task0, dmem0, T(start), T(step))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end


    function rand_task(host::Array{T}; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task)
        iris_task_cmd_init_reset_random_uniform_seq(task0, dmem0, seed, T(0), T(1))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function zeros_task(host::Array{T}; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task) 
        iris_task_cmd_init_reset_assign(task0, dmem0, T(0))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function ones_task(host::Array{T}; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task)
        iris_task_cmd_init_reset_assign(task0, dmem0, T(1))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function arange_task(host::Array{T}; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task)
        iris_task_cmd_init_reset_arith_seq(task0, dmem0, T(0), T(1))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function linspace_task(host::Array{T}; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task)
        iris_task_cmd_init_reset_arith_seq(task0, dmem0, T(start), T(step))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end

    function geomspace_task(host::Array{T}; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task)
        iris_task_cmd_init_reset_geom_seq(task0, dmem0, T(start), T(step))
        if submit
            iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        end
        return (task=task0, dmem=dmem0)
    end


    function identify_in_out(julia_kernel_type, gws, kernel, args...)
        f = nothing 
        if (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native && isdefined(Main, Symbol(kernel*"_cuda"))
            f = getfield(Main, Symbol(kernel*"_cuda"))
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native && isdefined(Main, Symbol(kernel*"_hip"))
            f = getfield(Main, Symbol(kernel*"_hip"))
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native && isdefined(Main, Symbol(kernel*"_openmp"))
            f = getfield(Main, Symbol(kernel*"_openmp"))
        elseif julia_kernel_type == IrisHRT.iris_julia_kernel_abstraction && isdefined(Main, Symbol(kernel))
            f = getfield(Main, Symbol(kernel))
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_jacc && isdefined(Main, Symbol(kernel))
            f = getfield(Main, Symbol(kernel))
        else
            error("Couldn't find function definition for kernel: ", kernel)
            return (in=[], out=[])
        end
        if f == nothing
            return (in=[], out=[])
        end
        m = nothing
        offset = 0
        if (julia_kernel_type & 0xFFFF) == iris_julia_jacc
            i = 0 
            j = 0
            k = 0
            offset = length(gws)
            if offset == 1
                m = @which f(i, args...)   # Get method instance
            elseif offset == 2
                m = @which f(i, j, args...)   # Get method instance
            elseif offset == 3
                m = @which f(i, j, k, args...)   # Get method instance
            else
                error("Couldn't find JACC function for gws:", gws, " for kernel:", kernel)
            end
        else
            m = @which f(args...)   # Get method instance
        end
        ast = Base.uncompressed_ast(m)  # Extract AST
        #println(ast)

        #ci = @code_lowered sample_function(10)
        ci = ast

        nargs = count(flag -> flag == :argument, ci.slotflags)
        #println("Slot flags:", ci.slotflags)
        #println("Slot names:", ci.slotnames)
        #println("Slot ssavalue types:", ci.ssavaluetypes)
        #println("Slot types:", typeof(ci.slotnames))
        #println("Number of function arguments: ", nargs)
        #println()

        vector_in = []
        vector_out = []

        # Iterate over each expression in the lowered code.
        for (i, item) in enumerate(ci.code)
            #println("Expression $(i):")
            #println("  ", item)
            expr = item
            # Process only expressions (Expr objects) that are assignments.
            if expr isa Expr && expr.head == :call
                # The called function is usually in expr.args[1]. It might be directly a Symbol
                # or an expression (e.g. a getfield call) that contains the symbol.
                f = expr.args[1]
                fname = nothing
                
                if f isa Symbol
                    fname = f
                elseif f isa Expr && f.head == :getfield
                    # For example, this might be something like (getfield Base :getindex)
                    fname = getfield(f, :args)[2]
                elseif f isa Core.SSAValue
                    nothing
                else
                    #println("f expr: ", typeof(f))
                    fname = f.name
                end
                #println("------- Call found----- func:", fname == :getindex)
                # Check if this call is to getindex or setindex!
                if fname == :getindex || fname == :setindex!
                    #println("Expression $(i):", item)
                    #println("  Found a call to $(fname):")
                    # Iterate through the call arguments (skipping the first which is the function itself).
                    for (j, arg) in enumerate(expr.args[2:2])
                        # If the argument is a Symbol, print it.
                        if arg isa Symbol
                            #println("    Argument $(j) is a Symbol: ", arg, " (type: ", typeof(arg), ")")
                        elseif arg isa Core.SlotNumber
                            sn = arg.id
                            slot_index = Int(arg.id)
                            original_var = ci.slotnames[slot_index]
                            o_original_var = getfield(ci, :slotnames)[slot_index]
                            if fname == :getindex
                                push!(vector_in, slot_index)
                            end
                            if fname == :setindex!
                                push!(vector_out, slot_index)
                            end
                        else
                            #println("    Argument $(j): ", arg)
                        end
                    end
                end
            else
                #println("  (Not an assignment expression.)")
            end
            #println()  # Blank line for clarity
        end
        vector_in = sort!(unique(vector_in))
        vector_out = sort!(unique(vector_out))
        #println("Automatic in:", vector_in, " and out:", vector_out)
        a_in = []
        a_out = []
        for i in vector_in
            push!(a_in, args[i-1-offset])
        end
        for i in vector_out
            push!(a_out, args[i-1-offset])
        end
        return (in=a_in, out=a_out)
    end

    function parallel_for((L, M, N)::Tuple{Int64, Int64, Int64}, f::Function, x... ; task=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true, dependencies=[], flush=Any[], metadata=nothing)
        return IrisHRT.task(gws=(L,M,N), task=task, parallel_for=true, submit=submit, policy=policy, wait=wait, flush=flush, dependencies=dependencies, metadata=metadata, kernel=String(Symbol(f)), args=Any[x...])
    end

    function parallel_for((M, N)::Tuple{Int64, Int64}, f::Function, x... ; task=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true, dependencies=[], flush=Any[], metadata=nothing)
        return IrisHRT.task(gws=(M,N), task=task, parallel_for=true, submit=submit, policy=policy, wait=wait, flush=flush, dependencies=dependencies, metadata=metadata, kernel=String(Symbol(f)), args=Any[x...])
    end

    function parallel_for(N::Int64, f::Function, x... ; task=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true, dependencies=[], flush=Any[], metadata=nothing)
        return IrisHRT.task(gws=N, task=task, parallel_for=true, submit=submit, policy=policy, wait=wait, flush=flush, dependencies=dependencies, metadata=metadata, kernel=String(Symbol(f)), args=Any[x...])
    end

    function parallel_reduce((M,N)::Tuple{Int64,Int64}, op, f::Function, x... ; init, task=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true, dependencies=[], flush=Any[], metadata=nothing)
        return IrisHRT.task(gws=(M,N), task=task, parallel_reduce=true, submit=submit, policy=policy, wait=wait, flush=flush, dependencies=dependencies, metadata=metadata, kernel=String(Symbol(f)), args=Any[x...])
    end

    function parallel_reduce(N::Int64, op, f::Function, x... ; init, task=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true, dependencies=[], metadata=nothing)
        return IrisHRT.task(gws=N, task=task, parallel_reduce=true, submit=submit, policy=policy, wait=wait, flush=flush, dependencies=dependencies, metadata=metadata, kernel=String(Symbol(f)), args=Any[x...])
    end

    function task(; in=Any[], input=Any[], out=Any[], output=Any[], flush=Any[], auto_in_out=false, lws=Int64[], gws=Int64[], off=Int64[], policy=IrisHRT.iris_roundrobin, wait=true, core=false, ka=false, parallel_reduce=false, parallel_for=false, host=false, task=nothing, metadata=nothing, kernel="kernel", args=[], submit=true, dependencies=[])
        call_args = args
        mem_params = Dict{Any, Any}()
        t_args = Tuple(args)
        auto_in = Any[]
        auto_out = Any[]
        if ! auto_in_out && length(in)==0 && length(out)==0 && length(args)!=0
            auto_in_out = true
        end
        julia_kernel_type = IrisHRT.iris_julia_native
        if core
            julia_kernel_type = IrisHRT.iris_core_native
        elseif ka
            julia_kernel_type = IrisHRT.iris_julia_kernel_abstraction
        elseif parallel_for
            julia_kernel_type = IrisHRT.iris_julia_jacc_parallel_for
        elseif parallel_reduce
            julia_kernel_type = IrisHRT.iris_julia_jacc_parallel_reduce
        elseif host
            julia_kernel_type = IrisHRT.iris_julia_native_host
        end
        if auto_in_out
            (auto_in, auto_out) = identify_in_out(julia_kernel_type, gws, kernel, t_args...)
        end
        for larray in vcat(out, output, flush, auto_out)
            p_array = larray
            if !isa(larray, IrisMem) 
                #println("Array is not IrisMem")
                p_array = pointer(larray)
                #println(Core.stdout, "Out ---- ", larray, " typeof:", typeof(larray))
                if !haskey(Main.__iris_dmem_map, p_array)
                    # Generate DMEM object if not found
                    #println("Array not found in global map. Out/Flush Creating DMEM object for: ", pointer(larray))
                    Main.__iris_dmem_map[p_array] = IrisHRT.iris_data_mem(larray)
                    #finalizer(dmem_finalizer, larray)
                else
                    #println("Array already mapped to Output DMEM object: ", larray)
                end
            end
            #println(Core.stdout, " out inserting p_array: ", p_array)
            mem_params[p_array] = IrisHRT.iris_w
        end
        for larray in vcat(input, in, auto_in)
            p_array = larray
            if !isa(larray, IrisMem) 
                p_array = pointer(larray)
                #println(Core.stdout, "------in inserting iris_r p_array: ", p_array)
                #println(Core.stdout, "In ---- ", larray, " typeof:", typeof(larray))
                #println(Core.stdout, "In ----  typeof:", typeof(larray))
                #println("DMEM object t: ", larray)
                if !haskey(Main.__iris_dmem_map, p_array)
                    # Generate DMEM object if not found
                    #println("Array not found in global map. In Creating DMEM object for: ", pointer(larray))
                    #println("Array is not IrisMem")
                    Main.__iris_dmem_map[p_array] = IrisHRT.iris_data_mem(larray)
                    #finalizer(dmem_finalizer, larray)
                else
                    #println("Array already mapped to Input DMEM object: ", larray)
                end
            end
            if !haskey(mem_params, p_array)
                #println(Core.stdout, "---- in inserting iris_r p_array: ", p_array, " type:", typeof(larray))
                mem_params[p_array] = IrisHRT.iris_r
            else
                #println(Core.stdout, " in inserting iris_rw p_array: ", p_array, " type:", typeof(larray))
                mem_params[p_array] = IrisHRT.iris_rw
            end
        end
        # create task structure
        task0 = task
        if task0 == nothing
            task0 = iris_task_create_struct()
        end

        params_info = Int32[]
        params = []
        kernel_params = []
        #println(Core.stdout, "--------------")
        #println(Core.stdout, "content: ", mem_params)
        #println(Core.stdout, "--------------")
        for (index, arg) in enumerate(call_args)
            #println(Core.stdout, "- s - s - s - : ", index, " type:", typeof(arg))
            if isa(arg, Array)
                p_arg = pointer(arg)
                #println(Core.stdout, "- s - s - s - : ", index, " type:", typeof(arg), " pointer:", p_arg, " length:", length(arg), " size:", size(arg))
                if haskey(Main.__iris_dmem_map, p_arg)
                    #push!(kernel_params, (Main.__iris_dmem_map[p_arg], mem_params[p_arg]))
                    push!(params, Ref(Main.__iris_dmem_map[p_arg]))
                    push!(params_info, Int32(mem_params[p_arg]))
                    if get_type(Main.__iris_dmem_map[p_arg]) == iris_custom_type
                        push_custom_type(task0.uid, index, get_dmem_custom_type(Main.__iris_dmem_map[p_arg]))
                    end
                    mem = Main.__iris_dmem_map[p_arg]
                    check_update_dmem_host_size(mem, arg)
                else
                    Main.__iris_dmem_map[p_arg] = IrisHRT.iris_data_mem(arg)
                    #finalizer(dmem_finalizer, arg)
                    #push!(kernel_params, (Main.__iris_dmem_map[p_arg], IrisHRT.iris_r))
                    push!(params, Ref(Main.__iris_dmem_map[p_arg]))
                    push!(params_info, Int32(IrisHRT.iris_rw))
                    if get_type(Main.__iris_dmem_map[p_arg]) == iris_custom_type
                        push_custom_type(task0.uid, index, get_dmem_custom_type(Main.__iris_dmem_map[p_arg]))
                    end
                end
            elseif isa(arg, IrisMem)
                #p_arg = pointer(arg)
                #push!(kernel_params, (arg, mem_params[arg])) 
                push!(params, Ref(arg))
                push!(params_info, Int32(mem_params[arg]))
                if get_type(arg) == iris_custom_type
                    push_custom_type(task0.uid, index, get_dmem_custom_type(arg))
                end
            else
                # Scalar element
                #push!(kernel_params, arg)
                type_data = get_iris_type(typeof(arg))
                #println(Core.stdout, " type_data: ", type_data, " arg:", arg, " type of arg: ", typeof(arg))
                push!(params_info, type_data | sizeof(arg))
                push!(params, Ref(arg))
            end
        end
        nparams = Int64(length(params))
        #println(Core.stdout, "kernel_params: ", kernel_params)
        #println(Core.stdout, "kernel name:", kernel)
        #println(Core.stdout, "off:", off)
        #println(Core.stdout, "gws:", gws)
        #println(Core.stdout, "lws:", lws)
        #println(Core.stdout, "kernel_params     :", kernel_params)

        # Set kernel type
        if (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native 
            ccall(Libdl.dlsym(lib, :iris_task_enable_julia_interface), Int32, (IrisTask, Int32), task0, Int32(julia_kernel_type))
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_kernel_abstraction
            ccall(Libdl.dlsym(lib, :iris_task_enable_julia_interface), Int32, (IrisTask, Int32), task0, Int32(julia_kernel_type))
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_jacc
            ccall(Libdl.dlsym(lib, :iris_task_enable_julia_interface), Int32, (IrisTask, Int32), task0, Int32(julia_kernel_type))
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_core_native
            # Do nothing here
        end
        
        #task0=IrisHRT.iris_task_julia(kernel, length(gws), off, gws, lws, kernel_params)
        if isa(gws, Number)
            gws = [gws]
        end
        if isa(lws, Number)
            lws = [lws]
        end
        if isa(off, Number)
            off = [off]
        end
        if length(gws) > 0 && isa(gws, Tuple)
            gws = collect(gws)
        end
        if length(lws) > 0 && isa(lws, Tuple)
            lws = collect(lws)
        end
        if length(off) > 0 && isa(off, Tuple)
            off = collect(off)
        end
        #println(Core.stdout, "--- gws:", gws, " typeof:", typeof(gws), " length:", length(gws))
        #println(Core.stdout, "--- lws:", lws, " typeof:", typeof(lws))
        #println(Core.stdout, "--- off:", off)
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
            #println(Core.stdout, " lws: ", lws, " typeof:", typeof(lws))
            lws_c = reinterpret(Ptr{UInt64}, pointer(lws))
        end
        c_params = reinterpret(Ptr{Ptr{Cvoid}}, pointer(params))
        # Create kernel with task
        GC.@preserve gws lws off begin
            ccall(Libdl.dlsym(lib, :iris_task_kernel), Int32, (IrisTask, Ptr{Cchar}, Int32, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}), task0, pointer(kernel), Int32(length(gws)), off_c, gws_c, lws_c, Int32(nparams), c_params, pointer(params_info))
        end
        if metadata != nothing
            IrisHRT.set_metadata(task0, metadata)
        end
        for mem in flush 
            if isa(mem, IrisMem)
                IrisHRT.iris_task_dmem_flush_out(task0, mem)
            else
                p_mem = pointer(mem)
                IrisHRT.iris_task_dmem_flush_out(task0, Main.__iris_dmem_map[p_mem])
            end
        end
        dependencies = [x for x in dependencies if x !== nothing]
        if length(dependencies) > 0
            iris_task_depend(task0, dependencies)
        end
        if isa(policy, Function)
            policy_ptr = String(Symbol(policy))
            #println("Julia Policy name>>>", policy_ptr)
            IrisHRT.iris_task_set_julia_policy(task0, policy_ptr)
            policy = IrisHRT.iris_julia_policy
        end
        if submit
            IrisHRT.iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        else
            IrisHRT.iris_task_set_policy(task0, Int32(policy))
        end
        return task0
    end
    function describe_backend(backend::IRISBackend)
        println("Backend Name: ", backend.backend_name)
    end

end  # module Iris

include("Tiling2D.jl")
include("Tiling3D.jl")

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
and(a, b) = a && b
or(a, b) = a || b
const ARRAY_TO_DMEM_MAP = Dict{Any, Any}()
const __iris_dmem_subarray_map = Dict{Any, Any}()
const __iris_dmem_map = Dict{Any, Any}()
const __iris_dmem_parallel_reduce_map = Dict{Any, Any}()
const __iris_dmem_mirrors_map = Dict{Any, Any}()
const __iris_cuda_devno_stream = Dict{Any, Any}()
const __iris_hip_devno_stream = Dict{Any, Any}()
const __iris_dmem_custom_type = Dict{Any, Any}()
const __iris_dmem_2_task = Dict{Any, Any}()
const IRIS_SUCCESS          =  0
const IRIS_ERROR            = -1
const IRIS_WARNING          = -2
const iris_path = ENV["IRIS"]
module DummyAMDGPU
    __precompile__()
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
    __precompile__()
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
    true
catch e
    println("CUDA is not defined")
    false
end

if cuda_available
    const MyCUDA = CUDA
else
    const MyCUDA = DummyCUDA
end

__precompile__()
const hip_available = try
    using AMDGPU
    if AMDGPU.has_rocm_gpu()
        #println(Core.stdout, "AMDGPU is defined and a ROCm GPU is available!")
        true  # Assign true if AMDGPU is available
    else
        println(Core.stdout, "AMDGPU is defined but no ROCm GPU available.")
        false  # Assign false if AMDGPU is defined but no GPU
    end
catch e
    println(Core.stdout, "AMDGPU is not defined!")
    false  # Assign false if AMDGPU is not available
end
if hip_available
    const MyAMDGPU = AMDGPU
else
    const MyAMDGPU = DummyAMDGPU
end

function __iris_jacc_elem_access(i, x)
    @inbounds x[i]
end
const __iris_jacc_add = __iris_jacc_elem_access
const __iris_jacc_munus = __iris_jacc_elem_access
const __iris_jacc_mul = __iris_jacc_elem_access
const __iris_jacc_div = __iris_jacc_elem_access
const __iris_jacc_min = __iris_jacc_elem_access
const __iris_jacc_max = __iris_jacc_elem_access

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
import Base: reshape
import Base: reinterpret
module IrisHRT

    __precompile__()
    import Base: Callable
    import Base: show

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
    const default_init = JACC.default_init

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
    @enum DeviceModel iris_cuda=1 iris_hip=3 iris_levelzero=4 iris_opencl=5 iris_openmp=6 iris_qiree=7 iris_model_all=(1<<25)

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
    const iris_julia_jacc_parallel_reduce = (1 << 16) | 3

    const iris_jacc_op_plus  = 1
    const iris_jacc_op_minus = 2
    const iris_jacc_op_mul   = 3
    const iris_jacc_op_div   = 4
    const iris_jacc_op_sum   = 5
    const iris_jacc_op_prod  = 6
    const iris_jacc_op_min   = 7
    const iris_jacc_op_max   = 8
    const iris_jacc_op_and   = 9
    const iris_jacc_op_or    = 10


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
    const iris_ocl = 1 << 26
    const iris_block_cycle = 1 << 27
    const iris_custom = 1 << 28
    const iris_julia_policy = 1 << 29
    const iris_all = 0x7FFFFFFF

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

    struct IrisTaskMem
        task::IrisTask
        dmem::IrisMem
    end

    struct IrisMem1D{T} <: AbstractVector{T}
        mem::IrisMem
    end

    struct IrisMem2D{T} <: AbstractMatrix{T}
        mem::IrisMem
    end

    struct IrisMem3D{T} <: AbstractArray{T,3}
        mem::IrisMem
    end

    struct IrisMem4D{T} <: AbstractArray{T,4}
        mem::IrisMem
    end

    struct IrisGraph
        class_obj::Ptr{Cvoid}
        uid::Culong
    end

    # Define the struct for IRISBackend
    struct IRISBackend
        backend_name::String
    end

    field_task(x::IrisTaskMem) = x.task
    field_task(x::IrisTask) = x
    field_dmem(x::IrisTaskMem) = x.dmem
    field_dmem(x::IrisMem) = x

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

    const UTask = Union{IrisTask, IrisTaskMem}
    const UMem  = Union{IrisMem, IrisTaskMem}
    const USize = Union{Csize_t, Int64}

    export iris_init
    export kernel_julia_wrapper
    export task_policy_handler
    export convert_args

    include("IrisUtils.jl")
    include("IrisLib.jl")
    include("IrisKernels.jl")
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

    function kernel_julia_wrapper(task_id::Culong, julia_kernel_type::Cint, target::Cint, devno::Cint, ctx::Ptr{Cvoid}, async_flag_int::Cint, stream_index::Cint, stream::Ptr{Ptr{Cvoid}}, nstreams::Cint, args_type::Ptr{Cint}, args::Ptr{Ptr{Cvoid}}, param_size::Ptr{Csize_t}, param_dim_size::Ptr{Csize_t}, nparams::Cint, c_gws::Ptr{Csize_t}, c_blocks::Ptr{Csize_t}, c_threads::Ptr{Csize_t}, shared_mem_size::Csize_t, dim::Cint, kernel_name::Ptr{Cchar})::Cint
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
            size_dims = remove_trailing_ones2(size_dims)
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
                    func_name = unsafe_string(kernel_name)
                    println(Core.stdout, "[Error-1] Kernel: ", func_name, " I is not pointer:", i, " current_type:", current_type)
                    error("Unsupported type: ", typeof(current_type))
                end
            else # Assuming 3 is for Char
                #println(Core.stdout, "I is a pointer:", i, " arg_ptr:", arg_ptr)
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
                    println(Core.stdout, "[Error-2] I is not pointer index:", i-start_index+1, " current_type:", current_type)
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
        j_gws = []
        # Iterate through each argument
        #println(Core.stdout, " dim:", dim)
        for i in 0:(dim - 1)
            # Load the type of the current argument
            push!(j_gws, UInt64(unsafe_load(c_gws+i*sizeof(Csize_t)))+0)
            push!(j_threads, UInt64(unsafe_load(c_threads+i*sizeof(Csize_t)))+0)
            push!(j_blocks, UInt64(unsafe_load(c_blocks+i*sizeof(Csize_t)))+0)
        end
        j_gws = Tuple(j_gws)
        j_threads = Tuple(j_threads)
        j_blocks = Tuple(j_blocks)

        #println(Core.stdout, " j_gws: ", j_gws)
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
                call_cuda_kernel_jacc(julia_kernel_type, devno, dim, func_name, j_gws, j_threads, j_blocks, cu_ctx, cu_stream, julia_args, b_async_flag, shared_mem_size)
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
                call_hip_kernel_jacc(julia_kernel_type, devno, dim, func_name, j_gws, j_threads, j_blocks, hip_ctx, hip_stream, julia_args, b_async_flag, shared_mem_size)
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
            #println(Core.stdout, "Calling OpenMP kernel: ", func_name)
            if julia_kernel_type == iris_julia_kernel_abstraction 
                call_openmp_kernel_ka(julia_kernel_type, func_name, j_threads, j_blocks, julia_args, b_async_flag)
            elseif (julia_kernel_type & 0xFFFF) == iris_julia_jacc
                call_openmp_kernel_jacc(julia_kernel_type, dim, func_name, j_gws, j_threads, j_blocks, julia_args, b_async_flag)
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


    # Bind functions from the IRIS library using ccall
    function iris_init(sync)::Int32
        ENV["LD_LIBRARY_PATH"] =  Main.iris_path * "/lib64:" * Main.iris_path * "/lib:" * ENV["LD_LIBRARY_PATH"]
        policy_ptr = @cfunction(task_policy_handler, Cint, (IrisTask, Ptr{Cchar}, Ptr{IrisDevice}, Int32, Ptr{Int32}))
        global stored_policy_ptr = policy_ptr
        func_ptr = @cfunction(kernel_julia_wrapper, Cint, (Culong, Cint, Cint,        Cint,        Ptr{Cvoid}, Cint, Cint,               Ptr{Ptr{Cvoid}},         Cint,           Ptr{Cint},       Ptr{Ptr{Cvoid}},         Ptr{Csize_t},  Ptr{Csize_t}, Cint,          Ptr{Csize_t}, Ptr{Csize_t},          Ptr{Csize_t}, Csize_t,        Cint,      Ptr{Cchar}))
        global stored_func_ptr = func_ptr
        use_julia_threads = false
        if use_julia_threads
            flag = iris_julia_init(func_ptr, 1)
            flag = flag & iris_init_c(sync)
            flag = flag & iris_julia_policy_init(policy_ptr)
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
            flag = iris_julia_init(func_ptr, 0)
            flag = flag & iris_init_c(sync)
            flag = flag & iris_julia_policy_init(policy_ptr)
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


    # Call Julia kernel 
    # CUDA kernel wrapper (KernelAbstraction kernels)
    function call_cuda_kernel_jacc(julia_kernel_type::Any, devno::Any, dim, func_name::String, j_gws::Any, threads::Any, blocks::Any, ctx::Any, stream::Any, args::Any, async_flag::Bool=false, shared_mem_size=0)
        if !Main.cuda_available
            error("CUDA device not available.")
        end

        func_name_target = func_name
        # Initialize CUDA
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #gws = map(*, threads, blocks)
        gws = j_gws
        #println(Core.stdout, "IrisHRT threads:", threads, " blocks:", blocks, " gws:", gws)
        if length(threads)==1
            threads = threads[1]
        end
        if length(blocks)==1
            blocks = blocks[1]
        end
        shmem = shared_mem_size 
        backend = MyCUDA.CUDAKernels.CUDABackend(false, false)
        spec = JACC.LaunchSpec{MyCUDA.CUDAKernels.CUDABackend}(stream=stream, threads=threads, blocks=blocks, shmem_size=shmem, sync=true)
        if async_flag
            MyCUDA.device!(Int(devno))
            MyCUDA.context!(ctx)
            MyCUDA.stream!(stream)
            spec.sync = false
        else
            spec.stream = JACC.default_stream(MyCUDA.CUDAKernels.CUDABackend)
        end
        #gws = Tuple(remove_trailing_ones2(collect(gws)))
        if length(gws)==1
            gws = gws[1]
        end
        #println(Core.stdout, " func: ", func_name, " gws:", gws)
        if julia_kernel_type == iris_julia_jacc_parallel_for
            JACC.parallel_for(spec, gws, func, args_tuple...)
        elseif julia_kernel_type == iris_julia_jacc_parallel_reduce
            op = get_jacc_op(args_tuple[1])
            init, out_mem, buffer = args_tuple[2:4]
            spec.ret = out_mem
            spec.buffer = buffer
            #println(Core.stdout, " op: ", op, " init:", init, " ret:", spec.ret, " buffer:", spec.buffer, " gws:", gws, " func:", func)
            JACC.parallel_reduce(spec, gws, op, func, args_tuple[5:end]...; init)
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
                    spec = (devno=devno, all_streams=all_streams, threads=threads, blocks=blocks, stream=stream, context=ctx)
                    invokelatest(func, spec, args_tuple...)
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
                spec = (devno=devno, all_streams=all_streams, threads=threads, blocks=blocks, stream=stream, context=ctx)
                invokelatest(func, spec, args_tuple...)
            end

            #CUDA.@sync @cuda threads=threads blocks=blocks func(args_tuple...)
            #func1(threads, blocks, args_tuple)
        end
        #synchronize(blocking = true)
    end

    function call_hip_kernel_jacc(julia_kernel_type::Any, devno::Any, dim, func_name::String, j_gws::Any, threads::Any, blocks::Any, ctx::Any, stream::Any, args::Any, async_flag::Bool=false, shared_mem_size=0)
        #iris_println("Func $func_name")
        # Initialize CUDA
        #func = getfield(Main, Symbol(func_name))
        func = getFunction(func_name)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        #gws = map(*, threads, blocks)
        gws = j_gws
        shmem = shared_mem_size
        if length(threads)==1
            threads = threads[1]
        end
        if length(blocks)==1
            blocks = blocks[1]
        end
        shmem=0
        spec = JACC.LaunchSpec{MyAMDGPU.ROCBackend}(stream=stream, threads=threads, blocks=blocks, shmem_size=shmem, sync=true)
        if async_flag
            MyAMDGPU.context!(ctx)
            MyAMDGPU.stream!(stream)
            spec.sync = false
        else
            MyAMDGPU.context!(ctx)
            spec.stream = JACC.default_stream(MyAMDGPU.ROCBackend)
        end
        backend = MyAMDGPU.ROCBackend()
        #gws = Tuple(remove_trailing_ones2(collect(gws)))
        if length(gws)==1
            gws = gws[1]
        end
        if julia_kernel_type == iris_julia_jacc_parallel_for
            JACC.parallel_for(spec, gws, func, args_tuple...)
        else
            op = get_jacc_op(args_tuple[1])
            init, out_mem, buffer = args_tuple[2:4]
            spec.ret = out_mem
            spec.buffer = buffer
            JACC.parallel_reduce(spec, gws, op, func, args_tuple[5:end]...; init)
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
                    spec = (devno=devno, all_streams=all_streams, threads=threads, blocks=blocks, stream=stream, context=ctx)
                    invokelatest(func, spec, args_tuple...)
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
                spec = (devno=devno, all_streams=all_streams, threads=threads, blocks=blocks, stream=stream, context=ctx)
                invokelatest(func, spec, args_tuple...)
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
            spec = (dev=devno,)
            invokelatest(func, spec, args_tuple...)
        end
    end

    function call_openmp_kernel_jacc(julia_kernel_type::Any, dim, func_name::String, j_gws::Any, threads::Any, blocks::Any, args::Any, async_flag::Bool=false)
        func_name_target = func_name
        #func = getfield(IrisKernelImpl, Symbol(func_name_target))
        #func = getfield(Main, Symbol(func_name_target))
        func = getFunction(func_name_target)
        # Convert the array of arguments to a tuple
        args_tuple = Tuple(args)
        # Call the function with arguments
        gws = j_gws
        backend = JACC.ThreadsImpl.ThreadsBackend()
        nthreads = Threads.nthreads()
        num_cpus = Sys.CPU_THREADS
        #gws = Tuple(remove_trailing_ones2(collect(gws)))
        if length(gws) == 1
            gws = gws[1]
        end
        if julia_kernel_type == iris_julia_jacc_parallel_for
            JACC.parallel_for(backend, gws, func, args_tuple...)
        else
            #println(Core.stdout, "------op:", typeof(args_tuple[1]), " -- ", typeof(args_tuple[2]))
            #println(Core.stdout, "data: ", args_tuple[1])
            #println(Core.stdout, "------op:", typeof(args_tuple[1]), " -- ", typeof(args_tuple[2]))
            op = get_jacc_op(args_tuple[1])
            init, out_mem, buffer = args_tuple[2:4]
            out_mem[1] = JACC.parallel_reduce(backend, gws, op, func, args_tuple[5:end]...; init)
            #copy_host(out_dmem, out)
        end
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

    function finalize(;gc_leave=true)::Int32
        if gc_leave
            IrisHRT.gc_leave(gc_state)
        end
        return iris_finalize()
    end

    function synchronize()::Int32
        return iris_synchronize()
    end

    function retain(task::Union{IrisTask,IrisTaskMem}, flag=1)
        iris_task_retain(field_task(task), Int32(flag))
    end

    function ndevs()::Int
        return Int(iris_ndevices())
    end

    function empty_task()
        return iris_task_create_struct()
    end

    function iris_task_depend(task::IrisTask, tasks::Vector{IrisTask})::Int32
        return iris_task_depend(task, Int32(length(tasks)), pointer(tasks)) 
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

    function set_metadata(task::Union{IrisTask,IrisTaskMem}, metadata, index=nothing)
        task = field_task(task)
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

    function is_dmem_in_cache(data)
        return haskey(Main.__iris_dmem_map, pointer(data))
    end 

    function get_dmem_from_cache(data, new=false, create=true; name="")
        if isa(data, Union{IrisMem, IrisTaskMem})
            return field_dmem(data)
        else
            if isa(data, SubArray)
                offsets = view_offsets_elements(data)
                dev_size = view_device_size(data)
                parent = data.parent
                host_size = size(parent)
                key = (pointer(data), host_size, offsets, dev_size)
                if !new && haskey(Main.__iris_dmem_subarray_map, key)
                    #if name != ""
                    #    println(Core.stdout, " name:", name, " dev_size:", dev_size, " offsets:", offsets, " host:", size(parent), " pointer:", pointer(data), " :: ", pointer(parent), "  E")
                    #end
                    return Main.__iris_dmem_subarray_map[key]
                else
                    if create
                        #if name != ""
                        #    println(Core.stdout, " name:", name, " dev_size:", dev_size, " offsets:", offsets, " host:", size(parent), " pointer:", pointer(data), " :: ", pointer(parent), "  N")
                        #end
                        d_mem = iris_data_mem(data)
                        Main.__iris_dmem_subarray_map[key] = d_mem
                        return d_mem
                    else
                        return nothing
                    end
                end   
            end 
            if !new && haskey(Main.__iris_dmem_map, pointer(data))
                return Main.__iris_dmem_map[pointer(data)]
            else
                if create
                    d_mem = iris_data_mem(data)
                    Main.__iris_dmem_map[pointer(data)] = d_mem
                    return d_mem
                else
                    return nothing
                end
            end
        end 
    end

    function flush(task::Union{IrisTask,IrisTaskMem}, mem::Union{IrisMem,IrisTaskMem})::Int32
        return iris_task_dmem_flush_out(field_task(task), field_dmem(mem))
    end

    function flush(task::Union{IrisTask,IrisTaskMem}, host::AbstractArray{T})::Int32 where T
        mem = get_dmem_from_cache(host)
        return iris_task_dmem_flush_out(field_task(task), mem)
    end

    function flush(mem; task=nothing, submit=true, wait=true, dependencies=[], policy=IrisHRT.iris_roundrobin, auto_dependency=true)::IrisTask
        task0 = task
        if task0 == nothing
            task0 = iris_task_create_struct()
        else
            task0 = field_task(task)
        end
        set_name(task0, "flush")
        mem = get_dmem_from_cache(mem)
        #println(Core.stdout, "flushing now: ", typeof(mem))
        iris_task_dmem_flush_out(task0, mem)
        dependencies = [x for x in dependencies if x !== nothing && x != task0]
        if length(dependencies) > 0
            IrisHRT.iris_task_depend(task0, dependencies)
        end
        #println(Core.stdout, "n deps:", dependencies, " task:", task0)
        if auto_dependency
            add_auto_dependencies(task0, [mem], [])
        end
        #println(Core.stdout, "added depe")
        if submit
            #println(Core.stdout, "submits")
            IrisHRT.iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        else
            IrisHRT.iris_task_set_policy(task0, Int32(policy))
        end
        #println(Core.stdout, "returning task")
        return task0
    end
    function release(array_obj::Array{T}) where T 
        dmem_finalizer(array_obj)
    end

    function dmem_finalizer(array_obj)
        p_array = pointer(array_obj)
        if haskey(Main.__iris_dmem_map, p_array)
            #println(Core.stdout, ">>>>>>> releasing memory: $p_array <<<<<<<<<<<")
            IrisHRT.iris_mem_release(Main.__iris_dmem_map[p_array])
            delete!(Main.__iris_dmem_map, p_array)       
        end
    end

    function dim(mem::UMem)
        return iris_dmem_get_dim(field_dmem(mem))
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
        iris_task_enable_julia_interface(task, IrisHRT.iris_julia_native)
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
        iris_task_kernel(task, kernel, dim, off_c, gws_c, lws_c, nparams, c_params, pointer(params_info))
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
        iris_task_kernel(task, kernel, dim, off_c, gws_c, lws_c, nparams, c_params, pointer(params_info))
        return task
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

    function async_submit(task::IrisTask)::Int32
        return iris_task_submit(task, iris_default, Ptr{Int8}(C_NULL), 0)
    end

    function wait(task::IrisTask)::Int32
        return iris_task_wait(task)
    end

    function wait(tasks::Vector{IrisTask})::Int32
        return iris_task_wait_all(Int32(length(tasks)), pointer(tasks))
    end
    
    function wait(mem::Union{IrisMem, IrisTaskMem})::Int32
        dmem = field_dmem(mem)
        if haskey(Main.__iris_dmem_2_task, dmem)
            task = Main.__iris_dmem_2_task[dmem]
            return wait(task)
        end
        return IRIS_SUCCESS
    end

    function release(task::IrisTask)::Int32
        return iris_task_release(task)
    end

    function register_pin(host::Array{T}) where T
        ptr = reinterpret(Ptr{Cvoid}, pointer(host))
        return iris_register_pin_memory(ptr, Csize_t(sizeof(host)))
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

    function zeros(mem::Union{IrisMem, IrisTaskMem})
        dmem = field_dmem(mem)
        T = get_julia_type(dmem)
        iris_mem_init_reset_assign(dmem, T(0))
    end

    function fill(::Type{T}, data, dims...) where T
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_assign(dmem, T(data))
        return dmem
    end

    function fill(mem::Union{IrisMem, IrisTaskMem}, data)
        dmem = field_dmem(mem)
        T = get_julia_type(dmem)
        iris_mem_init_reset_assign(dmem, T(data))
    end

    function ones(T, dims...)
        dmem = iris_data_mem(T, dims...)
        iris_mem_init_reset_assign(dmem, T(1))
        return dmem
    end

    function ones(mem::Union{IrisMem, IrisTaskMem})
        dmem = field_dmem(mem)
        T = get_julia_type(dmem)
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
        element_size = sizeof(T)
        host_cptr = C_NULL
        #call_stack()
        #println(Core.stdout, "Type of element: ", T, " dim:", dim, " Size:", host_size, " Element size:", element_size)
        element_type = get_iris_type(T)
        iris_mem = nothing
        if dev_size != nothing 
            if length(offset) == 0
                offset = Base.zeros(Csize_t, dim)
            end
            GC.@preserve offset dim_size_v dev_size begin 
            iris_mem = iris_data_mem_create_tile_struct_with_type(host_cptr, pointer(offset), pointer(dim_size_v), pointer(dev_size), element_size, dim, element_type)
            end
        else
            GC.@preserve dim_size_v begin 
            iris_mem = iris_data_mem_create_struct_nd(host_cptr, dim_size_v, dim, element_size, element_type)
            end
        end
        if isstructtype(T) 
            push_dmem_custom_type(iris_mem.uid, T)
        end
        return iris_mem
    end

    function iris_data_mem(host::AbstractArray{T}) where T 
        #size = Csize_t(length(host) * sizeof(T))
        host_size = collect(size(host))
        dim = length(host_size)
        element_size = sizeof(T)
        host_cptr = reinterpret(Ptr{Cvoid}, pointer(host))
        host_root = host_cptr
        #println(Core.stdout, "Type of element: ", T, " Size:", size(host), " host size:", host_size, " host size type:", typeof(host_size), " Element size:", element_size, " pointer:", pointer(host))
        element_type = get_iris_type(T, iris_pointer)
        iris_mem = nothing
        if !isa(host, SubArray)
            #println(Core.stdout, "DMEM: ", host_cptr, ", ", size(host), ", ", element_size)
            GC.@preserve host_size begin
            iris_mem = iris_data_mem_create_struct_nd(host_cptr, pointer(host_size), dim, element_size, element_type)
            end
        else
            offset = view_offsets_elements(host)
            dev_size = view_device_size(host)
            host_cptr = reinterpret(Ptr{Cvoid}, pointer(host.parent))
            host_size = collect(size(host.parent))
            dim = length(host_size)
            #println(Core.stdout, "DMEM: ", host_root, ", ", size(host), ", ", element_size, ", ", host_root, ", ", dev_size)
            #println(Core.stdout, " offset:", offset, " dev_size:", dev_size, " host_size:", host_size)
            GC.@preserve host_size offset dev_size begin
            iris_mem = iris_data_mem_create_tile_struct_with_type(host_cptr, pointer(offset), pointer(host_size), pointer(dev_size), element_size, dim, element_type)
            end
        end
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
        #println(Core.stdout, "DMEM: ", host_root, ", ", size(host), ", ", element_size, ", ", host_root, ", ", dev_size)
        #println(Core.stdout, "Type of element: ", T, " Size:", size(host), " Element size:", element_size)
        element_type = get_iris_type(T, iris_pointer)
        if length(offset) == 0
            offset = Base.zeros(Csize_t, dim)
        end
        iris_mem = nothing
        GC.@preserve offset host_size dev_size begin 
        iris_mem = iris_data_mem_create_tile_struct_with_type(host_cptr, pointer(offset), pointer(host_size), pointer(dev_size), element_size, dim, element_type)
        end
        if isstructtype(T) 
            push_dmem_custom_type(iris_mem.uid, T)
        end
        return iris_mem
    end

    function dmem(::Type{T}, dims...) where T
        return iris_data_mem(T, dims...)
    end

    function get_parent(host::AbstractArray{T}) where T
        if isa(host, SubArray) 
            return host.parent
        else
            return nothing
        end
    end

    function dmem(host::AbstractArray{T}; new=false, name="") where T
        if new
            #cached_dmem = get_dmem_from_cache(host, false, false)
            #if cached_dmem != nothing
            #    flag = size(cached_dmem) != size(host)
            #    println(Core.stdout, "DMEM: ", [pointer(host), host_ptr(cached_dmem), host_size(cached_dmem), dev_size(cached_dmem), offsets(cached_dmem), get_parent(host), view_offsets_elements(host), size(host), flag])
            #    if flag
            #        call_stack()
            #    end
            #end
            return iris_data_mem(host)
        else
            return get_dmem_from_cache(host; name=name)
        end
    end

    function print_dmem(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
        iris_data_mem_print(field_dmem(mem))
    end

    function print_dmem(host::AbstractArray{T}) where T
        mem = get_dmem_from_cache(host)
        iris_data_mem_print(mem)
    end

    function enable_regions(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
        return iris_data_mem_enable_outer_dim_regions(field_dmem(mem))
    end

    function regions_count(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
        return iris_data_mem_n_regions(field_dmem(mem))
    end

    function regions(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
        mem_list = []
        nregions = regions_count(mem)
        for i in 1:nregions
            rmem = iris_data_mem_create_region_struct(mem, i-1)
            push!(mem_list, rmem)
        end
        return mem_list
    end

    function region(mem::Union{IrisMem, IrisTaskMem}, i::Int)
        return iris_data_mem_create_region_struct(mem, i-1)
    end

    function array(::Type{T}, dims...) where T
        return dmem(T, dims...)
    end

    function array(host::AbstractArray{T}; new=false) where T
        return dmem(host)
    end

    function dmem_offset(host, dev_size, offset=[]) 
        return iris_data_mem(host, dev_size, offset)
    end

    function reshape(bmem::Union{IrisMem, IrisTaskMem}, dims...) 
        mem = field_dmem(bmem)
        host_cptr = iris_get_dmem_host(mem)
        host_cptr = reinterpret(Ptr{Cvoid}, host_cptr)
        elem_size = element_size(mem) 
        elem_type = element_type(mem)
        dim_size = dims
        #println(Core.stdout, "Type of element: ", " dim:", dims)
        if length(dims) == 1 && isa(dims[1], Tuple)
            dim_size = dims[1]
        end
        host_size = prod(dim_size) * elem_size
        dim_size_v = collect(dim_size)
        key = (mem.uid, dim_size_v, elem_size, elem_type)
        if haskey(Main.__iris_dmem_mirrors_map, key)
            return Main.__iris_dmem_mirrors_map[key]
        end 
        dim = length(dim_size)
        iris_mem=nothing
        GC.@preserve dim_size_v begin 
            iris_mem = iris_data_mem_create_struct_nd(host_cptr, pointer(dim_size_v), dim, elem_size, elem_type)
        end
        #println(Core.stdout, "Type of element: ", " HostSize:", host_size, " Dims:", dim_size_v, " N-dims:", dim, " Element size:", elem_size, " pointer:", host_cptr, " uid:", iris_mem.uid)
        #print_dmem(iris_mem)
        if elem_type == iris_custom_type 
            T = get_dmem_custom_type(mem)
            push_dmem_custom_type(iris_mem.uid, T)
        end
        iris_dmem_set_source(iris_mem, mem)
        Main.__iris_dmem_mirrors_map[key] = iris_mem
        #print_dmem(iris_mem)
        return iris_mem
    end

    function reshape(host::AbstractArray{T}, dims...) where T
        source_mem = get_dmem_from_cache(host)
        return reshape(source_mem, dims...)
    end

    isdivisible(m, n) = iszero(m % n)
    function Base.reinterpret(::Type{T}, bmem::Union{IrisMem, IrisTaskMem}) where T
        mem = field_dmem(bmem)
        host_cptr = iris_get_dmem_host(mem)
        host_cptr = reinterpret(Ptr{Cvoid}, host_cptr)
        elem_size = element_size(mem) 
        iris_elem_type = element_type(mem)
        elem_type = get_julia_type(mem)
        dst_elem_type = T
        dst_elem_size = sizeof(T)
        #println(Core.stdout, " elem_size: ", elem_size, " dst_elem_size:", dst_elem_size)
        if !isdivisible(elem_size, dst_elem_size) && !isdivisible(dst_elem_size, elem_size)
            error("ArgumentError: cannot reinterpret an `$elem_type` array to `$T`")
        end
        hsize = collect(dev_size(mem))
        dst_host_size = hsize
        if !isdivisible(hsize[1] * elem_size, dst_elem_size)
            error("ArgumentError: cannot reinterpret an `$elem_type` array to `$T` for the source array of size: $hsize")
        end
        #println(Core.stdout, "Type of element: ", " HostSize:", dst_host_size, " Element size:", dst_elem_size, " pointer:", host_cptr)

        
        factor = elem_size / dst_elem_size
        dst_elem_type = get_iris_type(T)
        hT = eltype(dst_host_size)
        #println(Core.stdout, "hsize:", hsize, " dst_host_size:", dst_host_size, " hT:", hT)
        dst_host_size[1] = hT(dst_host_size[1] * factor)
        key = (mem.uid, dst_host_size, dst_elem_size, dst_elem_type)
        if haskey(Main.__iris_dmem_mirrors_map, key)
            return Main.__iris_dmem_mirrors_map[key]
        end 
        dim = length(dst_host_size)

        iris_mem=nothing
        GC.@preserve dst_host_size begin 
            iris_mem = iris_data_mem_create_struct_nd(host_cptr, pointer(dst_host_size), dim, dst_elem_size, dst_elem_type)
        end
        #println(Core.stdout, "Type of element: ", " HostSize:", dst_host_size, " N-dims:", dim, " Element size:", dst_elem_size, " pointer:", host_cptr, " hT:", hT, " uid:", iris_mem.uid)
        if dst_elem_type == iris_custom_type 
            push_dmem_custom_type(iris_mem.uid, T)
        end
        iris_dmem_set_source(iris_mem, mem)
        Main.__iris_dmem_mirrors_map[key] = iris_mem
        return iris_mem
    end

    function reinterpret_host(::Type{T1}, host::AbstractArray{T2}) where {T1, T2}
        source_mem = get_dmem_from_cache(host)
        return reinterpret(T1, source_mem)
    end

    function host_ptr(mem::Union{IrisMem, IrisTaskMem})
        return iris_get_dmem_host(field_dmem(mem))
    end

    function host_ptr(host::AbstractArray{T}) where T
        source_mem = get_dmem_from_cache(host)
        return host_ptr(source_mem)
    end

    function uid(mem::Union{IrisMem, IrisTaskMem})
        return iris_mem_get_uid(field_dmem(mem))
    end

    function uid(host::AbstractArray{T}) where T
        source_mem = get_dmem_from_cache(host)
        return uid(source_mem)
    end

    function host(bmem::Union{IrisMem, IrisTaskMem}; flush=false)
        host, dsize = get_host_4_dmem(bmem)
        return host
    end

    function host(host::AbstractArray{T}; flush=false) where T
        source_mem = get_dmem_from_cache(host)
        return host(source_mem)
    end

    function valid_host_root(bmem::Union{IrisMem, IrisTaskMem}; flush=false)
        mem = field_dmem(bmem)
        arg_ptr = iris_get_dmem_valid_host(mem)
        if arg_ptr == C_NULL
            return nothing
        end
        if flush
            IrisHRT.flush(mem)
        end
        size = host_size(mem)
        j_ptr = ptr_reinterpret(mem.uid, 0, 1, arg_ptr, get_type(mem))
        return unsafe_wrap(Array, j_ptr, size, own=false)
    end

    function valid_host(bmem::Union{IrisMem, IrisTaskMem}; flush=false)
        host, dsize = get_host_4_dmem(bmem)
        return host
    end

    function valid_host(host::AbstractArray{T}; flush=false) where T
        source_mem = get_dmem_from_cache(host)
        return valid_host(source_mem)
    end


    function get_type(mem::Union{IrisMem, IrisTaskMem})
        return iris_get_mem_element_type(field_dmem(mem))
    end

    function get_julia_type(mem::Union{IrisMem, IrisTaskMem})
        element_type = iris_get_julia_type(field_dmem(mem))
        return element_type
    end
    
    function host_ndim(mem::Union{IrisMem, IrisTaskMem})
        return iris_dmem_get_host_size_dim(field_dmem(mem))
    end

    function ndim(mem::Union{IrisMem, IrisTaskMem})
        return iris_dmem_get_dim(field_dmem(mem))
    end

    function host_size(bmem::Union{IrisMem, IrisTaskMem}, trail=true)
        mem = field_dmem(bmem)
        size_ptr = iris_dmem_get_host_size(mem)
        dim = iris_dmem_get_host_size_dim(mem)
        dims_array = unsafe_wrap(Array, size_ptr, dim)
        if trail
            tr_size = remove_trailing_ones2(Int.(dims_array))
        end
        return Tuple(tr_size)
    end

    function dev_size(bmem::Union{IrisMem, IrisTaskMem})
        mem = field_dmem(bmem)
        size_ptr = iris_dmem_get_dev_size(mem)
        dim = iris_dmem_get_dim(mem)
        dims_array = unsafe_wrap(Array, size_ptr, dim)
        tr_size = remove_trailing_ones2(Int.(dims_array))
        return Tuple(tr_size)
    end

    function offsets(bmem::Union{IrisMem, IrisTaskMem})
        mem = field_dmem(bmem)
        size_ptr = iris_dmem_get_offsets_from_host(mem)
        dim = iris_dmem_get_dim(mem)
        dims_array = unsafe_wrap(Array, size_ptr, dim)
        return Tuple(Int.(dims_array))
    end

    function element_type(mem::Union{IrisMem, IrisTaskMem})
        return iris_dmem_get_elem_type(field_dmem(mem))
    end

    function element_size(mem::Union{IrisMem, IrisTaskMem})
        return iris_dmem_get_elem_size(field_dmem(mem))
    end

    function enable_pin()
        return iris_data_mem_set_pin_flag(true)
    end

    function disable_pin()
        return iris_data_mem_set_pin_flag(false)
    end

    function copy_host(mem::Union{IrisMem, IrisTaskMem}, host::AbstractArray)
        host_ptr = reinterpret(Ptr{Cvoid}, pointer(host))
        return iris_data_mem_copyto_host(field_dmem(mem), host_ptr)
    end

    function refresh(mem::Union{IrisMem, IrisTaskMem})
        return iris_data_mem_refresh(field_dmem(mem))
    end

    function get_source(mem::Union{IrisMem, IrisTaskMem})
        return iris_dmem_get_source(field_dmem(mem))
    end

    function update_host_size(mem::Union{IrisMem, IrisTaskMem}, host_size)
        vec = collect(Csize_t, host_size)
        GC.@preserve vec begin 
        iris_data_mem_update_host_size(field_dmem(mem), pointer(vec))
        end
    end

    function update_dev_size(mem::Union{IrisMem, IrisTaskMem}, dev_size)
        vec = collect(Csize_t, dev_size)
        GC.@preserve vec begin 
        iris_data_mem_update_dev_size(field_dmem(mem), pointer(vec))
        end
    end

    function update_off(mem::Union{IrisMem, IrisTaskMem}, off)
        vec = collect(Csize_t, off)
        GC.@preserve vec begin 
        iris_data_mem_update_off(field_dmem(mem), pointer(vec))
        end
    end

    function release(mem::Union{IrisMem, IrisTaskMem})::Int32
        host_cptr = host_ptr(mem)
        if host_cptr == C_NULL
            return iris_mem_release(field_dmem(mem))
        else
            if haskey(Main.__iris_dmem_map, host_cptr)
                #println(Core.stdout, ">>>>>>> releasing memory: $host_ptr <<<<<<<<<<<")
                status = IrisHRT.iris_mem_release(Main.__iris_dmem_map[host_cptr])
                delete!(Main.__iris_dmem_map, host_cptr)
                return status
            else
                return iris_mem_release(field_dmem(mem))
            end
        end
    end

    function graph()
        return iris_graph_create_empty()
    end

    function add_task(graph::IrisGraph, task::IrisTask, policy=IrisHRT.iris_roundrobin)
        return iris_graph_task(graph, task, policy, C_NULL)
    end

    function release(graph::IrisGraph)::Int32
        return iris_graph_release(graph)
    end

    function wait(graph::IrisGraph)::Int32
        return iris_graph_wait(graph)
    end

    function wait(graphs::Vector{IrisGraph})::Int32
        return iris_graph_wait_all(Int32(length(graphs)), pointer(graphs))
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
        for (key, value) in Main.__iris_dmem_subarray_map
            IrisHRT.iris_mem_release(value)
        end
        for (key, value) in Main.__iris_dmem_mirrors_map
            IrisHRT.iris_mem_release(value)
        end
        # Clear all elements in the dictionary
        empty!(Main.__iris_dmem_2_task)
        empty!(Main.__iris_dmem_map)
        empty!(Main.__iris_dmem_subarray_map)
        empty!(Main.__iris_dmem_mirrors_map)
        empty!(Main.__iris_dmem_parallel_reduce_map)
        empty!(Main.__iris_dmem_custom_type)
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
            update_host_size(mem, size(larray))
        end
    end

    function remove_trailing_ones2(arr::Array{T}) where T
        idx = findlast(x -> x != 1, arr)
        return idx === nothing ? Int[1] : arr[1:idx]
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
            dmem0 = get_dmem_from_cache(host)
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

    function fill_task(::Type{T}, data, dims...; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(T, dims..., task=task, dmem=dmem) 
        iris_task_cmd_init_reset_assign(task0, dmem0, T(data))
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

    function fill_task(host::Array{T}, data; task=nothing, dmem=nothing, submit=true, policy=IrisHRT.iris_roundrobin, wait=true) where T
        (task0, dmem0) = get_init_task_dmem(host, dmem=dmem, task=task) 
        iris_task_cmd_init_reset_assign(task0, dmem0, T(data))
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


    function find_shared_vars(julia_kernel_type, gws, kernel, args...)
        #println(Core.stdout, " all args:", args, " legnth:", length(args))
        m, offset = get_method(julia_kernel_type, gws, kernel, args...)
        if m == nothing
            return (in=[], out=[])
        end
        extra_args = offset - length(gws)
        #println(Core.stdout, " m = ", m, " offset:", offset, " gws:", gws, " length:", length(gws))
        # pull out the root Expr from whatever uncompressed_ast gives us
        ast = Base.uncompressed_ast(m)
        #println(Core.stdout, m)
        #println(Core.stdout, ast)
        root = ast
        found = Set{Any}()
        # recursive descent
        all_outs = []
        function parse_expr(code)
            #println("Parsing Expression offset:", offset, " extra_args:", extra_args)
            required_expr = false
            for (i, item) in enumerate(code)
                #println("Expression $(i):", item, " required:", required_expr)
                #println("  ", item)
                push!(all_outs, item)
                node = item
                # Process only expressions (Expr objects) that are assignments.
                if node isa Expr && required_expr && node.head == :(=)
                    index = Int(node.args[2].args[2].id)-2*length(gws) + offset - 1
                    #println(" 222Expr: ", node, " head:", node.head, " args: ", node.args, " index:", index, " length:", length(args))
                    if index >=1  && index <= length(args)
                        esize = 1
                        data = args[index]
                        if data isa Union{IrisTaskMem, IrisMem}
                            esize = element_size(data)
                        else
                            esize = sizeof(eltype(data))
                        end
                        push!(found, length(data) * esize)
                        #println("---- Node:", node, " args:", node.args, " offset:", (offset, extra_args), " kernel:", kernel, " index:", index, " esize:", esize)
                    end
                    required_expr = false
                elseif node isa Expr && node.head == :call
                    #println(" Expr: ", node, " head:", node.head, " args: ", node.args)
                    if length(node.args)>=3 && node.args[3] === :shared
                        required_expr = true
                    elseif length(node.args)>=3 && node.args[3] === :(:shared)
                        required_expr = true
                    elseif required_expr
                        index = Int(node.args[2].id) -2*length(gws) + offset
                        if index >=1  && index <= length(args)
                            esize = 1
                            data = args[index]
                            if data isa Union{IrisTaskMem, IrisMem}
                                esize = element_size(data)
                            else
                                esize = sizeof(eltype(data))
                            end
                            push!(found, length(data) * esize)
                            #println("---- Node:", node, " args:", node.args, " offset:", offset, " kernel:", kernel, " index:", index, " esize:", esize)
                        end
                        required_expr = false
                    end
                elseif node isa Expr && node.head == :block
                    parse_expr(node.args)           
                end
            end
        end
        parse_expr(root.code)
        #println("Completed Parsing Expression ", found)
        if length(found) > 0
            return sum(collect(found))
        else
            return 0
        end
    end
    function get_method(julia_kernel_type, gws, kernel, args...)
        f = nothing 
        if (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native && (f=getFunction(kernel*"_cuda")) != nothing
            nothing
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native && (f=getFunction(kernel*"_hip")) != nothing
            nothing
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native && (f=getFunction(kernel*"_openmp")) != nothing
            nothing
        elseif julia_kernel_type == IrisHRT.iris_julia_kernel_abstraction && (f=getFunction(kernel)) != nothing
            nothing
        elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_jacc && (f=getFunction(kernel)) != nothing
            nothing
        else
            error("Couldn't find function definition for kernel: ", kernel)
            return nothing, 0 #(in=[], out=[])
        end
        if f == nothing
            return nothing, 0 #(in=[], out=[])
        end
        m = nothing
        offset = 0
        function convert_iris_mem(mem)
            d = ndim(mem)
            if d == 1
                return IrisMem1D{eltype(mem)}(mem)
            elseif d == 2
                return IrisMem2D{eltype(mem)}(mem)
            elseif d == 3
                return IrisMem3D{eltype(mem)}(mem)
            elseif d == 4
                return IrisMem4D{eltype(mem)}(mem)
            else
                println(Core.stdout, "[Warning] Unhandled dim:", d," kernel:", kernel, " for convert_iris_mem")
                return mem
            end
        end
        conv_args = []
        for each_arg in args
            if each_arg isa UMem
                push!(conv_args, convert_iris_mem(field_dmem(each_arg)))
            else
                push!(conv_args, each_arg)
            end
        end 
        #println(Core.stdout, "Args: ", conv_args)
        conv_args = Tuple(conv_args)
        if (julia_kernel_type & 0xFFFF) == iris_julia_jacc
            i = 0 
            j = 0
            k = 0
            offset = length(gws)
            #println(Core.stdout, "Args: ", conv_args...)
            #println(Core.stdout, "f: ", f)
            if julia_kernel_type == iris_julia_jacc_parallel_for
                if offset == 1
                    m = @which f(i, conv_args...)   # Get method instance
                elseif offset == 2
                    m = @which f(i, j, conv_args...)   # Get method instance
                elseif offset == 3
                    m = @which f(i, j, k, conv_args...)   # Get method instance
                else
                    error("Couldn't find JACC function for gws:", gws, " for kernel:", kernel)
                end
            else # iris_julia_jacc_parallel_reduce
                if offset == 1
                    m = @which f(i, conv_args[5:end]...)   # Get method instance
                elseif offset == 2
                    m = @which f(i, j, conv_args[5:end]...)   # Get method instance
                elseif offset == 3
                    m = @which f(i, j, k, conv_args[5:end]...)   # Get method instance
                else
                    error("Couldn't find JACC function for gws:", gws, " for kernel:", kernel)
                end
                offset += 4
            end
        else
            m = @which f(conv_args...)   # Get method instance
        end
        #println(Core.stdout, "JACC function for kernel:", kernel, " for offset:", offset)
        return m, offset
    end
    function identify_in_out(julia_kernel_type, gws, kernel, args...)
        m, offset = get_method(julia_kernel_type, gws, kernel, args...)
        if m == nothing
            return (in=[], out=[])
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
                            index = Int(slot_index) -2*length(gws) + offset - 1
                            original_var = ci.slotnames[slot_index]
                            o_original_var = getfield(ci, :slotnames)[slot_index]
                            if fname == :getindex
                                push!(vector_in, index)
                            end
                            if fname == :setindex!
                                push!(vector_out, index)
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
        a_in = []
        a_out = []
        a_in_types = []
        a_out_types = []
        for i in vector_in
            push!(a_in, args[i])
            push!(a_in_types, typeof(args[i]))
        end
        for i in vector_out
            push!(a_out, args[i])
            push!(a_out_types, typeof(args[i]))
        end
        #println("-----Automatic f:", kernel, " in:", vector_in, " and out:", vector_out, " n_args:", length(args))
        #println("-----Automatic f:", kernel, " in:", a_in_types, " and out:", a_out_types, " n_args:", length(args))
        return (in=a_in, out=a_out)
    end

    function parallel_for((L, M, N)::Union{Tuple{Int64, Int64, Int64},Tuple{UInt64, UInt64, UInt64}}, f::Union{Function,String}, x... ; eargs...)
        return IrisHRT.task(;gws=(L,M,N), parallel_for=true, kernel=f, args=Any[x...], eargs...)
    end

    function parallel_for((M, N)::Union{Tuple{Int64, Int64},Tuple{UInt64, UInt64}}, f::Union{Function,String}, x... ; eargs...)
        return IrisHRT.task(;gws=(M,N), parallel_for=true, kernel=f, args=Any[x...], eargs...) 
    end

    function parallel_for(N::Union{Int64,UInt64}, f::Union{Function,String}, x... ; eargs...)
        return IrisHRT.task(;gws=N, parallel_for=true, kernel=f, args=Any[x...], eargs...) 
    end

    function parallel_reduce_core(N::Union{Int64, UInt64, Tuple{Int64}, Tuple{UInt64}, NTuple{2, Integer}, NTuple{3, Integer}}, f::Union{Function,String}, x... ; result=nothing, buffer=nothing, task=nothing, submit=true, flush=false, policy=IrisHRT.iris_roundrobin, wait=true, auto_dependency=true, dependencies=[], metadata=nothing, op=nothing, init=nothing)
        kernel_name = f
        if isa(f, Function) 
            kernel_name = String(Symbol(f))
        end
        d_flush=[]
        d_out=[]
        d_in = []
        out_mem = result
        tmp_mem = buffer
        first_x = nothing
        for arg in x
            if isa(arg, Union{AbstractArray, IrisMem, IrisTaskMem})
                mem = get_dmem_from_cache(arg)
                first_x = mem
                break
            end
        end
        if out_mem == nothing
            out_mem = dmem(IrisHRT.get_julia_type(first_x), 1)
        else
            out_mem = get_dmem_from_cache(out_mem)
        end
        map_key = (first_x.uid, out_mem.uid)
        if tmp_mem == nothing
            if !haskey(Main.__iris_dmem_parallel_reduce_map, map_key)
                tmp_mem = dmem(IrisHRT.get_julia_type(first_x), dev_size(first_x))
                Main.__iris_dmem_parallel_reduce_map[map_key] = (tmp_mem, out_mem)
            else
                tmp_mem, prev_mem = Main.__iris_dmem_parallel_reduce_map[map_key]
            end
        else
            tmp_mem = field_dmem(tmp_mem)
        end
        if flush
            push!(d_flush, out_mem)
        else
            push!(d_out, out_mem)
        end
        push!(d_out, tmp_mem)
        for arg in x[1:end]  #x[2:end]
            if isa(arg, Union{AbstractArray, IrisMem, IrisTaskMem})
                mem = get_dmem_from_cache(arg)
                push!(d_in, mem)
            end
        end
        push!(d_in, tmp_mem)
        push!(d_in, out_mem)
        if init == nothing
            init = get_jacc_init(op, out_mem)
        end
        IrisHRT.fill(out_mem, init)
        IrisHRT.refresh(out_mem)
        IrisHRT.fill(tmp_mem, init)
        IrisHRT.refresh(tmp_mem)
        iris_op = convert_jacc_op_2_iris_op(op)
        #println(Core.stdout, "Creating task: gws:", N, " iris_op:", iris_op, " func:", f, " init:", init, " out_mem.uid:", out_mem.uid, " args=", x...)
        task0 = IrisHRT.task(gws=N, task=task, parallel_reduce=true, 
                    in=d_in, out=d_out, submit=submit, policy=policy, 
                    wait=wait, flush=d_flush, 
                    auto_dependency=auto_dependency, 
                    dependencies=dependencies, metadata=metadata, 
                    kernel=kernel_name, 
                    args=Any[iris_op, init, out_mem, tmp_mem, x...])
        #println(Core.stdout, " parallel_for:", task0)
        return IrisTaskMem(task0, out_mem)
    end

    function get_iris_jacc_elem_access(op)
        if op == +
            return "__iris_jacc_add"
        elseif op == *
            return "__iris_jacc_mul"
        elseif op == -
            return "__iris_jacc_minus"
        elseif op == /
            return "__iris_jacc_div"
        elseif op == sum
            return "__iris_jacc_add"
        elseif op == prod
            return "__iris_jacc_mul"
        elseif op == min
            return "__iris_jacc_min"
        elseif op == max
            return "__iris_jacc_max"
        end
    end
    function parallel_reduce(N::Integer, op::Callable, f::Union{Callable,String}, x...; eargs...)
        return parallel_reduce_core(N, f, x...; op=op, eargs...)
    end
    function parallel_reduce(N::Integer, f::Union{Callable,String}, x...; eargs...)
        return parallel_reduce_core(N, f, x...; eargs...)
    end
    function parallel_reduce(
            (M, N)::NTuple{2, Integer}, op::Callable, f::Union{Callable,String}, x...;
            eargs...)
        return parallel_reduce_core((M, N), f, x...; op=op, init=init)
    end
    function parallel_reduce(
            (M, N)::NTuple{2, Integer}, f::Union{Callable,String}, x...;
            eargs...)
        return parallel_reduce_core((M, N), f, x...; init=init)
    end
    function parallel_reduce(op::Base.Callable, A::AbstractArray; args...)
        s = size(A)
        kernel = get_iris_jacc_elem_access(op)
        if length(s) == 1
            return parallel_reduce_core(s[1], kernel, A; op=op, args...)
        else
            return parallel_reduce_core(s, kernel, A; op=op, args...)
        end
    end
    function parallel_reduce(op::Base.Callable, A::AbstractVector; args...)
        kernel = get_iris_jacc_elem_access(op)
        return parallel_reduce_core(length(A), kernel, A; op=op, args...)
    end
    function parallel_reduce(op::Base.Callable, A::AbstractMatrix; args...)
        kernel = get_iris_jacc_elem_access(op)
        return parallel_reduce_core(size(A), kernel, A; op=op, args...)
    end
    function parallel_reduce(op::Base.Callable, A::Union{IrisMem,IrisTaskMem}; args...)
        kernel = get_iris_jacc_elem_access(op)
        return parallel_reduce_core(dev_size(A), kernel, A; op=op, args...)
    end
    function parallel_reduce(A::AbstractArray; args...)
        s = size(A)
        kernel = get_iris_jacc_elem_access(+)
        if length(s) == 1
            return parallel_reduce_core(s[1], kernel, A; op=+, args...)
        else
            return parallel_reduce_core(s, kernel, A; op=+, args...)
        end
    end
    function parallel_reduce(A::Union{IrisMem,IrisTaskMem}; args...)
        kernel = get_iris_jacc_elem_access(+)
        return parallel_reduce_core(dev_size(A), kernel, A; op=+, args...)
    end
    function parallel_reduce(A::AbstractMatrix; args...)
        kernel = get_iris_jacc_elem_access(+)
        return parallel_reduce_core(size(A), kernel, A; op=+, args...)
    end
    function parallel_reduce(A::AbstractVector; args...)
        kernel = get_iris_jacc_elem_access(+)
        return parallel_reduce_core(length(A), kernel, A; op=+, args...)
    end

    function Base.sum(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
        return parallel_reduce(+, IrisHRT.field_dmem(x); eargs...)
    end
    function Base.prod(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
        return IrisHRT.parallel_reduce(*, IrisHRT.field_dmem(x); eargs...)
    end
    function Base.maximum(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
        return IrisHRT.parallel_reduce(max, IrisHRT.field_dmem(x); eargs...)
    end
    function Base.minimum(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
        return IrisHRT.parallel_reduce(max, IrisHRT.field_dmem(x); eargs...)
    end
    function __iris_dot_v_v(i, X, Y)
        @inbounds X[i] * Y[i]
    end
    function __iris_dot_v_s(i, X, y)
        @inbounds X[i] * y
    end
    function __iris_dot_s_v(i, x, Y)
        @inbounds x * Y[i]
    end
    function parallel_reduce_size(x)
        s = size(x)
        if length(s) == 1
            return s[1]
        end
        return s
    end
    function inner(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}, y::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}; eargs...)
        N = parallel_reduce_size(x)
        x = get_dmem_from_cache(x)
        y = get_dmem_from_cache(y)
        return parallel_reduce(N, +, "IrisHRT:__iris_dot_v_v", x, y; eargs...)
    end
    function inner(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}, y::T; eargs...) where T <: Number
        N = parallel_reduce_size(x)
        x = get_dmem_from_cache(x)
        return parallel_reduce(N, +, "IrisHRT:__iris_dot_v_s", x, y; eargs...)
    end
    function inner(x::T, y::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}; eargs...) where T <: Number
        N = parallel_reduce_size(y)
        y = get_dmem_from_cache(y)
        return parallel_reduce(N, +, "IrisHRT:__iris_dot_s_v", x, y; eargs...)
    end
    function dot(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}, y::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}; eargs...)
        N = parallel_reduce_size(x)
        x = get_dmem_from_cache(x)
        y = get_dmem_from_cache(y)
        return parallel_reduce(N, +, "IrisHRT:__iris_dot_v_v", x, y; eargs...)
    end
    function dot(x::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}, y::T; eargs...) where T <: Number
        N = parallel_reduce_size(x)
        x = get_dmem_from_cache(x)
        return parallel_reduce(N, +, "IrisHRT:__iris_dot_v_s", x, y; eargs...)
    end
    function dot(x::T, y::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem,AbstractArray,AbstractVector}; eargs...) where T <: Number
        N = parallel_reduce_size(y)
        y = get_dmem_from_cache(y)
        return parallel_reduce(N, +, "IrisHRT:__iris_dot_s_v", x, y; eargs...)
    end
    function sum_data(x; eargs...)
        return parallel_reduce(+, x; eargs...)
    end
    function prod_data(x; eargs...)
        return parallel_reduce(*, x; eargs...)
    end
    function maximum_data(x; eargs...)
        return parallel_reduce(max, x; eargs...)
    end
    function minimum_data(x; eargs...)
        return parallel_reduce(min, x; eargs...)
    end

    function custom_print(im::IrisMem)
        #println(Core.stdout, " flushed now!")
        host, dsize = get_host_4_dmem(im)
        return host #"IrisMem with data: " * join(host, ", ")
    end

    # Overriding show so that println calls custom_print.
    function Base.show(io::IO, im::IrisMem)
        print(io, custom_print(im))
    end

    function Base.show(io::IO, im::IrisTaskMem)
        #println(Core.stdout, " called show!")
        print(io, custom_print(im.dmem))
    end

    function adjust_tuple(data, l, default_value=0)
        n_missing = l - length(data)
        if n_missing > 0
            default = (default_value,) * n_missing
            jjj
        else
            return data
        end
    end

    function adjust_tuple(t::Tuple, n::Int; pad=0)
        k = length(t)
        if k >= n
            # Truncate the tuple to the first n elements
            return t[1:n]
        else
            # Pad the tuple with (n - k) copies of pad
            return (t..., ntuple(_ -> pad, n - k)...)
        end
    end

    function get_tile(host, offset, tile_size)
        # Build index ranges for each dimension.
        # For each dimension, the range starts at offset[i] and ends at offset[i] + tile_size[i] - 1.
        if length(offset) > length(tile_size)
            tile_size = adjust_tuple(tile_size, length(offset), pad=1)
        end
        #println(Core.stdout, " offset:", offset, " tile:", tile_size, " size:", size(host))
        index_ranges = tuple(((offset[i]+1 : (offset[i] + tile_size[i])) for i in 1:length(offset))...)
        #println(Core.stdout, " indexes:", index_ranges)        
        # Slice the host array using the generated ranges.
        tile = @view host[index_ranges...]
        # Print the tile.
        #println("Tile:", tile)
        return tile
    end

    function get_host_4_dmem(mem::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem})
        dmem = field_dmem(mem)
        host = IrisHRT.valid_host_root(dmem; flush=true)
        hsize = IrisHRT.host_size(dmem)
        dsize = IrisHRT.dev_size(dmem)
        offs = IrisHRT.offsets(dmem)
        if hsize != dsize
            p = get_source(dmem)
            #println(Core.stdout, " host:", size(host), " hsize:", hsize, " dsize:", dsize, " offs:", offs, " p-uid:", p.uid, " class:", p.class_obj)
            if p.class_obj != C_NULL
                p_host, p_dsize = get_host_4_dmem(p)
                if p_dsize != dsize
                    host = Base.reshape(p_host, dsize)
                else
                    host = p_host
                end
            else
                host = get_tile(host, offs, dsize)
            end
        end
        return host, dsize
    end

    # Custom getindex: Accepts exactly N indices 
    # (which can be integers, ranges, or other index types)
    function Base.getindex(mem::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, inds::Vararg{Any, N}) where N
        # Add any custom behavior before/after element access if desired
        host, dsize = IrisHRT.get_host_4_dmem(mem)
        return Base.getindex(host, inds...)
    end
    function Base.lastindex(mem::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem})
        # Add any custom behavior before/after element access if desired
        dmem = field_dmem(mem)
        dsize = IrisHRT.host_size(dmem)
        return dsize
    end


    function set_julia_policy(task0, policy)
        if isa(policy, Function)
            policy_ptr = String(Symbol(policy))
            #println("Julia Policy name>>>", policy_ptr)
            IrisHRT.iris_task_set_julia_policy(task0, policy_ptr)
            policy = IrisHRT.iris_julia_policy
        end
        return policy
    end

    function view_offsets_elements(v::AbstractArray{T}) where T
        if !isa(v, SubArray)
            return [0, 0, 0]
        end
        p = parent(v)
        nd = ndims(p)
        out = ntuple(i -> begin
            # For each dimension, if the index is a range, take its first element;
            # if it’s a scalar, use that directly.
            idx = v.indices[i] isa AbstractRange ? first(v.indices[i]) : v.indices[i]
            # Compute the offset in terms of number of elements:
            idx - firstindex(p, i)
        end, nd)
        return collect(out)
    end
    function view_offsets(v::SubArray)
        p = parent(v)
        nd = ndims(p)
        out = ntuple(i -> begin
            # If the index is a range, use its first element; if it’s a scalar, use it directly.
            idx = v.indices[i] isa AbstractRange ? first(v.indices[i]) : v.indices[i]
            # Compute offset relative to the parent’s first index in this dimension,
            # multiplied by the parent’s stride for that dimension.
            (idx - firstindex(p, i)) * stride(p, i)
        end, nd)
        return collect(out)
    end

    function view_device_size(v::SubArray)
        # Determine the number of dimensions in the parent array.
        nd = ndims(parent(v))
        # For each parent's dimension, if the corresponding index in the view is a range,
        # use its length; if it's a scalar, treat its size as 1.
        out = ntuple(i -> v.indices[i] isa AbstractRange ? length(v.indices[i]) : 1, nd)
        return collect(out)
    end

    function copyto!(dst, src; task=nothing, flush=false, wait=true, submit=true, policy=IrisHRT.iris_roundrobin, auto_dependency=true, dependencies=[])
        task0 = task
        if task0 == nothing
            task0 = iris_task_create_struct()
            iris_task_set_name(task0, "copyto")
        else
            task0 = field_task(task)
        end
        if !isa(src, Union{IrisMem, IrisTaskMem})
            src = get_dmem_from_cache(src)
        else
            src = field_dmem(src)
        end
        if !isa(dst, Union{IrisMem, IrisTaskMem})
            dst = get_dmem_from_cache(dst)
        else
            dst = field_dmem(dst)
        end
        n_dependencies = IrisHRT.IrisTask[field_task(x) for x in dependencies if x !== nothing && x != task0]
        if length(n_dependencies) > 0
            IrisHRT.iris_task_depend(task0, n_dependencies)
        end
        if auto_dependency
            add_auto_dependencies(task0, [src], [dst])
        end
        IrisHRT.iris_task_dmem2dmem(task0, src, dst)
        if flush
            IrisHRT.iris_task_dmem_flush_out(task0, dst)
        end
        policy = IrisHRT.set_julia_policy(task0, policy)
        Main.__iris_dmem_2_task[dst] = task0
        if submit
            #println(Core.stdout, "IRIS task submit wait:", wait)
            IrisHRT.iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        else
            IrisHRT.iris_task_set_policy(task0, Int32(policy))
        end
        return task0
    end

    function copy(src; eargs...)
        if !isa(src, Union{IrisMem, IrisTaskMem})
            src = get_dmem_from_cache(src)
        else
            src = field_dmem(src)
        end
        dst = iris_data_mem_copy(src)
        host = valid_host_root(dst)
        Main.__iris_dmem_map[pointer(host)] = dst
        task0 = copyto!(dst, src; eargs...)
        return IrisTaskMem(task0, dst)
    end

    function add_auto_dependencies(task0, in_list, out_list)
        n_dependencies = IrisHRT.IrisTask[]
        for mem in in_list
            if isa(mem, Union{IrisMem, IrisTaskMem})
                mem_obj = IrisHRT.field_dmem(mem)
                if haskey(Main.__iris_dmem_2_task, mem_obj)
                    dep_task = Main.__iris_dmem_2_task[mem_obj]
                    if dep_task.uid != task0.uid
                        push!(n_dependencies, field_task(Main.__iris_dmem_2_task[mem_obj]))
                    end
                end
            else
                mem_obj = get_dmem_from_cache(mem) 
                if haskey(Main.__iris_dmem_2_task, mem_obj)
                    dep_task = Main.__iris_dmem_2_task[mem_obj]
                    if dep_task.uid != task0.uid
                        push!(n_dependencies, field_task(dep_task))
                    end
                end
            end
        end
        for mem in out_list
            if isa(mem, Union{IrisMem, IrisTaskMem})
               Main.__iris_dmem_2_task[field_dmem(mem)] = task0
            else
               mem_obj = get_dmem_from_cache(mem) 
               Main.__iris_dmem_2_task[mem_obj] = task0
            end
        end
        if length(n_dependencies) > 0
            IrisHRT.iris_task_depend(task0, n_dependencies)
            #println(Core.stdout, " dependencies: ", n_dependencies)
        end
    end

    function predef_task(f, x...; eargs...)
        return task(;kernel=f, args=x, predef_task=true, eargs...)
    end

    function task(; in=Any[], out=Any[], flush=Any[], refresh=[], auto_in_out=false, lws=Int64[], gws=Int64[], off=Int64[], policy=IrisHRT.iris_roundrobin, wait=true, core=false, ka=false, parallel_reduce=false, parallel_for=false, host=false, predef_task=false, auto_dependency=true, task=nothing, metadata=nothing, kernel="kernel", args=[], submit=true, dependencies=[])
        call_args = args
        mem_params = Dict{Any, Any}()
        t_args = args
        if !isa(args, Tuple)
            t_args = Tuple(args)
        end
        auto_in = Any[]
        auto_out = Any[]
        if ! auto_in_out && length(in)==0 && length(out)==0 && length(args)!=0
            auto_in_out = true
        end
        kernel_name = kernel
        if kernel isa Symbol
            kernel_name = String(kernel)
        elseif kernel isa Function
            kernel_name = String(Symbol(kernel))
        elseif kernel isa Callable
            kernel_name = String(Symbol(kernel))
        end
        if predef_task == true
            core=true
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
        if auto_in_out && predef_task == false
            (auto_in, auto_out) = identify_in_out(julia_kernel_type, gws, kernel_name, t_args...)
        end
        shared_mem = 0
        if julia_kernel_type == IrisHRT.iris_julia_jacc_parallel_reduce ||
            julia_kernel_type == IrisHRT.iris_julia_jacc_parallel_for
            shared_mem = find_shared_vars(julia_kernel_type, gws, kernel_name, t_args...)
            #println(Core.stdout, " Shared args:", shared_mem)
        end
        for larray in vcat(out, flush, auto_out)
            p_dmem = get_dmem_from_cache(larray)
            #println(Core.stdout, " out inserting p_mem: ", p_dmem.uid)
            mem_params[p_dmem] = IrisHRT.iris_w
        end
        for larray in vcat(in, auto_in)
            p_dmem = get_dmem_from_cache(larray)
            if !haskey(mem_params, p_dmem)
                #println(Core.stdout, "---- in inserting iris_r p_mem: ", p_dmem, " type:", typeof(larray))
                mem_params[p_dmem] = IrisHRT.iris_r
            else
                #println(Core.stdout, " in inserting iris_rw p_mem: ", p_dmem, " type:", typeof(larray))
                mem_params[p_dmem] = IrisHRT.iris_rw
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
            if isa(arg, Union{AbstractArray, IrisMem, IrisTaskMem})
                p_dmem = get_dmem_from_cache(arg)
                push!(params, Ref(p_dmem))
                push!(kernel_params, p_dmem)
                if haskey(mem_params, p_dmem)
                    push!(params_info, Int32(mem_params[p_dmem]))
                else
                    push!(params_info, IrisHRT.iris_rw)
                end
                if get_type(p_dmem) == iris_custom_type
                    push_custom_type(task0.uid, index, get_dmem_custom_type(p_dmem))
                end
            else
                # Scalar element
                push!(kernel_params, arg)
                type_data = get_iris_type(typeof(arg))
                #println(Core.stdout, " type_data: ", type_data, " arg:", arg, " type of arg: ", typeof(arg))
                push!(params_info, type_data | sizeof(arg))
                push!(params, Ref(arg))
            end
        end
        for mem in refresh
            p_dmem = get_dmem_from_cache(mem)
            IrisHRT.refresh(p_dmem)
        end
        nparams = Int64(length(params))
        #println(Core.stdout, "kernel_params: ", kernel_params)
        #println(Core.stdout, "kernel name:", kernel_name)
        #println(Core.stdout, "off:", off)
        #println(Core.stdout, "gws:", gws)
        #println(Core.stdout, "lws:", lws)
        #println(Core.stdout, "kernel_params     :", kernel_params)

        # Set kernel type
        
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
        #println(Core.stdout, "Before IRIS task creation")
        GC.@preserve gws lws off kernel_name begin
            if !predef_task
                ikernel = iris_task_kernel_struct(task0, kernel_name, length(gws), off_c, gws_c, lws_c, nparams, c_params, pointer(params_info))
				if (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_native 
					iris_kernel_enable_julia_interface(ikernel, julia_kernel_type)
				elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_kernel_abstraction
					iris_kernel_enable_julia_interface(ikernel, julia_kernel_type)
				elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_julia_jacc
					iris_kernel_enable_julia_interface(ikernel, julia_kernel_type)
				elseif (julia_kernel_type & 0xFFFF) == IrisHRT.iris_core_native
					# Do nothing here
				end
                if shared_mem > 0
                    iris_kernel_set_shared_memory_size(ikernel, shared_mem)
                end
            else
                f = getFunction(kernel_name)
                #println(Core.stdout, " kname:", kernel_name, " f:", f)
                f(task0, kernel_params...)
            end
        end
        #println(Core.stdout, "After IRIS task creation")
        if metadata != nothing
            IrisHRT.set_metadata(task0, metadata)
        end
        for mem in flush 
            p_dmem = get_dmem_from_cache(mem)
            IrisHRT.iris_task_dmem_flush_out(task0, p_dmem)
        end
        #println(Core.stdout, "IRIS task dependencies")
        n_dependencies = IrisHRT.IrisTask[field_task(x) for x in dependencies if x !== nothing && x != task0]
        if length(n_dependencies) > 0
            IrisHRT.iris_task_depend(task0, n_dependencies)
        end
        if auto_dependency
            add_auto_dependencies(task0, vcat(in, auto_in), vcat(out, flush, auto_out))
        end
        policy = IrisHRT.set_julia_policy(task0, policy)
        #println(Core.stdout, "IRIS task submit")
        if submit
            IrisHRT.iris_task_submit(task0, policy, Ptr{Int8}(C_NULL), Int64(wait))
        else
            IrisHRT.iris_task_set_policy(task0, Int32(policy))
        end
        return task0
    end
    function describe_backend(backend::IRISBackend)
        println(Core.stdout, "Backend Name: ", backend.backend_name)
    end

end  # module Iris

import Base: size
import Base: sizeof
import Base: axes

function dim(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.ndim(dmem)
end

function uid(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.uid(dmem)
end

function dev_size(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.dev_size(dmem)
end

function dev_size(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem}, index::Int64)
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.dev_size(dmem)[index]
end

function host_size(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.host_size(dmem)
end

function host_size(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem}, index::Int64)
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.host_size(dmem)[index]
end

function sizeof(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.iris_mem_get_size(dmem)
end

function size(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.dev_size(dmem)
end

function size(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem}, index::Int64)
    dmem = IrisHRT.field_dmem(mem)
    return IrisHRT.dev_size(dmem)[index]
end

function axes(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    size = collect(IrisHRT.host_size(dmem))
    axes_from_size = tuple((Base.OneTo(n) for n in size)...)
    return axes_from_size
end

function axes(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem}, index::Int64)
    dmem = IrisHRT.field_dmem(mem)
    size = collect(IrisHRT.host_size(dmem))
    axes_from_size = tuple((Base.OneTo(n) for n in size)...)
    return axes_from_size[index]
end

function Base.length(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    size = dev_size(mem)
    s = Base.prod(size)
    return s
end

function Base.pointer(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    dmem = IrisHRT.field_dmem(mem)
    host, dsize = IrisHRT.get_host_4_dmem(dmem)
    return pointer(host)
end

function Base.eltype(mem::Union{IrisHRT.IrisMem, IrisHRT.IrisTaskMem})
    return IrisHRT.get_julia_type(mem)
end

Base.eltype(::Type{IrisHRT.IrisMem1D{T}}) where {T} = T
Base.length(v::IrisHRT.IrisMem1D) = length(v.mem)
Base.size(v::IrisHRT.IrisMem1D) = size(v.mem)
Base.getindex(v::IrisHRT.IrisMem1D, i::Int) = Base.getindex(v.mem, i)
Base.axes(v::IrisHRT.IrisMem1D) = Base.axes(v.mem)
Base.IndexStyle(::Type{<:IrisHRT.IrisMem1D}) = IndexLinear()  # tells code it’s a simple linear index

Base.eltype(::Type{IrisHRT.IrisMem2D{T}}) where {T} = T
Base.length(v::IrisHRT.IrisMem2D) = length(v.mem)
Base.size(v::IrisHRT.IrisMem2D) = size(v.mem)
Base.getindex(v::IrisHRT.IrisMem2D, i::Int, j::Int) = Base.getindex(v.mem, i, j)
Base.axes(v::IrisHRT.IrisMem2D) = Base.axes(v.mem)
Base.IndexStyle(::Type{<:IrisHRT.IrisMem2D}) = IndexCartesian()  # tells code it’s a simple linear index

Base.eltype(::Type{IrisHRT.IrisMem3D{T}}) where {T} = T
Base.length(v::IrisHRT.IrisMem3D) = length(v.mem)
Base.size(v::IrisHRT.IrisMem3D) = size(v.mem)
Base.getindex(v::IrisHRT.IrisMem3D, i::Int, j::Int, k::Int) = Base.getindex(v.mem, i, j, k)
Base.axes(v::IrisHRT.IrisMem3D) = Base.axes(v.mem)
Base.IndexStyle(::Type{<:IrisHRT.IrisMem3D}) = IndexCartesian()  # tells code it’s a simple linear index

Base.eltype(::Type{IrisHRT.IrisMem4D{T}}) where {T} = T
Base.length(v::IrisHRT.IrisMem4D) = length(v.mem)
Base.size(v::IrisHRT.IrisMem4D) = size(v.mem)
Base.getindex(v::IrisHRT.IrisMem4D, i::Int, j::Int, k::Int, l::Int) = Base.getindex(v.mem, i, j, k, l)
Base.axes(v::IrisHRT.IrisMem4D) = Base.axes(v.mem)
Base.IndexStyle(::Type{<:IrisHRT.IrisMem4D}) = IndexCartesian()  # tells code it’s a simple linear index

include("Tiling1D.jl")
include("Tiling2D.jl")
include("Tiling3D.jl")
include("Kernels.jl")

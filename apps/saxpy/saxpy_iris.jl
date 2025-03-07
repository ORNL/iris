
using Libdl
if !haskey(ENV, "IRIS_ARCHS")
ENV["IRIS_ARCHS"] = "cuda:hip:openmp"
end
#ENV["IRIS_ARCHS"] = "cuda"
#ENV["IRIS"] = "/noback/nqx/Ranger/tmp/iris.dev.prof/install.julia"
const iris_path = ENV["IRIS"]
ENV["LD_LIBRARY_PATH"] =  iris_path * "/lib64:" * iris_path * "/lib:" * ENV["LD_LIBRARY_PATH"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)

using .IrisHRT
const iris_arch = ENV["IRIS_ARCHS"]
println(Core.stdout, "IRIS_ARCHS is set to $iris_arch")
println(Core.stdout, "IRIS is set to $iris_path")

using CUDA
#using AMDGPU
using Base.Threads

#s = CUDA.CuStream()
#println("CuStream: $s")
#println("Size of CuContext type: ", sizeof(CUDA.CuContext), " bytes")
#println("Size of CuStream type: ", sizeof(CUDA.CuStream), " bytes")
#println("Size of CUstream type: ", sizeof(CUDA.CUstream), " bytes")
mutable struct CuStream1
    const handle::CUDA.CUstream
    Base.@atomic valid::Bool
    const ctx::Union{Nothing,CUDA.CuContext}
end
#println("Size of CuStream1 type: ", sizeof(CuStream1), " bytes")

function saxpy_cuda(Z, A, X, Y)
    # Calculate global index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds Z[i] = A*X[i] + Y[i]
    return nothing
end

function saxpy_hip(Z, A, X, Y)
    # Calculate global index
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    @inbounds Z[i] = A * X[i] + Y[i]
    return nothing
end
function saxpy_openmp(Z, A, X, Y)
    n = length(X)
    @assert length(Y) == n "Vectors X and Y must have the same length"
    @assert length(Z) == n "Vectors X and Z must have the same length"
    nthreads = Threads.nthreads()
    num_cpus = Sys.CPU_THREADS
    #println(Core.stdout, "NThreads: $nthreads N CPUs:$num_cpus")
    IrisHRT.@maybethreads for i in 1 : n
        Z[i] = A * X[i] + Y[i]
    end
end

#@precompile saxpy_cuda(CuArray{Flat32}, Float32, CuArray{Flat32}, CuArray{Flat32})
#maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
#threads = min(SIZE, maxPossibleThreads)
#blocks = ceil(Int, SIZE / threads)

function saxpy_iris_new_v1(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)
    gc_state = IrisHRT.gc_safe_enter()
    #@iris in=[X,Y] out=[Z] flush=[Z] sync=0 gws=Int64[SIZE] saxpy(Z, A, X, Y)
    task0=IrisHRT.task(in=[X,Y],   out=[Z], wait=0, gws=Int64[SIZE], kernel="saxpy", args=[Z, A, X, Y], dependencies=[])
    task1=IrisHRT.task(in=[X,Y],   out=[Z], wait=0, gws=Int64[SIZE], kernel="saxpy", args=[Z, A, X, Y], dependencies=[task0])
    task2=IrisHRT.task(in=[X,Y], flush=[Z], wait=0, gws=Int64[SIZE], kernel="saxpy", args=[Z, A, X, Y], dependencies=[task0, task1])

    IrisHRT.synchronize()
    IrisHRT.gc_leave(gc_state)
end

function saxpy_iris_new(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)
    gc_state = IrisHRT.gc_safe_enter()
    mem_X = IrisHRT.dmem(X)
    mem_Y = IrisHRT.dmem(Y)
    mem_Z = IrisHRT.dmem(Z)
    #@iris in=[X,Y] out=[Z] flush=[Z] sync=0 gws=Int64[SIZE] saxpy(Z, A, X, Y)
    task0=IrisHRT.task(in=[mem_X,mem_Y], out=[mem_Z], wait=0, gws=Int64[SIZE], kernel="saxpy", args=[mem_Z, A, mem_X, mem_Y], dependencies=[])
    task1=IrisHRT.task(in=[mem_X,mem_Y], out=[mem_Z], wait=0, gws=Int64[SIZE], kernel="saxpy", args=[mem_Z, A, mem_X, mem_Y], dependencies=[task0])
    task2=IrisHRT.task(in=[mem_X,mem_Y], flush=[mem_Z], wait=0, gws=Int64[SIZE], kernel="saxpy", args=[mem_Z, A, mem_X, mem_Y], dependencies=[task0, task1])

    IrisHRT.synchronize()
    IrisHRT.gc_leave(gc_state)
end

function saxpy_iris(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    gc_state = @ccall(jl_gc_safe_enter()::Int8)
    #println("Initialized IRIS")
    # Create IRIS memory objects
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    saxpy_params = [(mem_Z, IrisHRT.iris_w), A, (mem_X, IrisHRT.iris_r), (mem_Y, IrisHRT.iris_r)]

    # Create IRIS task
    task0 = IrisHRT.iris_task_julia("saxpy", 1, Int64[], [SIZE], Int64[], saxpy_params)

    # Flush the output
    IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    IrisHRT.iris_task_submit(task0, IrisHRT.iris_roundrobin, Ptr{Int8}(C_NULL), 0)
    
    IrisHRT.synchronize()
    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)
    @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
end

function saxpy_iris_native(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    #println("Initialized IRIS")
    # Create IRIS memory objects
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    saxpy_params      =      [Ref(mem_Z),     Ref(A),               Ref(mem_X),     Ref(mem_Y)]
    saxpy_params_info = Int32[IrisHRT.iris_w, sizeof(Float32), IrisHRT.iris_r, IrisHRT.iris_r]

    # Create IRIS task
    task0 = IrisHRT.iris_task_native("saxpy", 1, Int64[], 
            [SIZE], Int64[], 4, saxpy_params, saxpy_params_info)
    # Flush the output
    IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    IrisHRT.iris_task_submit(task0, IrisHRT.iris_roundrobin, Ptr{Int8}(C_NULL), 1)

    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)

end
const TASK_CUDA_CONTEXT = Symbol("TASK_CUDA_CONTEXT")

# Function to use the task-local CUDA context
function use_task_context()
    # Retrieve the task-local CUDA context
    task_ctx = task_local_state(TASK_CUDA_CONTEXT, nothing)

    if task_ctx !== nothing
        # Activate the task-local context
        CUDA.context!(task_ctx)
        println(Core.stdout, "Using task-local CUDA context: ", task_ctx)
    else
        println(Core.stdout, "No task-local CUDA context found.")
    end
end

# Assuming c_ctx is a `Ptr{CUDA.CUctx_st}` passed from C
function set_cuda_context!(c_ctx::Ptr{Cvoid})
    # Load the pointer to CUcontext
    cu_ctx_ptr = unsafe_load(reinterpret(Ptr{Ptr{CUDA.CUctx_st}}, c_ctx))

    # Convert the pointer to CuContext
    cu_ctx_conv = CUDA.CuContext(cu_ctx_ptr)

    # Set the CUDA context
    CUDA.context!(cu_ctx_conv)
    return cu_ctx_conv
end

function saxpy_iris2_cuda(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    #println("Initialized IRIS")
    # Create IRIS memory objects
    println(Core.stdout, "L1")
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)
    
    println(Core.stdout, "L2")

    # Set up the task parameters for the kernel
    saxpy_params      =      [(mem_Z, IrisHRT.iris_w), A, (mem_X, IrisHRT.iris_r), (mem_Y, IrisHRT.iris_r)]

    println(Core.stdout, "L3")
    # Create IRIS task
    task0 = IrisHRT.iris_task_julia("saxpy", 1, Int64[], 
            [SIZE], Int64[], saxpy_params)
    # Flush the output
    #IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    println(Core.stdout, "L4")
    c_ctx = IrisHRT.iris_dev_get_ctx(0)
    println(Core.stdout, "L5 IRIS Ctx:$c_ctx")
    c_stream = IrisHRT.iris_dev_get_stream(0, 1)
    #println("Ctx: $c_ctx")
    println(Core.stdout, "L6")
    cu_dev = CUDA.device!(0)
    cu_ctx_conv = set_cuda_context!(c_ctx)

#cu_ctx_ptr = unsafe_load(reinterpret(Ptr{Ptr{CUDA.CUctx_st}}, c_ctx))
#cu_ctx_conv = CUDA.CuContext(cu_ctx_ptr)
    cu_ctx = unsafe_load(reinterpret(Ptr{CuContext}, c_ctx))
    println(Core.stdout, "L7 $cu_ctx_conv $cu_ctx")
    cu_stream_ptr = unsafe_load(reinterpret(Ptr{CUDA.CUstream}, c_stream))
    println(Core.stdout, "L8")
    iris_stream = IRISCuStream(cu_stream_ptr, true, cu_ctx_conv)
    println(Core.stdout, "L9 iris_stream:$iris_stream")
    cu_stream = unsafe_load(reinterpret(Ptr{CUDA.CuStream}, Base.unsafe_convert(Ptr{IRISCuStream}, Ref(iris_stream))))
    #println("Ctx: $cu_ctx")
    println(Core.stdout, "L10 cu_stream:$cu_stream")
    IrisHRT.iris_task_kernel_launch_disabled(task0, 1)
    println(Core.stdout, "L11")
    IrisHRT.iris_task_submit(task0, IrisHRT.iris_roundrobin, Ptr{Int8}(C_NULL), 1)
    println(Core.stdout, "L12")

    X_ptr_dev = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_X, 0))
    X_ptr_j = reinterpret(CuPtr{Float32}, X_ptr_dev)
    X_ptr = CUDA.unsafe_wrap(CuArray, X_ptr_j, (SIZE,), own=false)
    println(Core.stdout, "L13 X:", X_ptr[max(1, end-9):end], "IRIS ptr:", X_ptr_dev)
    Y_ptr_dev = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_Y, 0))
    Y_ptr_j = reinterpret(CuPtr{Float32}, Y_ptr_dev)
    Y_ptr = unsafe_wrap(CuArray, Y_ptr_j, (SIZE,), own=false)
    println(Core.stdout, "L14 Y:", Y_ptr[max(1, end-9):end], "IRIS ptr:", Y_ptr_dev)
    Z_ptr_dev = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_Z, 0))
    Z_ptr_j = reinterpret(CuPtr{Float32}, Z_ptr_dev)
    Z_ptr = unsafe_wrap(CuArray, Z_ptr_j, (SIZE,), own=false)
    println(Core.stdout, "L15 Z:", Z_ptr[max(1, end-9):end], "IRIS ptr:", Z_ptr_dev)
    size_dims = (Int64(SIZE),)

    all_args = [Z_ptr, A, X_ptr, Y_ptr]
    println(Core.stdout, "L16")
    #println(Core.stdout, "In X:$X_ptr Y:$Y_ptr Z:$Z_ptr A:$A")
    println(Core.stdout, "L17")
    #CUDA.context!(cu_ctx)
    #IrisHRT.call_cuda_kernel("saxpy", (SIZE,), (1,), all_args)
    println(Core.stdout, "L18 IRIS context:", cu_ctx, " IRIS:", CUDA.context())
    #CUDA.context!(cu_ctx_conv)
    #IrisHRT.call_cuda_kernel("saxpy", (SIZE,), (1,), all_args)
    println(Core.stdout, "L18-0 IRIS context:", cu_ctx_conv, " IRIS:", CUDA.context())
    maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE / threads)
    #blocks = (1,)
    #threads = (SIZE,)
    p_ctx = CUDA.context()
    println(Core.stdout, "L18-1 Julia context:$p_ctx")
    gc_state = @ccall(jl_gc_safe_enter()::Int8)
    println(Core.stdout, "---------ASynchronous CUDA execution blocks:$blocks threads:$threads stream:$cu_stream----------")
    println(Core.stdout, "type of X: ", typeof(X_ptr))
    direct_path=true
    if direct_path 
        #task = CUDA.@async begin
        #    CUDA.context!(cu_ctx_conv)
            @cuda threads=threads blocks=blocks stream=cu_stream launch=true saxpy_cuda(all_args...)
            println(Core.stdout, "Waiting now/..")
            #CUDA.synchronize(cu_stream)
            #copyto!(Z, Z_ptr)
            #println(Core.stdout, "1-1st Out Z :", Z_ptr[max(1, end-9):end])
        #end
        #wait(task)
    else
        all_args = [Z_ptr, X_ptr, Y_ptr]
        kernel_func = CUDA.@cuda launch=false saxpy_cuda_v1(all_args...)
        println(Core.stdout, "Launching kernel with raw stream...", kernel_func.fun)

        blockdim = CUDA.CuDim3(blocks)
        threaddim = CUDA.CuDim3(threads)
        blockDimX = UInt32(blockdim.x)
        blockDimY = UInt32(blockdim.y)
        blockDimZ = UInt32(blockdim.z)
        gridDimX = UInt32(threaddim.x)
        gridDimY = UInt32(threaddim.y)
        gridDimZ = UInt32(threaddim.z)
        shmem=Integer(0)
        #A_ptr = Base.unsafe_convert(Ptr{Cvoid}, pointer_from_objref(Ref(A)))
        #A_ptr = Base.unsafe_convert(Ptr{Cvoid}, Ref(A))
        A_ptr = Base.unsafe_convert(Ptr{Cvoid}, Ref(A))
        #kernel_params = Ref((pointer(Z_ptr), A_ptr, pointer(X_ptr), pointer(Y_ptr)))
        #kernel_params = Ref((reinterpret(Ptr{Cvoid}, pointer(Z_ptr)), A_ptr, reinterpret(Ptr{Cvoid}, pointer(X_ptr)), reinterpret(Ptr{Cvoid}, pointer(Y_ptr))))
        kernel_params = Ref((reinterpret(Ptr{Cvoid}, pointer(Z_ptr)), reinterpret(Ptr{Cvoid}, pointer(X_ptr)), reinterpret(Ptr{Cvoid}, pointer(Y_ptr))))
        kernel_params_ptr = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, kernel_params)
        #kernel_params = Ref((Z_ptr_j, Ref(A), X_ptr_j, Y_ptr_j))
        
        println(Core.stdout, "~~~~Func: ", kernel_func)
        println(Core.stdout, "Ref-A ", A_ptr, "Size of:", sizeof(A), " A:", A)
        println(Core.stdout, "Func type: ", kernel_func.fun)
        println(Core.stdout, "kernel_params:", kernel_params)
        host_ref = kernel_func.fun.handle

        # Reinterpret the reference to a pointer
        c_void_ptr = Base.unsafe_convert(Ptr{Cvoid}, host_ref)

        # Load the pointer (unsafe)
        k_args = [Z_ptr_dev, 
                A_ptr, 
                X_ptr_dev, 
                Y_ptr_dev]
        println(Core.stdout, "k_args size:", sizeof(k_args))
        k_ptrs = Ref(k_args)



        extra = C_NULL
        call_stream = reinterpret(Ptr{Cvoid}, c_stream)
        #IrisHRT.iris_vendor_kernel_launch(0, c_void_ptr, gridDimX, gridDimY, gridDimZ, 
        #        blockDimX, blockDimY, blockDimZ,
        #        0, call_stream, kernel_params_ptr)
        lib = Libdl.dlopen("libcuda.so")
        result = ccall(Libdl.dlsym(lib, :cuLaunchKernel), Int32,
               (Ptr{Cvoid}, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
               c_void_ptr, UInt32(gridDimX), UInt32(gridDimY), UInt32(gridDimZ),
               UInt32(blockDimX), UInt32(blockDimY), UInt32(blockDimZ), UInt32(0),
               call_stream, kernel_params_ptr, C_NULL)

    end
    #ccall(
    #    (:cuLaunchKernel, CUDA.libcuda), Cint,
    #    (Ptr{Cvoid},  # Kernel function
    #     Cuint, Cuint, Cuint,  # Grid dimensions
    #     Cuint, Cuint, Cuint,  # Block dimensions
    #     Cuint, Ptr{Cvoid},  # Shared memory and stream
    #     Ptr{Ptr{Cvoid}}),   # Kernel arguments
    #    c_void_ptr,  # Kernel function pointer
    #    gridDimX, gridDimY, gridDimZ,  # Grid dimensions
    #    blockDimX, blockDimY, blockDimZ,  # Block dimensions
    #    32, cu_stream_ptr,  # Shared memory and stream
    #    kernel_params# Kernel arguments
    #)
    #task = CUDA.@async begin
    #    @cuda threads=threads blocks=blocks stream=cu_stream saxpy_cuda(all_args...)
    #    CUDA.synchronize(cu_stream)
    #end
    #wait(task)
    #CUDA.task_local_state!(TASK_CUDA_CONTEXT => cu_ctx)
    #CUDA.@async begin
    #    CUDA.device!(0)
    #    CUDA.context!(cu_ctx)
    #    #use_task_context()
    #    #p_ctx = CUDA.context()
    #    println(Core.stdout, "L18-2 $p_ctx")
    #    tls = CUDA.task_local_storage()[:CUDA]
    #    println(Core.stdout, "L18-3 $tls")
    #    tls.device = cu_dev
    #    tls.context = cu_ctx
    #    tls.streams[1] = cu_stream
    #    println(Core.stdout, "L18-4 $tls")
    #    @cuda threads=threads blocks=blocks stream=cu_stream saxpy_cuda(all_args...) 
    #end

    copyto!(Z, Z_ptr)
    println(Core.stdout, "1st Out Z :", Z_ptr[max(1, end-9):end])
    println(Core.stdout, "L19")
    IrisHRT.iris_synchronize()
    @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
    println(Core.stdout, "L20")
    CUDA.device!(0)
    CUDA.context!(cu_ctx)
    copyto!(Z, Z_ptr)
    println(Core.stdout, "Out Z :", Z_ptr[max(1, end-9):end])

    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)

end

function saxpy_julia(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    # Check that X and Y have the same length
    @assert length(X) == length(Y) "Vectors X and Y must have the same length"

    # Perform the SAXPY operation
    for i in 1:length(X)
        Z[i] += A * X[i] + Y[i]
    end
end

function compare_arrays(X::Vector{Float32}, Y::Vector{Float32})::Bool
    SIZE = length(X)
    # Check that SIZE does not exceed the length of X and Y
    @assert SIZE <= length(X) && SIZE <= length(Y) "SIZE must not exceed the length of vectors X and Y"

    for i in 1:SIZE
        if X[i] != Y[i]
            return false
        end
    end
    return true
end

# Check if there are enough arguments
if length(ARGS) < 1
    println("Usage: julia example.jl <arg1>")
    exit(1)
end


m = parse(Int, ARGS[1])
# Example usage
SIZE = m

function main(SIZE)
#1024*128*2048
A = Float32(2.0f0)  # Assuming A is a constant defined somewhere
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
Z = zeros(Float32, SIZE)
Ref_Z = zeros(Float32, SIZE)
# Initialize IRIS
#@spawn IrisHRT.iris_println("******Hello World******1")
IrisHRT.init(1)
#@spawn IrisHRT.iris_println("******Hello World******2")
exit
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
saxpy_julia(A, X, Y, Ref_Z)
julia_start = time()
#saxpy_iris2_cuda(A, X, Y, Z)
#saxpy_iris(A, X, Y, Z)
maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
threads = min(SIZE, maxPossibleThreads)
blocks = ceil(Int, SIZE / threads)
saxpy_iris_new(A, X, Y, Z)
julia_time0 = time() - julia_start
println("1st Julia time: ", julia_time0)
println("Z     :", Z[max(1, end-9):end])
println("Ref_Z :", Ref_Z[max(1, end-9):end])

julia_time1  = julia_time0
julia_time2  = julia_time0

if false
julia_start = time()
saxpy_iris(A, X, Y, Z)
julia_time1 = time() - julia_start
println("2nd Julia time: ", julia_time1)
println("Z     :", Z[max(1, end-9):end])
println("Ref_Z :", Ref_Z[max(1, end-9):end])

julia_start = time()
saxpy_iris(A, X, Y, Z)
julia_time2 = time() - julia_start
println("3rd Julia time: ", julia_time2)
println("Z     :", Z[max(1, end-9):end])
println("Ref_Z :", Ref_Z[max(1, end-9):end])
end

println("SIZE:$SIZE time0:$julia_time0 time1:$julia_time1 time2:$julia_time2")
#julia_iris_start = time()
#output = compare_arrays(Z, Ref_Z)
#println("Output Matching: ", output)
#saxpy_iris(A, X, Y, Z)
#julia_iris_time = time() - julia_iris_start
#println("Julia IRIS time: ", julia_iris_time)
#output = compare_arrays(Z, Ref_Z)
#println("Output Matching: ", output)
# Finalize IRIS
IrisHRT.finalize()
end

main(SIZE)


ENV["IRIS_ARCHS"] = "cuda"
#ENV["IRIS"] = "/noback/nqx/Ranger/tmp/iris.dev.prof/install.zenith"

const iris_path = ENV["IRIS"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)
using .IrisHRT

println("Size of Cint in bytes: ", sizeof(Cint), " bytes")
println("Size of Cint in bits: ", sizeof(Cint) * 8, " bits")

# Define a CUDA kernel function
using CUDA
using AMDGPU
using Base.Threads
const iris_arch = ENV["IRIS_ARCHS"]
function saxpy_cuda(Z, A, X, Y)
    # Calculate global index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds Z[i] = A * X[i] + Y[i]
    return nothing
end

function saxpy_hip(Z, A, X, Y)
    # Calculate global index
    i = (workgroupIdx().x - 0x1) * workgroupDim().x + workitemIdx().x
    @inbounds Z[i] = A * X[i] + Y[i]
    return nothing
end

function saxpy_openmp(Z, A, X, Y)
    n = length(X)
    @assert length(Y) == n "Vectors X and Y must have the same length"
    @assert length(Z) == n "Vectors X and Z must have the same length"
    n = length(X)
    nthreads = Threads.nthreads()
    num_cpus = Sys.CPU_THREADS
    #println(Core.stdout, "NThreads: $nthreads N CPUs:$num_cpus")
    IrisHRT.@maybethreads for i in 1 : n
        Z[i] = A * X[i] + Y[i]
    end
end

function saxpy_direct_cuda(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})

    A = 2.0
    SIZE=256
    X_host = ones(SIZE)
    Y_host = ones(SIZE)
    Z_host = ones(SIZE)

    SIZE=length(X)
    X_device = CuArray(X)
    Y_device = CuArray(Y)
    Z_device = CuArray(Z)

    maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE / threads)
    #println("Threads: $threads Blocks: $blocks")
    println("In X:$X_device Y:$Y_device Z:$Z_device A:$A")
    CUDA.@sync @cuda threads=threads blocks=blocks saxpy_cuda(Z_device, A, X_device, Y_device) 
    
    println("Out Z: $Z_device")

end

function saxpy_iris2_cuda(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    #println("Initialized IRIS")
    # Create IRIS memory objects
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    saxpy_params      =      [(mem_Z, IrisHRT.iris_w), A, (mem_X, IrisHRT.iris_r), (mem_Y, IrisHRT.iris_r)]

    # Create IRIS task
    task0 = IrisHRT.iris_task_julia("saxpy", 1, Int64[], 
            [SIZE], Int64[], saxpy_params)
    # Flush the output
    #IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    c_ctx = IrisHRT.iris_dev_get_ctx(0)
    #println("Ctx: $c_ctx")
    cu_ctx = unsafe_load(reinterpret(Ptr{CuContext}, c_ctx))
    #println("Ctx: $cu_ctx")
    IrisHRT.iris_task_kernel_launch_disabled(task0, 1)
    IrisHRT.iris_task_submit(task0, IrisHRT.iris_roundrobin, Ptr{Int8}(C_NULL), 1)

    X_ptr = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_X, 0))
    X_ptr = reinterpret(CuPtr{Float32}, X_ptr)
    X_ptr = unsafe_wrap(CuArray, X_ptr, (SIZE,), own=false)
    Y_ptr = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_Y, 0))
    Y_ptr = reinterpret(CuPtr{Float32}, Y_ptr)
    Y_ptr = unsafe_wrap(CuArray, Y_ptr, (SIZE,), own=false)
    Z_ptr = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_Z, 0))
    Z_ptr = reinterpret(CuPtr{Float32}, Z_ptr)
    Z_ptr = unsafe_wrap(CuArray, Z_ptr, (SIZE,), own=false)
    size_dims = (Int64(SIZE),)

    all_args = [Z_ptr, A, X_ptr, Y_ptr]
    println("In X:$X_ptr Y:$Y_ptr Z:$Z_ptr A:$A")
    CUDA.device!(0)
    CUDA.context!(cu_ctx)
    IrisHRT.call_cuda_kernel("saxpy", (SIZE,), (1,), all_args)
    #CUDA.@sync @cuda threads=(SIZE,) blocks=(0x001,) saxpy_cuda(all_args...) 
    copyto!(Z, Z_ptr)
    println("Out Z :$Z_ptr")

    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)

end

function saxpy_iris2_hip(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    #println("Initialized IRIS")
    #println("Xorig:$X, $Y, $Z")
    # Create IRIS memory objects
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    saxpy_params      =      [(mem_Z, IrisHRT.iris_w), A, (mem_X, IrisHRT.iris_r), (mem_Y, IrisHRT.iris_r)]

    # Create IRIS task
    task0 = IrisHRT.iris_task_julia("saxpy", 1, Int64[], 
            [SIZE], Int64[], saxpy_params)
    # Flush the output
    #IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    c_ctx = IrisHRT.iris_dev_get_ctx(0)
    #println("Ctx: $c_ctx")
    hip_ctx = unsafe_load(reinterpret(Ptr{HIPContext}, c_ctx))
    #println("Ctx: $hip_ctx")
    IrisHRT.iris_task_kernel_launch_disabled(task0, 1)
    IrisHRT.iris_task_submit(task0, IrisHRT.iris_roundrobin, Ptr{Int8}(C_NULL), 1)

    X_ptr = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_X, 0))
    X_ptr = reinterpret(Ptr{Float32}, X_ptr)
    X_ptr = unsafe_wrap(ROCArray, X_ptr, (SIZE,), lock=false)
    #println("X2: $X_ptr")
    Y_ptr = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_Y, 0))
    Y_ptr = reinterpret(Ptr{Float32}, Y_ptr)
    Y_ptr = unsafe_wrap(ROCArray, Y_ptr, (SIZE,), lock=false)
    #println("Y2: $Y_ptr")
    Z_ptr = Ptr{Float32}(IrisHRT.iris_mem_arch_ptr(mem_Z, 0))
    Z_ptr = reinterpret(Ptr{Float32}, Z_ptr)
    Z_ptr = unsafe_wrap(ROCArray, Z_ptr, (SIZE,), lock=false)
    #println("Z2: $Z_ptr")
    size_dims = (Int64(SIZE),)

    all_args = [Z_ptr, A, X_ptr, Y_ptr]
    println("In X:$X_ptr Y:$Y_ptr Z:$Z_ptr A:$A")
    #AMDGPU.device!(AMDGPU.devices()[1])
    #AMDGPU.context!(hip_ctx)
    AMDGPU.@sync @roc groupsize=(SIZE,) gridsize=(1,) saxpy_hip(Z_ptr, Float32(2.0), X_ptr, Y_ptr)
    #IrisHRT.call_hip_kernel("saxpy", (SIZE,), (1,), all_args)
    #CUDA.@sync @cuda threads=(SIZE,) blocks=(0x001,) saxpy_cuda(all_args...) 
    copyto!(Z, Z_ptr)
    println("Out Z :$Z_ptr")

    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)
end

function saxpy_iris(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    println("Initialized IRIS")
    # Create IRIS memory objects
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    saxpy_params = [(mem_Z, IrisHRT.iris_w), A, (mem_X, IrisHRT.iris_r), (mem_Y, IrisHRT.iris_r)]

    # Create IRIS task
    # @iris in=(mem_X, mem_Y) out=(mem_Z) saxpy_cuda(mem_Z, A, mem_X, mem_Y)
    task0 = IrisHRT.iris_task_julia("saxpy", 1, Int64[], [SIZE], Int64[], saxpy_params)

    # Flush the output
    IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    IrisHRT.iris_task_submit(task0, IrisHRT.iris_roundrobin, Ptr{Int8}(C_NULL), 1)

    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)
end

function saxpy_iris_old(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    println("Initialized IRIS")
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



# Example usage
SIZE = 8
#1024*128*2048
A = Float32(2.0f0)  # Assuming A is a constant defined somewhere
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
Z = zeros(Float32, SIZE)
Ref_Z = zeros(Float32, SIZE)
# Initialize IRIS
IrisHRT.iris_init(Int32(1))

julia_start = time()
#saxpy_direct_cuda(A, X, Y, Z)
if iris_arch == "cuda"
saxpy_iris2_cuda(A, X, Y, Z)
end
if iris_arch == "hip"
saxpy_iris2_hip(A, X, Y, Z)
end
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
saxpy_julia(A, X, Y, Ref_Z)
saxpy_iris(A, X, Y, Z)
julia_time = time() - julia_start
#println("Julia time: ", julia_time)
#julia_iris_start = time()
#output = compare_arrays(Z, Ref_Z)
#println("Output Matching: ", output)
#saxpy_iris(A, X, Y, Z)
#julia_iris_time = time() - julia_iris_start
#println("Julia IRIS time: ", julia_iris_time)
#output = compare_arrays(Z, Ref_Z)
#println("Output Matching: ", output)
println("Z     :", Z)
println("Ref_Z :", Ref_Z)
# Finalize IRIS
IrisHRT.iris_finalize()


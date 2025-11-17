
ENV["IRIS_ARCHS"] = "cuda"
using Julius
using Base.Threads
using CUDA

#__precompile__(false)
function saxpy_cuda(Z, A, X, Y)
    # Calculate global index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds Z[i] = A * X[i] + Y[i]
    return nothing
end
function saxpy_cuda_wrapper(threads, blocks, args_tuple)
    println(Core.stdout, "threads:$threads blocks:$blocks")
    CUDA.@sync @cuda threads=threads blocks=blocks saxpy_cuda(args_tuple...)
end

function saxpy_hip(Z, A, X, Y)
    # Calculate global index
    i = (workgroupIdx().x - 0x1) * workgroupDim().x + workitemIdx().x
    @inbounds Z[i] = A * X[i] + Y[i]
    return nothing
end
#using GPUCompiler
#ctx = GPUCompiler.JuliaContext()
#println(Core.stdout, "Julia Context: $ctx")
function saxpy_openmp(Z, A, X, Y)
    n = length(X)
    @assert length(Y) == n "Vectors X and Y must have the same length"
    @assert length(Z) == n "Vectors X and Z must have the same length"
    n = length(X)
    nthreads = Threads.nthreads()
    num_cpus = Sys.CPU_THREADS
    #println(Core.stdout, "NThreads: $nthreads N CPUs:$num_cpus")
    Julius.@maybethreads for i in 1 : n
        Z[i] = A * X[i] + Y[i]
    end
end

function saxpy_julius(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    println(Core.stdout, "******Output******")
    task0 = Julius.@jtask in:[X, Y], flush:[Z], gws:size(Z), wait:true, saxpy(Z, A, X, Y)
    println(Core.stdout, "----- Completed ------")
end

function saxpy_iris(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    println(Core.stdout, "******Output******")
    SIZE = length(X)

    println(Core.stdout, "----------Initialized IRIS-------")
    # Create IRIS memory objects
    mem_X = Julius.iris_data_mem(X)
    mem_Y = Julius.iris_data_mem(Y)
    mem_Z = Julius.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    println(Core.stdout, "-------Initialized saxpy_params")
    saxpy_params = [(mem_Z, Julius.iris_w), A, (mem_X, Julius.iris_r), (mem_Y, Julius.iris_r)]
    println(Core.stdout, "-------After Initialized saxpy_params")

    # Create IRIS task
    # @iris in=(mem_X, mem_Y) out=(mem_Z) saxpy_cuda(mem_Z, A, mem_X, mem_Y)
    #task0 = Julius.@jtask in:[X, Y], flush:[Z], saxpy(Z, A, X, Y)
    task0 = Julius.iris_task_julia("saxpy", 1, Int64[], [SIZE], Int64[], saxpy_params)

    # Flush the output
    Julius.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    println(Core.stdout, "----- Submitting ------")
    Julius.iris_task_submit(task0, Julius.iris_roundrobin, Ptr{Int8}(C_NULL), 1)

    println(Core.stdout, "----- Releasing ------")
    # Release memory objects
    Julius.iris_mem_release(mem_X)
    Julius.iris_mem_release(mem_Y)
    Julius.iris_mem_release(mem_Z)
    println(Core.stdout, "----- Completed ------")
end

function saxpy_iris_old(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    println("Initialized IRIS")
    # Create IRIS memory objects
    mem_X = Julius.iris_data_mem(X)
    mem_Y = Julius.iris_data_mem(Y)
    mem_Z = Julius.iris_data_mem(Z)
    
    # Set up the task parameters for the kernel
    saxpy_params      =      [Ref(mem_Z),     Ref(A),               Ref(mem_X),     Ref(mem_Y)]
    saxpy_params_info = Int32[Julius.iris_w, sizeof(Float32), Julius.iris_r, Julius.iris_r]

    # Create IRIS task
    task0 = Julius.iris_task_native("saxpy", 1, Int64[], 
            [SIZE], Int64[], 4, saxpy_params, saxpy_params_info)
    # Flush the output
    Julius.iris_task_dmem_flush_out(task0, mem_Z)
    # Submit the task
    Julius.iris_task_submit(task0, Julius.iris_roundrobin, Ptr{Int8}(C_NULL), 1)

    # Release memory objects
    Julius.iris_mem_release(mem_X)
    Julius.iris_mem_release(mem_Y)
    Julius.iris_mem_release(mem_Z)

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
d = Julius.init()
println(Core.stdout, "******Output: $d")
julia_start = time()
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
println(Core.stdout, "******Input X: $X")
println(Core.stdout, "******Input Y: $Y")
println(Core.stdout, "******Input A: $A")
saxpy_julia(A, X, Y, Ref_Z)
println(Core.stdout, "******Zref: $Ref_Z")
println(Core.stdout, "******Input A: $A")
println(Core.stdout, "******Input X: $X")
println(Core.stdout, "******Input Y: $Y")
println(Core.stdout, "******Input Zref: $Ref_Z")
saxpy_julius(A, X, Y, Z)
julia_time = time() - julia_start
println("Z     :", Z)
println("Ref_Z :", Ref_Z)
# Finalize IRIS
Julius.finalize()


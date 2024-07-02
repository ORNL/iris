
ENV["IRIS_ARCHS"] = "cuda"
ENV["IRIS"] = "/noback/nqx/Ranger/tmp/iris.dev.prof/install.zenith"

const iris_path = ENV["IRIS"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)
using .IrisHRT
using CUDA

println("Size of Cint in bytes: ", sizeof(Cint), " bytes")
println("Size of Cint in bits: ", sizeof(Cint) * 8, " bits")

# Define a CUDA kernel function
function saxpy_cuda(Z, A, X, Y)
    # Calculate global index
    i = threadIdx().x + blockIdx().x * blockDim().x
    # Check bounds
    if i <= length(Z)
        Z[i] = A * X[i] + Y[i]
    end
    return
end

function saxpy_iris(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    println("Initialized IRIS")
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
SIZE = 10
#1024*128*2048
A = Float32(2.0f0)  # Assuming A is a constant defined somewhere
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
Z = zeros(Float32, SIZE)
Ref_Z = zeros(Float32, SIZE)
# Initialize IRIS
IrisHRT.iris_init(Int32(1))

julia_start = time()
saxpy_julia(A, X, Y, Ref_Z)
saxpy_iris(A, X, Y, Z)
julia_time = time() - julia_start
println("Julia time: ", julia_time)
julia_iris_start = time()
output = compare_arrays(Z, Ref_Z)
println("Output Matching: ", output)
#saxpy_iris(A, X, Y, Z)
#julia_iris_time = time() - julia_iris_start
#println("Julia IRIS time: ", julia_iris_time)
#output = compare_arrays(Z, Ref_Z)
#println("Output Matching: ", output)
println("Z     :", Z)
println("Ref_Z :", Ref_Z)
# Finalize IRIS
IrisHRT.iris_finalize()


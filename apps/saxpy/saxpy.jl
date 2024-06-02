
const iris_path = ENV["IRIS"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)
using .IrisHRT

function saxpy_iris(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    IrisHRT.iris_init(Int32(1))

    # Initialize IRIS
    ndevs = IrisHRT.iris_ndevices()

    # Retrieve the number of devices
    println("Number of devices: ", ndevs[])

    # Create IRIS memory objects
    mem_X = IrisHRT.iris_data_mem(X)
    mem_Y = IrisHRT.iris_data_mem(Y)
    mem_Z = IrisHRT.iris_data_mem(Z)

    # Set up the task parameters for the kernel
    saxpy_params      =      [Ref(mem_Z),     A,               Ref(mem_X),     Ref(mem_Y)]
    saxpy_params_info = Int32[IrisHRT.iris_w, sizeof(Float32), IrisHRT.iris_r, IrisHRT.iris_r]

    # Create IRIS task
    task0 = IrisHRT.iris_task_spec("saxpy", 1, Int64[], 
            [SIZE], Int64[], 4, 
            saxpy_params, 
            saxpy_params_info)
    # Flush the output
    IrisHRT.iris_task_dmem_flush_out(task0, mem_Z)

    # Submit the task
    TARGET = IrisHRT.iris_roundrobin # Assuming TARGET is defined somewhere
    IrisHRT.iris_task_submit(task0, TARGET, Ptr{Int8}(C_NULL), 1)

    print(Z)
    # Release memory objects
    IrisHRT.iris_mem_release(mem_X)
    IrisHRT.iris_mem_release(mem_Y)
    IrisHRT.iris_mem_release(mem_Z)

    # Finalize IRIS
    IrisHRT.iris_finalize()
end

function saxpy(A::Float32, X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
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
A = Float32(2.0f0)  # Assuming A is a constant defined somewhere
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
Z = zeros(Float32, SIZE)
Ref_Z = zeros(Float32, SIZE)
saxpy(A, X, Y, Ref_Z)
saxpy_iris(A, X, Y, Z)
output = compare_arrays(Z, Ref_Z)
println("Matchine: ", output)
println("Z     :", Z)
println("Ref_Z :", Ref_Z)

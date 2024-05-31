
iris_jl_path=ENV["IRIS"] * "/include/iris/Iris.jl"
include(iris_jl_path)
using .Iris

function saxpy_iris(X::Vector{Float32}, Y::Vector{Float32}, Z::Vector{Float32})
    SIZE = length(X)

    # Initialize IRIS
    iris_init(1, Ref(Ptr{Ptr{Cchar}}(C_NULL)), 1)

    # Retrieve the number of devices
    ndevs = Ref{Cint}()
    iris_device_count(ndevs)
    println("Number of devices: ", ndevs[])

    # Create IRIS memory objects
    mem_X = Ref{iris_mem}()
    mem_Y = Ref{iris_mem}()
    mem_Z = Ref{iris_mem}()

    iris_data_mem_create(mem_X, pointer(X), SIZE * sizeof(Float32))
    iris_data_mem_create(mem_Y, pointer(Y), SIZE * sizeof(Float32))
    iris_data_mem_create(mem_Z, pointer(Z), SIZE * sizeof(Float32))

    # Create IRIS task
    task0 = Ref{iris_task}()
    iris_task_create(task0)

    # Set up the parameters for the kernel
    A = 2.0f0  # Assuming A is a constant defined somewhere
    saxpy_params = [mem_Z[], A, mem_X[], mem_Y[]]
    saxpy_params_info = [iris_w, sizeof(Float32), iris_r, iris_r]

    # Launch the kernel
    iris_task_kernel(task0[], "saxpy", 1, C_NULL, Ref(SIZE), C_NULL, 4, saxpy_params, saxpy_params_info)

    # Flush the output
    iris_task_dmem_flush_out(task0[], mem_Z[])

    # Submit the task
    TARGET = 0  # Assuming TARGET is defined somewhere
    iris_task_submit(task0[], TARGET, C_NULL, 1)

    # Release memory objects
    iris_mem_release(mem_X[])
    iris_mem_release(mem_Y[])
    iris_mem_release(mem_Z[])

    # Finalize IRIS
    iris_finalize()
end

# Example usage
SIZE = 1000
X = rand(Float32, SIZE)
Y = rand(Float32, SIZE)
Z = zeros(Float32, SIZE)

saxpy_iris(X, Y, Z)

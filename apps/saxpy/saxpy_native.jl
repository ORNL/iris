const iris_path = ENV["IRIS"]
ENV["LD_LIBRARY_PATH"] =  iris_path * "/lib64:" * iris_path * "/lib:" * ENV["LD_LIBRARY_PATH"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)

using .IrisHRT
using Base.Threads

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
    nthreads = Threads.nthreads()
    num_cpus = Sys.CPU_THREADS
    IrisHRT.@maybethreads for i in 1 : n
        Z[i] = A * X[i] + Y[i]
    end
end

function call(SIZE, A, X, Y, Z)
    julia_start = time()
    task0=IrisHRT.task( in=[X,Y],   
                        flush=[Z], 
                        wait=1, 
                        gws=Int64[SIZE], 
                        kernel="saxpy", 
                        args=[Z, A, X, Y], 
                        dependencies=[])
    IrisHRT.clear_map()
    julia_time = time() - julia_start
    println("Julia time: ", julia_time)
    return Z
end
m = parse(Int, ARGS[1])
# Example usage
SIZE = m

#SIZE=8
A=2.0f0
IrisHRT.init()
gc_state = IrisHRT.gc_safe_enter()
X = IrisHRT.ones(Float32, SIZE)
Y = IrisHRT.ones(Float32, SIZE)
Z = IrisHRT.dmem(Float32, SIZE)
println("NDim:", IrisHRT.ndim(X))
println("Type:", IrisHRT.get_type(X))
println("Size:", IrisHRT.host_size(X))
#println("Data:", IrisHRT.valid_host(X))
call(SIZE, A, X, Y, Z)

Z_host = IrisHRT.host(Z)
println(Core.stdout, "Z    :", Z_host[max(1, end-9):end])

IrisHRT.gc_leave(gc_state)
IrisHRT.finalize()

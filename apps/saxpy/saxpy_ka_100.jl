const iris_path = ENV["IRIS"]
ENV["LD_LIBRARY_PATH"] =  iris_path * "/lib64:" * iris_path * "/lib:" * ENV["LD_LIBRARY_PATH"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)

using .IrisHRT
using Base.Threads
using KernelAbstractions

# Opportunities
## Can we identify in and outs automatically?
## PIM specific intrinsics identification
@kernel function saxpy(Z, A, X, Y)
    i = @index(Global)
    if i <= length(Z)
        Z[i] = A*X[i] + Y[i]
    end
end

@kernel function saxpy_ka_cuda(Z, A, X, Y)
    i = @index(Global)
    if i <= length(Z)
        Z[i] = A*X[i] + Y[i]
    end
end

@kernel function saxpy_ka_hip(Z, A, X, Y)
    i = @index(Global)
    if i <= length(Z)
        Z[i] = A*X[i] + Y[i]
    end
end

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

function call(SIZE, A, X, Y, Z, ka_flag)
    task0=IrisHRT.task( in=[X,Y],
                    flush=[Z],
                    wait=0,
                    gws=Int64[SIZE],
                    kernel="saxpy",
                    args=[Z, A, X, Y],
                    ka=ka_flag,
                    dependencies=[])
end

function call2(SIZE, A, X, Y, Z, ka_flag)
    task0=IrisHRT.task( in=[X,Y],
                    flush=[Z],
                    wait=0,
                    gws=Int64[SIZE],
                    kernel="saxpy2",
                    args=[Z, A, X, Y],
                    ka=ka_flag,
                    dependencies=[])
end

if length(ARGS) < 1
    println("Usage: julia saxpy_hetero.jl <arg1> <arg2> [<arg3>]")
    exit(1)
end
m = 32
ntimes = 10
ka_flag = false
if length(ARGS) > 0
    m = parse(Int, ARGS[1])
end
if length(ARGS) > 1 
    ntimes = parse(Int, ARGS[2])
end
if length(ARGS) > 2
    ka_flag = parse(Bool, ARGS[3]) 
end
# Example usage

SIZE = m
function main(SIZE)
    println("--------------")
    #SIZE=8
    A=2.0f0
    IrisHRT.init()
    gc_state = IrisHRT.gc_safe_enter()
    X = ones(Float32, SIZE)
    Y = ones(Float32, SIZE)
    Z = zeros(Float32, SIZE)
    GC.enable(false)
    #println("Data:", IrisHRT.valid_host(X))
    julia_start = time()
    for i in 1:ntimes
            call(SIZE, A, X, Y, Z, ka_flag)
    end
    IrisHRT.synchronize()
    julia_time = time() - julia_start
    println("Julia time: ", julia_time)
    println("Avg saxpy time: ", julia_time/ntimes)
    IrisHRT.clear_map()

    IrisHRT.gc_leave(gc_state)
    IrisHRT.finalize()
    GC.enable(true)
end

main(SIZE)

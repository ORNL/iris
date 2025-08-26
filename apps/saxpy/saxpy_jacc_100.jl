using Julius
using Base.Threads
using KernelAbstractions

function saxpy_policy(task, devs, ndevs, out_devs)
    println(Core.stdout, "Task: ", task)
    println(Core.stdout, "Devs:", devs)
    println(Core.stdout, "ndevs:", ndevs)
    println(Core.stdout, "out_devs:", out_devs)
    metadata = Julius.metadata(task)
    println(Core.stdout, "Metadata: ", metadata)
    out_devs[1] = 1
    return 1
end
# Opportunities
## Can we identify in and outs automatically?
## PIM specific intrinsics identification
function saxpy(i, Z, A, X, Y)
    if i <= length(Z)
        @inbounds Z[i] = A*X[i] + Y[i]
    end
end

function call(i, SIZE, A, X, Y, Z)
    julia_start = time()
    task0 = Julius.parallel_for(SIZE, saxpy, Z, A, X, Y, flush=[Z], policy=saxpy_policy, metadata=i)
    julia_time = time() - julia_start
    println("Julia time: ", julia_time)
    return Z
end
m = parse(Int, ARGS[1])
# Example usage

ntimes = 100
if length(ARGS) > 0
    m = parse(Int, ARGS[1])
end
if length(ARGS) > 1 
    ntimes = parse(Int, ARGS[2])
end
SIZE = m
function main(SIZE)
    println("--------------")
    #SIZE=8
    A=2.0f0
    Julius.init()
    X = ones(Float32, SIZE)
    Y = ones(Float32, SIZE)
    Z = zeros(Float32, SIZE)
    GC.enable(false)
    #println("Data:", Julius.valid_host(X))
    julia_start = time()
    for i in 1:ntimes
            call(i, SIZE, A, X, Y, Z)
    end
    Julius.synchronize()
    julia_time = time() - julia_start
    println("Total Julia time: ", julia_time)
    println("Avg saxpy time: ", julia_time/ntimes)
    Julius.finalize()
end

main(SIZE)

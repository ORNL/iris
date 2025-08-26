using Julius
using Base.Threads
using KernelAbstractions

# Opportunities
## Can we identify in and outs automatically?
## PIM specific intrinsics identification

function saxpy(i, Z, A, X, Y)
   @inbounds Z[i] = A*X[i] + Y[i]
end
function call(SIZE, A, X, Y, Z)
    julia_start = time()
    #task0 = Julius.parallel_for(SIZE, saxpy, Z, A, X, Y, flush=[Z])
    task0 = Julius.@jparallel_for size:[SIZE], flush:[Z], saxpy(Z, A, X, Y)
    julia_time = time() - julia_start
    println("Julia time: ", julia_time)
    return Z
end
m = parse(Int, ARGS[1])
# Example usage

SIZE = m
function main(SIZE)
    println("--------------")
    #SIZE=8
    A=2.0f0
    Julius.init()
    X = Julius.ones(Float32, SIZE)
    Y = Julius.ones(Float32, SIZE)
    Z = Julius.dmem(Float32, SIZE)

    println("NDim:", Julius.ndim(X))
    println("Type:", Julius.get_type(X))
    println("Size:", Julius.host_size(X))
    #println("Data:", Julius.valid_host(X))
    call(SIZE, A, X, Y, Z)

    Z_host = Julius.host(Z)
    println(Core.stdout, "Z    :", Z_host)
    println(Core.stdout, "Z    :", Z_host[1:min(SIZE, 9)])

    Julius.finalize()
end

main(SIZE)

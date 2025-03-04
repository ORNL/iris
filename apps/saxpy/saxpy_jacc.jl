include(ENV["IRIS"]*"/include/iris/IrisHRT.jl")

using .IrisHRT
using Base.Threads
using KernelAbstractions

# Opportunities
## Can we identify in and outs automatically?
## PIM specific intrinsics identification

function saxpy(i, Z, A, X, Y)
    if i <= length(Z)
        @inbounds Z[i] = A*X[i] + Y[i]
    end
end
function call(SIZE, A, X, Y, Z)
    julia_start = time()
    task0 = IrisHRT.parallel_for(SIZE, saxpy, Z, A, X, Y, flush=[Z])
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
    IrisHRT.init()
    X = IrisHRT.ones(Float32, SIZE)
    Y = IrisHRT.ones(Float32, SIZE)
    Z = IrisHRT.dmem(Float32, SIZE)

    println("NDim:", IrisHRT.ndim(X))
    println("Type:", IrisHRT.get_type(X))
    println("Size:", IrisHRT.host_size(X))
    #println("Data:", IrisHRT.valid_host(X))
    call(SIZE, A, X, Y, Z)

    Z_host = IrisHRT.host(Z)
    println(Core.stdout, "Z    :", Z_host)
    println(Core.stdout, "Z    :", Z_host[1:min(SIZE, 9)])

    IrisHRT.finalize()
end

main(SIZE)

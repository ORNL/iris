include(ENV["IRIS"] * "/include/iris/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

function iris_jacc_add(i, z, x, y)
        @inbounds z[i] = x[i] + y[i]
end
function sum(i, x)
    @inbounds x[i]
end

function main()
    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = IrisHRT.ones(Float64, 8)
    y = IrisHRT.ones(Float64, 8)
    z = IrisHRT.zeros(Float64, 8)
    alpha = 2.5

    IrisHRT.parallel_for(N, Main.iris_jacc_add, z, x, y)
    out = IrisHRT.parallel_reduce(N, Main.sum, z)
    IrisHRT.synchronize()
    println("Y: ", z)
    println("out: ", out)
end
main()
IrisHRT.finalize()

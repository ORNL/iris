include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function iris_jacc_add(i, z, x, y, alpha)
        @inbounds z[i] = alpha * x[i] + y[i]
end
function sumdd(i, x)
    @inbounds x[i]
end

function main()
    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = IrisHRT.ones(Float64, N)
    y = IrisHRT.ones(Float64, N)
    z = IrisHRT.zeros(Float64, N)
    alpha = 2.5

    IrisHRT.parallel_for(N, Main.iris_jacc_add, z, x, y, alpha)
    println("Z: ", z)
    out = IrisHRT.parallel_reduce(N, sumdd, z)
    IrisHRT.synchronize()
    println("Z: ", z)
    println("out: ", out)
    out_host = IrisHRT.host(out)
    @test out_host == [35.0]
end
@testset "iris_jacc_parallel_for_reduce" begin
    main()
end
IrisHRT.finalize()

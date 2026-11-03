include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function iris_equal(a, b)
	@inbounds a == b
end

X = IrisHRT.fill(Float32, 2, 8)
Y = IrisHRT.fill(Float32, 2, 8)

println(" typeof(X):", typeof(X))
println(" typeof(Y):", typeof(Y))

Z = IrisHRT.parallel_reduce(length(X), and, iris_equal, X, Y)
#Z = ==(X, Y)

IrisHRT.wait(Z)
IrisHRT.flush(Z)
println("Z: ", Z)
@testset "copyto_test" begin
#    Z_host = IrisHRT.host(Z)
#    @test Z_host == Float32[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
end
IrisHRT.finalize()

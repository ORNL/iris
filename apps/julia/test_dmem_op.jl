include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

X = IrisHRT.ones(Float32, 8)
Y = IrisHRT.fill(Float32, 2, 8)
A = IrisHRT.ones(Float32, 8)

B = X * Y
Z = A + B

IrisHRT.wait(Z)
IrisHRT.flush(Z)
println("Z: ", Z)
@testset "copyto_test" begin
    Z_host = IrisHRT.host(Z)
    @test Z_host == Float32[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
end
IrisHRT.finalize()

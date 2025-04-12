include(ENV["IRIS"]*"/src/IrisHRT.jl")
using .IrisHRT
using Test
using Random
Random.seed!(1234)

IrisHRT.init()

N=16
x = Float32[
        1.0f0, 4.0f0, 8.3f0, 2.3f0, 
        5.8f0, 3.8f0, 4.5f0, 7.2f0,
        1.1f0, 4.1f0, 8.5f0, 2.4f0, 
        5.8f0, 3.8f0, 4.6f0, 7.3f0,
    ]
y = copy(x)
x = reshape(x, 4, 4)
x1 = @view x[:,2]

m = Float32[0f0]
for i in 1:100
IrisHRT.parallel_reduce(max, y; result=m)
end
IrisHRT.flush(m)

println("x: ", x)
println("x1: ", x1)
println("max: ", m)

@testset "iris_max" begin
    @test m[1] == 8.5
end
IrisHRT.finalize()


include(ENV["IRIS"]*"/src/IrisHRT.jl")
using .IrisHRT
using Test
using Random
Random.seed!(1234)

IrisHRT.init()

x = reshape(collect(Float32,1:128*4), 128,4)
y = copy(x)
m1 = Float32[0f0]
m2 = Float32[0f0]
test_out = Float32[129.0f0, 193.0f0]
x_d = IrisHRT.dmem(x; region_outer_dim=true)

@testset "max" begin
m1 = Float32[0f0]
for (index, i) in enumerate([1, 65])
    IrisHRT.update_size_no_strides(x, [i, 4])
    regions = IrisHRT.regions(x) 
    IrisHRT.parallel_reduce(max, regions[2]; result=m1)
    IrisHRT.flush(m1)
    println("max of size(", i, "): ", m1)
    @test m1[1] == test_out[index]
end
end
IrisHRT.finalize()


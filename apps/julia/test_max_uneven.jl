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
x_d = IrisHRT.dmem(x; region_outer_dim=true)

IrisHRT.update_tile_size(x, [1, 4])
regions = IrisHRT.regions(x) 

IrisHRT.parallel_reduce(max, regions[2]; result=m1)
IrisHRT.flush(m1)


IrisHRT.update_tile_size(x, [65, 4])
regions = IrisHRT.regions(x) 

IrisHRT.parallel_reduce(max, regions[2]; result=m2)
IrisHRT.flush(m2)

println("max1: ", m1)
println("max2: ", m2)
#println("r2: ", regions[2])

@testset "max_uneven" begin
    @test m1[1] == 129.0
    @test m2[1] == 193.0
end
IrisHRT.finalize()


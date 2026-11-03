include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

A = ones(Float64, 4, 4)
AA = IrisHRT.reshape(A, 16)

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

/(AA, AA, sqrt(24))

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

@testset "reshape_op_overhead" begin
    AA_host = IrisHRT.host(AA)
    @test AA_host == [0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154, 0.20412414523193154] 
end

IrisHRT.finalize()

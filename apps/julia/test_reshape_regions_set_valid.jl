include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

A = reshape(collect(1:64), 8, 8)
A_dmem = IrisHRT.reshape(A, 8, 4, 2)
A_dmem1 = IrisHRT.reshape(A, 8, 4, 2)
A_dmem2 = IrisHRT.reshape(A, 8, 2, 4)

IrisHRT.enable_regions(A_dmem1, 2)
n_regions = IrisHRT.regions_count(A_dmem1)
IrisHRT.set_valid_regions(A_dmem1, 2)
valid_regions = IrisHRT.regions_count(A_dmem1)

@testset "reshape_regions_set_valid" begin
    @test A_dmem.uid == A_dmem1.uid
    @test A_dmem.uid != A_dmem2.uid
    @test n_regions == 4
    @test valid_regions == 2
end

IrisHRT.finalize()

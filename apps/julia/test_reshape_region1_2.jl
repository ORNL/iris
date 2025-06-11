include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

A = reshape(collect(1:16), 4, 4)
AA_reshape = reshape(A, 4, 2, 2)
println(Core.stdout, " Original A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " Original AA: ", AA_reshape, " ptr:", typeof(AA_reshape), " size:", size(AA_reshape))

AA = IrisHRT.reshape(A, 4, 2, 2)

IrisHRT.enable_regions(AA, 1)

ARs = IrisHRT.regions(AA)

+(ARs[2], ARs[2], ARs[2])

IrisHRT.synchronize()
println(Core.stdout, "ARs[1]: ", ARs[1]) 

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

A_host = IrisHRT.valid_host(A)
AA_host = IrisHRT.valid_host(AA)

ARs1_host = IrisHRT.valid_host(ARs[1])
ARs2_host = IrisHRT.valid_host(ARs[2])

@testset "reshape_region1_2" begin
    @test A_host == [1 5 9 13; 4 12 20 28; 3 7 11 15; 4 8 12 16]
    @test AA_host == [1 5; 4 12; 3 7; 4 8;;; 9 13; 20 28; 11 15; 12 16]
    @test ARs1_host == [1 9; 5 13]   
    @test ARs2_host == [4 20; 12 28]
end

IrisHRT.finalize()


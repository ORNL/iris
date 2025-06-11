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

IrisHRT.enable_regions(AA, 2)

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

@testset "reshape_region2_2" begin
    @test A_host == [1 10 9 26; 2 12 10 28; 3 14 11 30; 4 16 12 32]
    @test AA_host == [1 10; 2 12; 3 14; 4 16;;; 9 26; 10 28; 11 30; 12 32]
    @test ARs1_host == [1 9; 2 10; 3 11; 4 12]
    @test ARs2_host == [10 26; 12 28; 14 30; 16 32]
end

IrisHRT.finalize()


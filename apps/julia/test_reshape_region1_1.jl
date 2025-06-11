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

+(ARs[1], ARs[1], ARs[1])

IrisHRT.synchronize()
println(Core.stdout, "ARs[1]: ", ARs[1]) 

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

A_host = IrisHRT.valid_host(A)
AA_host = IrisHRT.valid_host(AA)

ARs1_host = IrisHRT.valid_host(ARs[1])
ARs2_host = IrisHRT.valid_host(ARs[2])

@testset "reshape_region1_1" begin
    @test A_host == [2 10 18 26; 2 6 10 14; 3 7 11 15; 4 8 12 16]
    @test AA_host == [2 10; 2 6; 3 7; 4 8;;; 18 26; 10 14; 11 15; 12 16]
    @test ARs1_host == [2 18; 10 26] 
    @test ARs2_host == [2 10; 6 14]
end

IrisHRT.finalize()


include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

A = reshape(collect(1:16), 4, 4)

AA = IrisHRT.reshape(A, 4, 2, 2)

IrisHRT.enable_regions(AA)

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

@testset "iris_max" begin
    @test A_host == [2 10 9 13; 4 12 10 14; 6 14 11 15; 8 16 12 16]
    @test AA_host == [2 10; 4 12; 6 14; 8 16;;; 9 13; 10 14; 11 15; 12 16]
    @test ARs1_host == [2 10; 4 12; 6 14; 8 16;;;] 
    @test ARs2_host == [9 13; 10 14; 11 15; 12 16;;;]
end

IrisHRT.finalize()


include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

A = reshape(collect(1:8), 4, 2)
println(Core.stdout, " Original A: ", A, " type:", typeof(A), " size:", size(A))

A_dmem = IrisHRT.dmem(A)
IrisHRT.enable_regions(A_dmem, 2)

ARs = IrisHRT.regions(A_dmem)

+(A_dmem, A_dmem, A_dmem)

IrisHRT.synchronize()
#println(Core.stdout, "ARs[1]: ", ARs[1]) 
#println(Core.stdout, "ARs[2]: ", ARs[2]) 


A_host = IrisHRT.valid_host(A_dmem)
IrisHRT.print_dmem(A_dmem)
println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))

ARs1_host = IrisHRT.valid_host(ARs[1])
ARs2_host = IrisHRT.valid_host(ARs[2])

IrisHRT.finalize()

@testset "reshape_region_2d" begin
    @test A_host ==  [2 10; 4 12; 6 14; 8 16] 
    @test ARs1_host == [2; 4; 6; 8;;]
    @test ARs2_host == [10; 12; 14; 16;;] 
end




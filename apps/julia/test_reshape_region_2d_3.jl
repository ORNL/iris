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


+(ARs[1], ARs[1], 1)
+(ARs[2], ARs[2], 1)

IrisHRT.synchronize()
#println(Core.stdout, "ARs[1]: ", ARs[1]) 
#println(Core.stdout, "ARs[2]: ", ARs[2]) 


#println(Core.stdout, "---------- Before flush Status ------------")
#IrisHRT.print_dmem(ARs[2])
#IrisHRT.print_dmem(A_dmem)
#println(Core.stdout, "---------- Before flush End of Status ------------")

A_host = IrisHRT.valid_host(A_dmem)
#println(Core.stdout, "---------- Status ------------")
#IrisHRT.print_dmem(ARs[2])
#IrisHRT.print_dmem(A_dmem)
#println(Core.stdout, "---------- End of Status ------------")
println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))

ARs1_host = IrisHRT.valid_host(ARs[1])
ARs2_host = IrisHRT.valid_host(ARs[2])

IrisHRT.finalize()

@testset "reshape_region_2d_3" begin
    @test A_host ==  [3 11; 5 13; 7 15; 9 17] 
    @test ARs1_host == [3; 5; 7; 9;;]
    @test ARs2_host == [11; 13; 15; 17;;] 
end




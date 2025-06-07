include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

C = reshape(collect(1:8), 4, 2)
IrisHRT.enable_regions(C, 1)
CRs = IrisHRT.regions(C)

+(CRs[2], CRs[2], CRs[2]);


D = reshape(collect(1:8), 4, 2)
D_dmem = IrisHRT.dmem(D)
IrisHRT.enable_regions(D_dmem, 2)

DRs = IrisHRT.regions(D_dmem)

+(D_dmem, D_dmem, D_dmem)

DB = zeros(Int64, 2,2)
copyto!(DB, DRs[2])

A = reshape(collect(1:8), 4, 2)
println(Core.stdout, " Original A: ", A, " type:", typeof(A), " size:", size(A))

A_dmem = IrisHRT.dmem(A)
IrisHRT.enable_regions(A_dmem, 2)

ARs = IrisHRT.regions(A_dmem)

+(A_dmem, A_dmem, A_dmem)


#+(ARs[1], ARs[1], 1)
+(ARs[2], ARs[2], 1)

B = zeros(Int64, 2,2)

#println(Core.stdout, "---------- Before copy Status ------------")
#IrisHRT.print_dmem(ARs[2])
#IrisHRT.print_dmem(A_dmem)
#println(Core.stdout, "---------- Before copy End of Status ------------")

copyto!(B, ARs[2])
IrisHRT.synchronize()
IrisHRT.flush(B)
IrisHRT.flush(DB)
#println(Core.stdout, "ARs[1]: ", ARs[1]) 
#println(Core.stdout, "ARs[2]: ", ARs[2]) 
println(Core.stdout, "CRs[1]: ", CRs[1]) 
println(Core.stdout, "CRs[2]: ", CRs[2]) 
println(Core.stdout, "DB: ", DB) 
println(Core.stdout, "B: ", B) 
println(Core.stdout, "A_dmem: ", A_dmem) 


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

CRs1_host = IrisHRT.valid_host(CRs[1])
CRs2_host = IrisHRT.valid_host(CRs[2])

IrisHRT.finalize()

@testset "reshape_region_copyto" begin
	@test CRs1_host == [1 5]
	@test CRs2_host == [4 12]
	@test DB == [10 14; 12 16]
	@test B == [11 15; 13 17]
    @test A_host ==  [2 11; 4 13; 6 15; 8 17] 
    @test ARs1_host == [2; 4; 6; 8;;]
    @test ARs2_host == [11; 13; 15; 17;;] 
end




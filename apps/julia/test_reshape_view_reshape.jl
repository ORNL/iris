include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

hA = reshape(collect(1:64), 8, 8)

hAA = IrisHRT.dmem(@view hA[3:6,3:6])

hAAA = IrisHRT.reshape(hAA, 2, 8)

hAAAA = IrisHRT.reshape(hAAA, 8, 2)

A = reshape(collect(1:64), 8, 8)

AA = IrisHRT.dmem(@view A[3:6,3:6])

AAA = IrisHRT.reshape(AA, 2, 8)

#RAAA = IrisHRT.reinterpret(Int8, AA)

AAAA = IrisHRT.reshape(AAA, 8, 2)


A_host = IrisHRT.valid_host(A)
AA_host = IrisHRT.valid_host(AA)
AAA_host = IrisHRT.valid_host(AAA)
AAAA_host = IrisHRT.valid_host(AAAA)

println(A_host)
println(AA_host)
println(AAA_host)
println(AAAA_host)

@testset "reshape_view_reshape" begin
	@test A_host == hA
	@test AA_host   == [19 27 35 43; 20 28 36 44; 21 29 37 45; 22 30 38 46]
	@test AAA_host  == [19 21 27 29 35 37 43 45; 20 22 28 30 36 38 44 46]
	@test AAAA_host == [19 35; 20 36; 21 37; 22 38; 27 43; 28 44; 29 45; 30 46]
end

IrisHRT.finalize()

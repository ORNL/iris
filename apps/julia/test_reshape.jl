include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

A = reshape(collect(1:16), 4, 4)

AA = IrisHRT.reshape(A, 4, 2, 2)

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

A_host = IrisHRT.valid_host(A)
AA_host = IrisHRT.valid_host(AA)

IrisHRT.print_dmem(A, " A_host:")
IrisHRT.print_dmem(AA, " AA_host:")

@testset "reshape" begin
    @test A_host == [1 5 9 13; 2 6 10 14; 3 7 11 15; 4 8 12 16]
    @test AA_host == [1 5; 2 6; 3 7; 4 8;;; 9 13; 10 14; 11 15; 12 16]
end

IrisHRT.finalize()

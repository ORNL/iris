include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(Core.stdout, host, " ptr:", pointer(host))
end


A = reshape(collect(1:16), 4, 4)
AA = reshape(A, 4, 2, 2)
AA = AA + AA + AA + AA + AA

println(Core.stdout, " A: ", A, " ptr:", pointer(A))
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))
tryme(A)
tryme(AA)
tryme(A)
tryme(AA)
tryme(@view AA[:,:,2])
tryme(@view AA[:,:,2])

println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA), " SubArray:", pointer(AA[:,:,2]))

DA = IrisHRT.linspace(Int64, 0, 1, 8)
IrisHRT.copyto!((@view AA[:,:,2]), DA; flush=true, submit=true, wait=true)

println(Core.stdout, " DA: ", DA)
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))

DB = [101, 102, 103, 104, 105, 106, 107, 108]
copyto!(AA[:, :, 2], DB)

println(Core.stdout, " DB: ", DB)
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))


@testset "copyto_test" begin
    @test AA == [5 25; 10 30; 15 35; 20 40;;; 0 4; 1 5; 2 6; 3 7]
end
IrisHRT.finalize()


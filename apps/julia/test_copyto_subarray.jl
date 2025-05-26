include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

A = reshape(collect(1:16), 4, 4)
AA = reshape(A, 4, 2, 2)

AA[1,1,1] = 45

println(Core.stdout, " A: ", A, " ptr:", pointer(A))
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))

DA = IrisHRT.linspace(Int64, 0, 1, 4, 2)
s = @view AA[:,:,2]
println(Core.stdout, " s: ", pointer(s))
IrisHRT.copyto!((@view AA[:,:,2]), DA; flush=true, submit=true, wait=true)
IrisHRT.flush(@view AA[:,:,2])
println(Core.stdout, "DA pointer: ", IrisHRT.uid(DA), " -- ", "----", IrisHRT.host_ptr(DA))
println(Core.stdout, "AA pointer: ", IrisHRT.uid(AA), " -- ", pointer(AA), "----", IrisHRT.host_ptr(AA))
println(Core.stdout, "AA sub pointer: ", IrisHRT.uid(AA[:,:,2]), " -- ", pointer(AA[:, :, 2]), "----", IrisHRT.host_ptr(AA[:,:,2]))
println(Core.stdout, "AA sub pointer: ", IrisHRT.uid(AA[:,:,2]), " -- ", pointer(AA[:, :, 2]), "----", IrisHRT.host_ptr(AA[:,:,2]))

DB = IrisHRT.linspace(Int64, 50, 1, 2, 2)
IrisHRT.copyto!((@view AA[2,:,:]), DB; flush=true, submit=true, wait=true)
IrisHRT.flush(@view AA[2,:,:])
println(Core.stdout, " DB: ", DB)
println(Core.stdout, " A: ", A, " ptr:", pointer(A))
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))

DC = IrisHRT.dmem(Int64, 2, 2)
IrisHRT.copyto!(DC, (@view AA[2,:,:]); flush=true, submit=true, wait=true)

println(Core.stdout, " DC: ", DC)
println(Core.stdout, " A: ", A, " ptr:", pointer(A))
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))

IrisHRT.copyto!((@view AA[1,:,:]), (@view AA[2,:,:]); flush=true, submit=true, wait=true)
IrisHRT.flush(@view AA[1,:,:])
println(Core.stdout, " A: ", A, " ptr:", pointer(A))
println(Core.stdout, " AA: ", AA, " ptr:", pointer(AA))

@testset "copyto_subarray" begin
    @test A == [50 51 52 53; 50 51 52 53; 3 7 2 6; 4 8 3 7]
    @test AA == [50 51; 50 51; 3 7; 4 8;;; 52 53; 52 53; 2 6; 3 7]
end
IrisHRT.finalize()

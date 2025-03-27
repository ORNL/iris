include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

A = reshape(collect(1:16), 4, 4)

AA = IrisHRT.reshape(A, 4, 2, 2)

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

IrisHRT.finalize()

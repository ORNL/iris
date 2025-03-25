include(ENV["IRIS"] * "/include/iris/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

function tryme(host::AbstractArray{T}) where T 
    println(host)
end

A = ones(Float64, 4, 4)
AA = IrisHRT.reshape(A, 16)

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

/(AA, AA, sqrt(24))

IrisHRT.flush(AA)

println(Core.stdout, " A: ", A, " type:", typeof(A), " size:", size(A))
println(Core.stdout, " AA: ", AA, " ptr:", typeof(AA), " size:", size(AA))

IrisHRT.finalize()

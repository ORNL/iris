include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

X = IrisHRT.ones(Float32, 8)
Y = IrisHRT.fill(Float32, 2, 8)
A = IrisHRT.ones(Float32, 8)

B = X * Y
Z = A + B

IrisHRT.wait(Z)
IrisHRT.flush(Z)
println("Z: ", Z)
IrisHRT.finalize()

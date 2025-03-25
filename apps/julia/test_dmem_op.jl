const iris_path = ENV["IRIS"]
ENV["LD_LIBRARY_PATH"] =  iris_path * "/lib64:" * iris_path * "/lib:" * ENV["LD_LIBRARY_PATH"]
const iris_jl = iris_path * "/include/iris/IrisHRT.jl"
include(iris_jl)

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

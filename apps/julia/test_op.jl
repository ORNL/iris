include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

op1 = IrisHRT.ones(Float64, 8)
op2 = IrisHRT.ones(Float64, 8)
out = IrisHRT.zeros(Float64, 8)
+(out, op1, op2)

println(Core.stdout, " op1:", op1, " op2:", op2, " out:", out)

IrisHRT.finalize()

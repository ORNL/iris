include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()
SIZE=8
r = IrisHRT.fill(Float64, 0.5, SIZE)
r_old = IrisHRT.dmem(Float64, SIZE)
#println(Core.stdout, " r: ", r[1:min(SIZE,10)])
IrisHRT.copyto!(r_old, r)
r_new = IrisHRT.copy(r_old)
println(Core.stdout, " r_old: ", r_old[1:min(SIZE,10)])
println(Core.stdout, " r_new", r_new[1:min(SIZE,10)])


IrisHRT.finalize()

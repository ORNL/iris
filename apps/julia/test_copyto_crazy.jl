include(ENV["IRIS"] * "/include/iris/IrisHRT.jl")
using .IrisHRT
IrisHRT.init()
SIZE=8
submit_wait_flag=false
r = IrisHRT.fill(Float64, 0.5, SIZE)
r_old = IrisHRT.dmem(Float64, SIZE)
#println(Core.stdout, " r: ", r[1:min(SIZE,10)])
IrisHRT.copyto(r_old, r; submit=submit_wait_flag, wait=submit_wait_flag, flush=submit_wait_flag)
r = r + r
IrisHRT.synchronize()
println(Core.stdout, " r:", r[1:min(SIZE,10)])
println(Core.stdout, " r_old: ", r_old[1:min(SIZE,10)])

IrisHRT.finalize()
